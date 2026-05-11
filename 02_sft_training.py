"""LoRA SFT training for the outbound-sales conversation model (Harmony-aware).

Each training example is rendered with the native GPT-OSS Harmony chat
template — reasoning lives in the analysis channel, the polished agent
response in the final channel.

Input fields read from each JSONL row:
    1. transcript                — current conversation transcript up to the objection
    2. previous_call_summary     — summary of prior calls with this customer
    3. current_call_summary      — summary of the current call so far
    4. customer context          — built via get_customer_context(row) (PLACEHOLDER)
    5. objection_summary         — summary of the current objection being handled

Targets (assistant turn):
    - reasoning                  — analysis channel (teacher CoT)
    - polished_target            — final channel (polished agent response)

Channel-aware loss (per-example per-channel-mean):
  - prefix tokens                 channel_mask 0 (masked, label=-100)
  - assistant framing tokens      channel_mask 0 (base model already knows Harmony)
  - analysis-channel content      channel_mask 1
  - final-channel content         channel_mask 2
  - padding                       channel_mask 0

For each example b:
    loss_a_b = mean(per_token_ce[channel_mask == 1])
    loss_f_b = mean(per_token_ce[channel_mask == 2])
    loss_b   = w_a * loss_a_b + w_f * loss_f_b
Batch loss = mean(loss_b across examples).

Transcript truncation: simple char-based — keep at most
`max_transcript_chars` (default 12000, ~2000 words) from the END of the
transcript. If the tokenized prefix+suffix still overflows `max_seq_len`,
tokens are dropped from the FRONT of the prefix so the suffix (which
contains the SFT targets) is never lost.

Customer-level 90/10 train/val split — no leakage: a single customer may
appear across multiple calls, so call- or row-level splits leak context.

Launch:
    # Single-GPU LoRA smoke test (no launcher needed):
    python 02_sft_training.py --config 02_train_config.yaml --max-samples 20 --epochs 1

    # Multi-GPU production run via `accelerate launch` (FSDP or DeepSpeed-ZeRO-3).
    accelerate launch --config_file <accelerate_config.yaml> 02_sft_training.py \\
        --config 02_train_config.yaml
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_DATA_FILE = "data/preprocessed.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints"
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_TRANSCRIPT_CHARS = 12000  # ~2000 words; tail of the transcript

# Harmony framing — verified against gpt_oss_inspection.ipynb cell 2.
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FINAL_HEADER = "<|channel|>final<|message|>"
ANALYSIS_END = "<|end|>"
ASSISTANT_RESTART = "<|start|>assistant"
FINAL_END = "<|return|>"


# ============================================================================
# PROMPT TEMPLATES — edit these to change what the model sees at train/inference
# ============================================================================

SYSTEM_CONTENT = """You are an outbound sales agent on the TSE channel. You will be given
the customer's context, a summary of prior calls with this customer, a summary of how the
current call has gone so far, the current objection the customer just raised, and the
verbatim transcript of the current call up to the objection.

Reason about how to handle this objection in the analysis channel, then produce the agent
turn you would say next in the final channel. In analysis, reason in first person present
tense as the agent would think silently between hearing the objection and choosing what to
say — flowing natural prose, no bullets, no headers, no enumerated steps. Reason as much as
the case warrants and no more. In the final channel, produce ONE natural agent turn — what
you would actually say next, in the words you would speak."""

USER_TMPL = """CUSTOMER CONTEXT:
{customer_context}

PREVIOUS CALL SUMMARY:
{previous_call_summary}

CURRENT CALL SUMMARY:
{current_call_summary}

CURRENT OBJECTION:
{objection_summary}

CURRENT TRANSCRIPT (up to objection):
{transcript}"""


def get_customer_context(row: dict) -> str:
    """PLACEHOLDER — return the customer-context text block for a training row.

    Customize this to match your data schema. For example, if the row has
    columns like 'sic4_industry', 'rev_tier', 'emp_ct':

        return (
            f"Industry (SIC4): {row.get('sic4_industry', 'unknown')}\\n"
            f"Revenue tier:    {row.get('rev_tier', 'unknown')}\\n"
            f"Employee count:  {row.get('emp_ct', 'unknown')}"
        )

    Or, if you've already pre-rendered a text block into a single column:

        return row.get("customer_context", "")

    The default below just reads a 'customer_context' field if present and
    falls back to an empty string, so an unmodified script won't crash on
    a missing schema — but YOU SHOULD REPLACE THIS with your real fetcher.
    """
    # TODO(Yi): replace with the real customer-context construction for your dataset
    return row.get("customer_context", "") or ""


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    """All knobs for one training run. Override via YAML or CLI."""

    # paths
    model_dir: str = DEFAULT_MODEL_DIR
    input: str = DEFAULT_DATA_FILE
    output_dir: str = DEFAULT_OUTPUT_DIR

    # data field names (must match the JSONL produced by your preproc glue step)
    customer_id_field: str = "customer_id"  # falls back to "cust_id" if missing
    transcript_field: str = "transcript"
    previous_call_summary_field: str = "previous_call_summary"
    current_call_summary_field: str = "current_call_summary"
    objection_summary_field: str = "objection_summary"
    reasoning_field: str = "reasoning"           # analysis-channel target (teacher CoT)
    polished_target_field: str = "polished_target"  # final-channel target

    # tokenization
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS  # tail-of-transcript cap
    max_samples: int | None = None  # smoke-test cap

    # split
    val_fraction: float = 0.10
    split_seed: int = 42

    # channel loss weights (per-example per-channel-mean formulation)
    w_analysis: float = 1.0
    w_final: float = 1.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # optimization
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch_fused"

    # checkpointing & logging
    save_steps: int = 30
    save_total_limit: int = 10
    eval_steps: int = 30
    logging_steps: int = 5
    report_to: str = "tensorboard"

    # misc
    seed: int = 42
    bf16: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config (TrainConfig fields). CLI flags override.")
    p.add_argument("--input", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--analysis-weight", type=float, default=None,
                   help="Weight on analysis-channel mean loss")
    p.add_argument("--final-weight", type=float, default=None,
                   help="Weight on final-channel mean loss")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke test: use only first N examples (before split)")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--max-transcript-chars", type=int, default=None,
                   help="Keep at most this many characters from the END of the transcript.")
    # Absorb the local-rank flag forwarded by accelerate / torchrun.
    p.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    """Load YAML (if any), then layer non-None CLI args on top."""
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    overrides = {
        "input": args.input,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "w_analysis": args.analysis_weight,
        "w_final": args.final_weight,
        "max_samples": args.max_samples,
        "max_seq_len": args.max_seq_len,
        "max_transcript_chars": args.max_transcript_chars,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def is_distributed_launch() -> bool:
    """True when launched under `accelerate launch` (FSDP or DeepSpeed) or `torchrun`."""
    if "LOCAL_RANK" in os.environ:
        return True
    return any(os.environ.get(k, "").lower() == "true"
               for k in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"))


# ============================================================================
# TOKENIZATION (manual span concat — channel boundaries known by construction)
# ============================================================================

def trim_transcript(transcript: str, max_chars: int) -> tuple[str, bool]:
    """Keep at most `max_chars` characters from the END of the transcript.

    Returns (trimmed, was_truncated). If trimming would cut mid-line, the
    leading partial line is dropped so the kept text starts at a clean newline.
    """
    if len(transcript) <= max_chars:
        return transcript, False
    trimmed = transcript[-max_chars:]
    # Drop the leading partial line if there's a newline within the kept slice,
    # so the prompt sees clean turns.
    nl = trimmed.find("\n")
    if 0 <= nl < len(trimmed) - 1:
        trimmed = trimmed[nl + 1:]
    return trimmed, True


def build_prefix_text(tokenizer, row: dict, cfg: "TrainConfig",
                      truncate_counter: list[int]) -> str:
    """Render system + user message ending at the assistant generation prompt."""
    transcript, was_truncated = trim_transcript(
        row[cfg.transcript_field], cfg.max_transcript_chars
    )
    if was_truncated:
        truncate_counter[0] += 1

    user_text = USER_TMPL.format(
        customer_context=get_customer_context(row),
        previous_call_summary=row[cfg.previous_call_summary_field],
        current_call_summary=row[cfg.current_call_summary_field],
        objection_summary=row[cfg.objection_summary_field],
        transcript=transcript,
    )
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def encode_with_spans(tokenizer, row: dict, cfg: "TrainConfig",
                      truncate_counter: list[int]) -> dict:
    """Tokenize one example into prefix + tagged assistant-turn segments."""
    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    a_header = enc(ANALYSIS_HEADER)
    a_content = enc(row[cfg.reasoning_field])
    a_end = enc(ANALYSIS_END)
    restart = enc(ASSISTANT_RESTART)
    f_header = enc(FINAL_HEADER)
    f_content = enc(row[cfg.polished_target_field])
    f_end = enc(FINAL_END)

    prefix_text = build_prefix_text(tokenizer, row, cfg, truncate_counter)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    return {
        "prefix_ids": prefix_ids,
        "segments": [
            ("a_header", a_header, "framing"),
            ("a_content", a_content, "analysis"),
            ("a_end", a_end, "framing"),
            ("restart", restart, "framing"),
            ("f_header", f_header, "framing"),
            ("f_content", f_content, "final"),
            ("f_end", f_end, "framing"),
        ],
    }


# ============================================================================
# DATA LOADING + CUSTOMER-LEVEL SPLIT
# ============================================================================

def _resolve_customer_id(example: dict, primary: str) -> str:
    """Try primary field, fall back to 'cust_id'. Raise if neither present."""
    if primary in example and example[primary] is not None:
        return str(example[primary])
    if "cust_id" in example and example["cust_id"] is not None:
        return str(example["cust_id"])
    raise ValueError(
        f"Example has no '{primary}' or 'cust_id' field. Customer-level split "
        f"requires a customer identifier on every row. Update your preprocessing "
        f"glue or set customer_id_field in TrainConfig to match your column."
    )


def customer_level_split(
    examples: list[dict],
    customer_id_field: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split examples 90/10 by customer ID. No customer appears in both sets."""
    ids = [_resolve_customer_id(ex, customer_id_field) for ex in examples]
    unique = sorted(set(ids))
    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_customers = set(shuffled[:n_val])

    train_examples, val_examples = [], []
    for ex, cid in zip(examples, ids):
        (val_examples if cid in val_customers else train_examples).append(ex)

    logging.info(
        "Customer-level split: %d unique customers → %d val (%d rows) / %d train (%d rows)",
        len(unique), n_val, len(val_examples),
        len(unique) - n_val, len(train_examples),
    )
    if len(val_examples) == 0:
        raise ValueError(
            f"Val set is empty after split (val_fraction={val_fraction}). "
            f"Increase val_fraction or add more customers."
        )
    return train_examples, val_examples


def build_dataset(examples: list[dict], tokenizer, cfg: TrainConfig) -> Dataset:
    """Tokenize all examples and pack into a HF Dataset with channel_mask."""
    input_ids_list, labels_list, attention_list, channel_mask_list = [], [], [], []
    truncate_counter = [0]
    n_prefix_trimmed = 0

    for ex in examples:
        encoded = encode_with_spans(tokenizer, ex, cfg, truncate_counter)
        prefix_ids = encoded["prefix_ids"]
        segments = encoded["segments"]

        # Assemble token IDs and channel_mask in lockstep.
        full_ids = list(prefix_ids)
        channel_mask = [0] * len(prefix_ids)
        for _name, ids, kind in segments:
            full_ids.extend(ids)
            if kind == "analysis":
                channel_mask.extend([1] * len(ids))
            elif kind == "final":
                channel_mask.extend([2] * len(ids))
            else:  # framing — base model already knows Harmony, don't train on it
                channel_mask.extend([0] * len(ids))

        # If overflow, drop tokens from the FRONT of the prefix so the suffix
        # (containing the SFT targets) is never lost.
        if len(full_ids) > cfg.max_seq_len:
            excess = len(full_ids) - cfg.max_seq_len
            full_ids = full_ids[excess:]
            channel_mask = channel_mask[excess:]
            n_prefix_trimmed += 1

        labels = [(tid if cm != 0 else -100) for tid, cm in zip(full_ids, channel_mask)]

        attention = [1] * len(full_ids)
        pad = cfg.max_seq_len - len(full_ids)
        if pad > 0:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            full_ids = full_ids + [pad_id] * pad
            labels = labels + [-100] * pad
            attention = attention + [0] * pad
            channel_mask = channel_mask + [0] * pad

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_list.append(attention)
        channel_mask_list.append(channel_mask)

    logging.info(
        "Transcript char-trimmed: %d / %d examples (max_transcript_chars=%d)",
        truncate_counter[0], len(examples), cfg.max_transcript_chars,
    )
    if n_prefix_trimmed:
        logging.warning(
            "Token-overflow prefix-trim: %d examples lost prefix tokens to fit "
            "max_seq_len=%d. Lower max_transcript_chars or raise max_seq_len.",
            n_prefix_trimmed, cfg.max_seq_len,
        )
    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_list,
        "channel_mask": channel_mask_list,
    })


# ============================================================================
# CHANNEL-AWARE TRAINER
# ============================================================================

class ChannelMeanLossTrainer(Trainer):
    """Trainer with per-example per-channel-mean loss.

    For each example: loss_a = mean(ce[mask==1]), loss_f = mean(ce[mask==2]),
    loss = w_a*loss_a + w_f*loss_f. Batch loss = mean across examples.

    Per-channel diagnostics are logged at every step (train) and at every eval.
    """

    def __init__(self, w_a: float = 1.0, w_f: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.w_a = w_a
        self.w_f = w_f
        self._last_loss_a: float | None = None
        self._last_loss_f: float | None = None
        self._eval_loss_a_sum = 0.0
        self._eval_loss_f_sum = 0.0
        self._eval_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        channel_mask = inputs.pop("channel_mask")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_channel = channel_mask[..., 1:].contiguous()

        B, Sm1, V = shift_logits.shape
        per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Sm1)

        zero = torch.zeros((), device=per_token.device, dtype=per_token.dtype)
        per_example_total = []
        per_example_a = []
        per_example_f = []
        for b in range(B):
            ch_b = shift_channel[b]
            ce_b = per_token[b]
            mask_a = (ch_b == 1)
            mask_f = (ch_b == 2)
            la = ce_b[mask_a].mean() if mask_a.any() else zero
            lf = ce_b[mask_f].mean() if mask_f.any() else zero
            per_example_total.append(self.w_a * la + self.w_f * lf)
            per_example_a.append(la)
            per_example_f.append(lf)

        total = torch.stack(per_example_total).mean()
        avg_a = torch.stack(per_example_a).mean()
        avg_f = torch.stack(per_example_f).mean()

        if model.training:
            self._last_loss_a = float(avg_a.detach().float().item())
            self._last_loss_f = float(avg_f.detach().float().item())
        else:
            self._eval_loss_a_sum += float(avg_a.detach().float().item())
            self._eval_loss_f_sum += float(avg_f.detach().float().item())
            self._eval_count += 1

        return (total, outputs) if return_outputs else total

    def log(self, logs: dict, *args, **kwargs):
        if "loss" in logs and "eval_loss" not in logs:
            if self._last_loss_a is not None:
                logs["loss_a"] = self._last_loss_a
            if self._last_loss_f is not None:
                logs["loss_f"] = self._last_loss_f
        super().log(logs, *args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_loss_a_sum = 0.0
        self._eval_loss_f_sum = 0.0
        self._eval_count = 0
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self._eval_count > 0:
            extras = {
                f"{metric_key_prefix}_loss_a": self._eval_loss_a_sum / self._eval_count,
                f"{metric_key_prefix}_loss_f": self._eval_loss_f_sum / self._eval_count,
            }
            metrics.update(extras)
            self.log(extras)
        return metrics


# ============================================================================
# MODEL
# ============================================================================

def build_model(model_dir: str, distributed: bool):
    """Load the base model.

    Under a distributed launcher (accelerate FSDP or DeepSpeed-ZeRO-3) we must NOT
    pass `device_map="auto"` — it conflicts with FSDP wrapping and bypasses
    `deepspeed.zero.Init` partitioning.
    """
    kwargs = dict(local_files_only=True, dtype="auto")
    if not distributed:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if not distributed:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    logging.info("Base model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = resolve_config(args)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distributed = is_distributed_launch()

    logging.info("Resolved config: %s", cfg)
    logging.info("Distributed launcher detected: %s", distributed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = []
    with open(cfg.input, "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    logging.info("Loaded %d examples from %s", len(raw), cfg.input)

    if cfg.max_samples is not None:
        raw = raw[:cfg.max_samples]
        logging.info("Smoke-test mode: trimmed to %d examples", len(raw))

    train_raw, val_raw = customer_level_split(
        raw, cfg.customer_id_field, cfg.val_fraction, cfg.split_seed,
    )

    train_ds = build_dataset(train_raw, tokenizer, cfg)
    val_ds = build_dataset(val_raw, tokenizer, cfg)
    logging.info(
        "Datasets built: train=%d, val=%d, padded to %d tokens",
        len(train_ds), len(val_ds), cfg.max_seq_len,
    )

    first = train_ds[0]
    cm = torch.tensor(first["channel_mask"])
    logging.info(
        "Sanity train_ds[0]: masked=%d, analysis=%d, final=%d",
        int((cm == 0).sum()), int((cm == 1).sum()), int((cm == 2).sum()),
    )

    model = build_model(cfg.model_dir, distributed=distributed)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=False,
        bf16=cfg.bf16,
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        logging_dir=str(output_dir / "runs"),
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=False,  # channel_mask must survive
        dataloader_pin_memory=False,
    )

    trainer = ChannelMeanLossTrainer(
        w_a=cfg.w_analysis,
        w_f=cfg.w_final,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if trainer.is_world_process_zero():
        with open(output_dir / "train_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    logging.info("Starting training. TensorBoard: tensorboard --logdir %s", output_dir / "runs")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    logging.info("Training finished in %.1fs (%.2f min)", elapsed, elapsed / 60.0)

    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_path))
        log_path = output_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logging.info("Adapter + tokenizer saved to %s", adapter_path)
        logging.info("Step-level log written to %s", log_path)


if __name__ == "__main__":
    main()
