"""LoRA SFT training for the outbound-sales conversation model (Harmony-aware).

Reads `data/preprocessed.jsonl` (produced by 01_preprocessing) and fine-tunes the
base model with LoRA in bf16. Each training example is rendered with the native
GPT-OSS Harmony chat template — reasoning lives in the analysis channel, the
polished agent response in the final channel — so the analysis channel structure
is reinforced rather than starved.

Channel-aware loss (per-example per-channel-mean):
  - prefix tokens                 channel_mask 0 (masked, label=-100)
  - assistant framing tokens      channel_mask 0 (base model already knows Harmony)
  - analysis-channel content      channel_mask 1
  - final-channel content         channel_mask 2
  - padding                       channel_mask 0

For each example b:
    loss_a_b = mean(per_token_ce[channel_mask == 1])   # 0 if channel empty
    loss_f_b = mean(per_token_ce[channel_mask == 2])
    loss_b   = w_a * loss_a_b + w_f * loss_f_b
Batch loss = mean(loss_b across examples).

The per-channel-mean formulation neutralizes length asymmetry: analysis is
typically 3–10× longer than final, so naive per-token loss would be dominated
by analysis. Equal channel weight by construction.

Customer-level 90/10 train/val split (configurable column name) — no leakage:
a single customer may appear across multiple calls, so call- or row-level
splits leak context.

Launch:
    # Single-GPU LoRA smoke test (no launcher needed):
    python 02_sft_training.py --config 02_train_config.yaml --max-samples 20 --epochs 1

    # Multi-GPU production run via `accelerate launch` (FSDP or DeepSpeed-ZeRO-3,
    # selected by the accelerate config file). Per cluster policy this is the
    # only supported multi-GPU launcher — do NOT call `deepspeed` directly.
    accelerate launch --config_file <accelerate_config.yaml> 02_sft_training.py \\
        --config 02_train_config.yaml
    # Or override the config's distributed type at the CLI:
    accelerate launch --config_file <accelerate_config.yaml> --use_fsdp \\
        02_sft_training.py --config 02_train_config.yaml

The script does not configure DeepSpeed/FSDP itself — Trainer auto-picks up the
accelerate plugin from the launch environment. For DeepSpeed-ZeRO-3 specifically
the accelerate config should set `zero3_init_flag: true` so parameters are
partitioned at `from_pretrained` time (equivalent to the old HfDeepSpeedConfig
hook); without that flag every rank tries to materialize the full model on
cuda:0 and OOMs.
"""

import argparse
import importlib.util
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

# Reuse Stage 01 helpers (system prompt, turn parsing, windowing).
_spec = importlib.util.spec_from_file_location(
    "preproc01", str(Path(__file__).resolve().parent / "01_preprocessing.py")
)
preproc01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preproc01)

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_DATA_FILE = "data/preprocessed.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints"
DEFAULT_MAX_SEQ_LEN = 4096

# Harmony framing — verified against gpt_oss_inspection.ipynb cell 2.
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FINAL_HEADER = "<|channel|>final<|message|>"
ANALYSIS_END = "<|end|>"
ASSISTANT_RESTART = "<|start|>assistant"
FINAL_END = "<|return|>"


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
    history_summary_field: str = "history_summary"
    reasoning_field: str = "reasoning"
    polished_target_field: str = "polished_target"
    context_field: str = "context"
    substantial_field: str = "is_substantial"

    # tokenization
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    window_size: int = 50  # max transcript turns; 0 disables windowing
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
    # Common per-run overrides — leave None so we know whether the user passed them.
    p.add_argument("--input", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--analysis-weight", type=float, default=None,
                   help="Weight on analysis-channel mean loss")
    p.add_argument("--final-weight", type=float, default=None,
                   help="Weight on final-channel mean loss")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke test: use only first N substantial examples (before split)")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None,
                   help="Keep only the last N turns of transcript. Pass 0 to disable windowing.")
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
        "window_size": args.window_size,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def is_distributed_launch() -> bool:
    """True when launched under `accelerate launch` (FSDP or DeepSpeed) or `torchrun`.

    Checked via env vars set by the launcher: LOCAL_RANK is set by torchrun and by
    accelerate in any multi-process config; the ACCELERATE_USE_* flags catch the
    single-process accelerate case where we still want to skip device_map.
    """
    if "LOCAL_RANK" in os.environ:
        return True
    return any(os.environ.get(k, "").lower() == "true"
               for k in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"))


# ============================================================================
# TOKENIZATION (manual span concat — channel boundaries known by construction)
# ============================================================================

def render_prefix(tokenizer, transcript: str, history_summary: str, context: str = "") -> str:
    """Render system + user message ending at the assistant generation prompt."""
    system_content = preproc01.build_agent_system_content(history_summary, context=context)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"transcript:\n{transcript}"},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def fit_prefix_to_budget(tokenizer, transcript: str, history_summary: str, context: str,
                         suffix_len: int, max_seq_len: int,
                         window_size: int | None) -> tuple[list[int], int]:
    """Window then drop oldest turns until prefix + suffix fits in max_seq_len.

    The system prompt and context block are NEVER truncated — only the transcript
    loses its earliest turns. Returns (prefix_ids, n_turns_kept).
    """
    turns = preproc01.parse_turns(transcript)
    if window_size is not None and len(turns) > window_size:
        turns = turns[-window_size:]

    while True:
        windowed = preproc01.render_turns(turns) if turns else ""
        prefix_text = render_prefix(tokenizer, windowed, history_summary, context)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) + suffix_len <= max_seq_len:
            return prefix_ids, len(turns)
        if not turns:
            return prefix_ids, 0
        turns = turns[1:]


def encode_with_spans(tokenizer, transcript: str, history_summary: str,
                      reasoning: str, polished_target: str, context: str,
                      max_seq_len: int, window_size: int | None) -> dict:
    """Tokenize a Harmony-rendered example and return the assistant-turn segments.

    Each segment is tagged with its kind ∈ {analysis, final, framing} so the caller
    can build a per-token channel_mask without re-parsing the tokenized output.
    """
    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    a_header = enc(ANALYSIS_HEADER)
    a_content = enc(reasoning)
    a_end = enc(ANALYSIS_END)
    restart = enc(ASSISTANT_RESTART)
    f_header = enc(FINAL_HEADER)
    f_content = enc(polished_target)
    f_end = enc(FINAL_END)
    suffix_len = sum(len(s) for s in (a_header, a_content, a_end, restart, f_header, f_content, f_end))

    prefix_ids, n_turns_kept = fit_prefix_to_budget(
        tokenizer, transcript, history_summary, context, suffix_len, max_seq_len, window_size
    )

    return {
        "prefix_ids": prefix_ids,
        "n_turns_kept": n_turns_kept,
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
    n_window_dropped = 0
    n_budget_dropped = 0
    n_overflow = 0

    for ex in examples:
        original_n_turns = len(preproc01.parse_turns(ex[cfg.transcript_field]))
        encoded = encode_with_spans(
            tokenizer,
            transcript=ex[cfg.transcript_field],
            history_summary=ex[cfg.history_summary_field],
            reasoning=ex[cfg.reasoning_field],
            polished_target=ex[cfg.polished_target_field],
            context=ex.get(cfg.context_field, ""),
            max_seq_len=cfg.max_seq_len,
            window_size=cfg.window_size if cfg.window_size > 0 else None,
        )
        prefix_ids = encoded["prefix_ids"]
        segments = encoded["segments"]
        n_kept = encoded["n_turns_kept"]
        suffix_len = sum(len(s[1]) for s in segments)

        if cfg.window_size > 0 and original_n_turns > cfg.window_size:
            n_window_dropped += 1
        post_window = (
            min(original_n_turns, cfg.window_size) if cfg.window_size > 0 else original_n_turns
        )
        if n_kept < post_window:
            n_budget_dropped += 1
        if len(prefix_ids) + suffix_len > cfg.max_seq_len:
            n_overflow += 1

        # Assemble token IDs and channel_mask.
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

        # Labels mirror input_ids, masked to -100 where channel_mask == 0.
        labels = [(tid if cm != 0 else -100) for tid, cm in zip(full_ids, channel_mask)]

        if len(full_ids) > cfg.max_seq_len:
            full_ids = full_ids[:cfg.max_seq_len]
            labels = labels[:cfg.max_seq_len]
            channel_mask = channel_mask[:cfg.max_seq_len]
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

    logging.info("Examples windowed (>%s turns): %d", cfg.window_size, n_window_dropped)
    logging.info("Examples that lost extra turns to token budget: %d", n_budget_dropped)
    if n_overflow:
        logging.warning(
            "Examples overflowing max_seq_len even with empty transcript: %d "
            "(system+context+suffix alone don't fit — increase max_seq_len)", n_overflow,
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

        # Shift for next-token prediction.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_channel = channel_mask[..., 1:].contiguous()

        B, Sm1, V = shift_logits.shape
        # Per-token CE; -100 positions contribute 0 (and we won't read them anyway).
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

        # Track diagnostics. Eval-mode batches accumulate so evaluate() can emit
        # the average; train-mode just stashes the latest for log() to inject.
        if model.training:
            self._last_loss_a = float(avg_a.detach().float().item())
            self._last_loss_f = float(avg_f.detach().float().item())
        else:
            self._eval_loss_a_sum += float(avg_a.detach().float().item())
            self._eval_loss_f_sum += float(avg_f.detach().float().item())
            self._eval_count += 1

        return (total, outputs) if return_outputs else total

    def log(self, logs: dict, *args, **kwargs):
        # Inject train-step per-channel losses. Skip eval logs (handled in evaluate).
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
            # Re-log so tensorboard sees the per-channel eval metrics.
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
            row = json.loads(line)
            if not row.get(cfg.substantial_field, True):
                continue
            raw.append(row)
    logging.info("Loaded %d substantial examples from %s", len(raw), cfg.input)

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

    # Sanity print on train example 0.
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

    # Snapshot resolved config for reproducibility.
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
