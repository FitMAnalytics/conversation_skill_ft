"""LoRA SFT training for the outbound-sales conversation model — TEACHER-COT VARIANT.

Builds on `02_sft_training_nocot_0511.py` (the version that runs in the Amex
environment) by adding analysis-channel supervision (teacher CoT). Prompt
templates live in YAML so prompts can be iterated without editing Python.

What is read from each JSONL row (field names are YAML-configurable):
  - transcript            — current conversation transcript up to the latest customer turn
  - previous_calls_summary — summary of prior calls (toggleable via YAML)
  - current_call_summary  — summary of the current call so far
  - customer_context      — pre-rendered customer + product context block

Targets (assistant turn):
  - analysis channel content — teacher CoT (e.g. JSONL field `teacher_cot`)
  - final channel content    — polished agent response (e.g. `polished_response`)

Channel-aware loss (per-example per-channel-mean):
    loss_a_b = mean(per_token_ce[channel_mask == 1])
    loss_f_b = mean(per_token_ce[channel_mask == 2])
    loss_b   = w_a * loss_a_b + w_f * loss_f_b
    batch loss = mean(loss_b across examples)         # .mean() is present

Defaults `w_analysis = w_final = 0.5` give a literal half/half total whose
magnitude is comparable to a single-channel loss.

System prompt routing on gpt-oss: the Harmony chat template re-routes the
`{"role": "system"}` content into the `developer` block under `# Instructions`.
You still author `system_prompt` in YAML — no need to write `{"role": "developer"}`
explicitly. The first rendered prompt is logged at startup so you can verify.

Launch:
    # Single-GPU LoRA smoke test:
    python 02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml \\
        --max-samples 20 --epochs 1

    # Multi-GPU FSDP (or DeepSpeed-ZeRO-3):
    accelerate launch --config_file <accelerate_config.yaml> \\
        02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-20b"
DEFAULT_DATA_FILE = "data/preprocessed.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints_cot_0520"
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_TRANSCRIPT_CHARS = 12000  # ~2000 words; tail of the transcript

# Harmony framing — verified against gpt_oss_inspection.ipynb cell 2.
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FINAL_HEADER = "<|channel|>final<|message|>"
ANALYSIS_END = "<|end|>"
ASSISTANT_RESTART = "<|start|>assistant"
FINAL_END = "<|return|>"

# Markers used inside `user_prompt_template` to delimit the optional
# previous-calls-summary section. When `include_previous_call_summary` is
# False, the entire block (markers + body) is removed before format-substitution.
PREV_OPEN_MARKER = "<<PREV_CALL_SECTION>>"
PREV_CLOSE_MARKER = "<<END>>"
_PREV_BLOCK_RE = re.compile(
    re.escape(PREV_OPEN_MARKER) + r"\n(.*?)" + re.escape(PREV_CLOSE_MARKER) + r"\n?\n?",
    re.DOTALL,
)
_PREV_OPEN_LINE_RE = re.compile(re.escape(PREV_OPEN_MARKER) + r"\n")
_PREV_CLOSE_LINE_RE = re.compile(re.escape(PREV_CLOSE_MARKER) + r"\n")


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

    # prompt templates (NEW — moved from Python into YAML)
    system_prompt: str = ""
    user_prompt_template: str = ""
    include_previous_call_summary: bool = True

    # data field names (must match the JSONL produced by your preproc glue step)
    transcript_field: str = "transcript"
    previous_call_summary_field: str = "previous_calls_summary"
    current_call_summary_field: str = "current_call_summary"
    customer_context_field: str = "customer_context"
    analysis_target_field: str = "teacher_cot"        # analysis-channel target
    final_target_field: str = "polished_response"     # final-channel target

    # tokenization
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS
    max_samples: int | None = None

    # channel loss weights (literal half/half by default)
    w_analysis: float = 0.5
    w_final: float = 0.5

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
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch_fused"

    # checkpointing & logging
    save_steps: int = 30
    save_total_limit: int = 10
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
                   help="Smoke test: use only first N examples")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--max-transcript-chars", type=int, default=None)
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
    """Keep at most `max_chars` characters from the END of the transcript."""
    if len(transcript) <= max_chars:
        return transcript, False
    trimmed = transcript[-max_chars:]
    nl = trimmed.find("\n")
    if 0 <= nl < len(trimmed) - 1:
        trimmed = trimmed[nl + 1:]
    return trimmed, True


def render_user_prompt(cfg: "TrainConfig", row: dict, transcript: str) -> str:
    """Render `cfg.user_prompt_template` against the row, honoring the prev-call toggle.

    Two-pass rendering:
      1. Strip or unwrap the <<PREV_CALL_SECTION>>...<<END>> block depending on
         `include_previous_call_summary`.
      2. format-substitute the surviving placeholders.
    """
    tmpl = cfg.user_prompt_template
    fields = {
        "customer_context": row.get(cfg.customer_context_field, "") or "",
        "current_call_summary": row[cfg.current_call_summary_field],
        "transcript": transcript,
    }
    if cfg.include_previous_call_summary:
        # Keep the body, strip just the marker lines.
        tmpl = _PREV_OPEN_LINE_RE.sub("", tmpl)
        tmpl = _PREV_CLOSE_LINE_RE.sub("", tmpl)
        fields["previous_calls_summary"] = row[cfg.previous_call_summary_field]
    else:
        # Drop the entire block (markers + body).
        tmpl = _PREV_BLOCK_RE.sub("", tmpl)
    return tmpl.format(**fields)


def build_prefix_text(tokenizer, row: dict, cfg: "TrainConfig",
                      truncate_counter: list[int]) -> str:
    """Render system + user message ending at the assistant generation prompt."""
    transcript, was_truncated = trim_transcript(
        row[cfg.transcript_field], cfg.max_transcript_chars
    )
    if was_truncated:
        truncate_counter[0] += 1

    user_text = render_user_prompt(cfg, row, transcript)
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def encode_with_spans(tokenizer, row: dict, cfg: "TrainConfig",
                      truncate_counter: list[int]) -> dict:
    """Tokenize one example into prefix + tagged assistant-turn segments."""
    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    a_header = enc(ANALYSIS_HEADER)
    a_content = enc(row[cfg.analysis_target_field])
    a_end = enc(ANALYSIS_END)
    restart = enc(ASSISTANT_RESTART)
    f_header = enc(FINAL_HEADER)
    f_content = enc(row[cfg.final_target_field])
    f_end = enc(FINAL_END)

    prefix_text = build_prefix_text(tokenizer, row, cfg, truncate_counter)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    return {
        "prefix_ids": prefix_ids,
        "prefix_text": prefix_text,
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
# DATA LOADING
# ============================================================================

def build_dataset(examples: list[dict], tokenizer, cfg: TrainConfig,
                  first_prefix_holder: list[str] | None = None) -> Dataset:
    """Tokenize all examples and pack into a HF Dataset with channel_mask.

    If `first_prefix_holder` is supplied, the first example's rendered prefix
    text is stashed there so the caller can log it for sanity-checking.
    """
    input_ids_list, labels_list, attention_list, channel_mask_list = [], [], [], []
    truncate_counter = [0]
    n_prefix_trimmed = 0

    for idx, ex in enumerate(examples):
        encoded = encode_with_spans(tokenizer, ex, cfg, truncate_counter)
        if first_prefix_holder is not None and idx == 0:
            first_prefix_holder.append(encoded["prefix_text"])
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

    For each example:
        loss_a = mean(ce[mask == 1])
        loss_f = mean(ce[mask == 2])
        loss   = w_a * loss_a + w_f * loss_f
    Batch loss = mean across examples. Per-channel diagnostics (`loss_a`,
    `loss_f`) logged at every training step.
    """

    def __init__(self, w_a: float = 0.5, w_f: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.w_a = w_a
        self.w_f = w_f
        self._last_loss_a: float | None = None
        self._last_loss_f: float | None = None

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

        return (total, outputs) if return_outputs else total

    def log(self, logs: dict, *args, **kwargs):
        if "loss" in logs:
            if self._last_loss_a is not None:
                logs["loss_a"] = self._last_loss_a
            if self._last_loss_f is not None:
                logs["loss_f"] = self._last_loss_f
        super().log(logs, *args, **kwargs)


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

    if not cfg.system_prompt or not cfg.user_prompt_template:
        raise ValueError(
            "system_prompt and user_prompt_template must both be set in the YAML. "
            "See 02_train_config_cot_0520.yaml for the expected format."
        )

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

    first_prefix_holder: list[str] = []
    train_ds = build_dataset(raw, tokenizer, cfg,
                             first_prefix_holder=first_prefix_holder)
    logging.info(
        "Dataset built: train=%d, padded to %d tokens",
        len(train_ds), cfg.max_seq_len,
    )

    first = train_ds[0]
    cm = torch.tensor(first["channel_mask"])
    logging.info(
        "Sanity train_ds[0]: masked=%d, analysis=%d, final=%d",
        int((cm == 0).sum()), int((cm == 1).sum()), int((cm == 2).sum()),
    )

    # Render-and-log sanity print (rank-0 only). Confirms that the chat template
    # routed `system_prompt` into the `<|start|>developer<|message|># Instructions`
    # block (gpt-oss behavior) and that the prev-call section was rendered or
    # stripped as configured.
    if first_prefix_holder and (not distributed or os.environ.get("LOCAL_RANK", "0") == "0"):
        logging.info(
            "===== Rendered prefix for train_raw[0] (specials visible) =====\n%s\n"
            "===== End rendered prefix =====",
            first_prefix_holder[0],
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
