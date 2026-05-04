"""LoRA SFT training for the outbound-sales conversation model (Harmony-aware).

Reads `data/preprocessed.jsonl` (produced by 01_preprocessing) and fine-tunes the
base model with LoRA in bf16. Each training example is rendered with the native
GPT-OSS Harmony chat template — reasoning lives in the analysis channel, the
polished agent response in the final channel — so the analysis channel structure
is reinforced rather than starved.

Per-token weighted loss replaces the simple -100 mask:
  - prefix tokens                 weight 0   (everything before the assistant turn)
  - analysis-channel content      weight `--analysis-weight` (default 0.2)
  - final-channel content         weight `--final-weight`    (default 1.0)
  - framing tokens (<|channel|>, <|message|>, <|end|>, <|return|>, <|start|>) inside
    the assistant turn            weight `--final-weight` (these MUST be learned)
  - padding                       weight 0

Loss = sum(weights * per_token_ce) / sum(weights), per example, then mean across batch.

Launch:
    # Single-GPU LoRA smoke test:
    python 02_sft_training.py --input data/preprocessed.jsonl --max-samples 20 --epochs 1

    # 8-GPU ZeRO-3 production run for GPT-OSS-120B:
    deepspeed --num_gpus=8 02_sft_training.py --input data/preprocessed.jsonl \\
        --deepspeed zero3 --epochs 3
"""

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
DEFAULT_EPOCHS = 3

MAX_SEQ_LEN = 4096
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10

# Harmony framing — verified against gpt_oss_inspection.ipynb cell 2.
ANALYSIS_HEADER = "<|channel|>analysis<|message|>"
FINAL_HEADER = "<|channel|>final<|message|>"
ANALYSIS_END = "<|end|>"
ASSISTANT_RESTART = "<|start|>assistant"
FINAL_END = "<|return|>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_DATA_FILE)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--analysis-weight", type=float, default=0.2,
                   help="Per-token weight on analysis-channel content tokens")
    p.add_argument("--final-weight", type=float, default=1.0,
                   help="Per-token weight on final-channel content + assistant framing tokens")
    p.add_argument("--deepspeed", default=None,
                   help="'zero3' for the built-in preset, or path to custom JSON. Omit for single-GPU.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke test: train on first N substantial examples")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--window-size", type=int, default=preproc01.DEFAULT_WINDOW_SIZE,
                   help="Keep only the last N turns of transcript. Pass 0 to disable windowing. "
                        "Token-budget truncation (drops oldest turns until prefix+suffix fits "
                        "within --max-seq-len) runs after windowing — system prompt and context "
                        "are never truncated.")
    # Absorb deepspeed launcher flag.
    p.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def build_zero3_config() -> dict:
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }


def resolve_deepspeed(arg: str | None):
    if arg is None:
        return None
    if arg == "zero3":
        return build_zero3_config()
    cfg_path = Path(arg)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {cfg_path}")
    return str(cfg_path)


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

    # Iteratively drop from the front until it fits.
    while True:
        windowed = preproc01.render_turns(turns) if turns else ""
        prefix_text = render_prefix(tokenizer, windowed, history_summary, context)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) + suffix_len <= max_seq_len:
            return prefix_ids, len(turns)
        if not turns:
            # Even an empty transcript overflows — return what we have, the caller
            # will hard-truncate. Never trim the system block; the caller logs this.
            return prefix_ids, 0
        turns = turns[1:]


def encode_with_spans(tokenizer, transcript: str, history_summary: str,
                      reasoning: str, polished_target: str, context: str,
                      max_seq_len: int, window_size: int | None) -> dict:
    """Tokenize a Harmony-rendered example and return per-token weight inputs.

    Returns prefix_ids (already windowed and budget-fit) plus the assistant-turn
    segments to append for weighted loss.
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


def build_dataset(examples: list[dict], tokenizer, max_seq_len: int,
                  analysis_weight: float, final_weight: float,
                  window_size: int | None) -> Dataset:
    """Tokenize all examples and pack into a HF Dataset with input_ids/labels/weights."""
    input_ids_list, labels_list, attention_list, weights_list = [], [], [], []
    n_window_dropped = 0   # examples that lost turns due to window_size
    n_budget_dropped = 0   # examples that lost additional turns due to token budget
    n_overflow = 0         # examples that overflow even with empty transcript

    for ex in examples:
        original_n_turns = len(preproc01.parse_turns(ex["transcript"]))
        encoded = encode_with_spans(
            tokenizer,
            transcript=ex["transcript"],
            history_summary=ex["history_summary"],
            reasoning=ex["reasoning"],
            polished_target=ex["polished_target"],
            context=ex.get("context", ""),
            max_seq_len=max_seq_len,
            window_size=window_size,
        )
        prefix_ids = encoded["prefix_ids"]
        segments = encoded["segments"]
        n_kept = encoded["n_turns_kept"]
        suffix_len = sum(len(s[1]) for s in segments)

        if window_size is not None and original_n_turns > window_size:
            n_window_dropped += 1
        post_window = min(original_n_turns, window_size) if window_size is not None else original_n_turns
        if n_kept < post_window:
            n_budget_dropped += 1
        if len(prefix_ids) + suffix_len > max_seq_len:
            n_overflow += 1

        # Assemble token IDs and weight vector.
        full_ids = list(prefix_ids)
        weights = [0.0] * len(prefix_ids)
        for _name, ids, kind in segments:
            full_ids.extend(ids)
            if kind == "analysis":
                weights.extend([analysis_weight] * len(ids))
            elif kind == "final":
                weights.extend([final_weight] * len(ids))
            else:  # framing
                weights.extend([final_weight] * len(ids))

        # Labels mirror input_ids; positions with weight 0 contribute nothing because
        # we use weights at loss time. We still set labels=-100 there so HF Trainer's
        # default token-counting heuristics behave (and any logging that reads labels).
        labels = [(tid if w > 0 else -100) for tid, w in zip(full_ids, weights)]

        # Pad / truncate to max_seq_len.
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            weights = weights[:max_seq_len]
        attention = [1] * len(full_ids)
        pad = max_seq_len - len(full_ids)
        if pad > 0:
            full_ids = full_ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad
            labels = labels + [-100] * pad
            attention = attention + [0] * pad
            weights = weights + [0.0] * pad

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_list.append(attention)
        weights_list.append(weights)

    logging.info("Examples windowed (>%s turns): %d", window_size, n_window_dropped)
    logging.info("Examples that lost extra turns to token budget: %d", n_budget_dropped)
    if n_overflow:
        logging.warning("Examples overflowing max_seq_len even with empty transcript: %d "
                        "(system+context+suffix alone don't fit — increase --max-seq-len)",
                        n_overflow)
    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_list,
        "loss_weights": weights_list,
    })


class WeightedLossTrainer(Trainer):
    """Trainer with per-token loss weights instead of -100 masking.

    Standard CE shifts logits/labels for next-token prediction. We do the same
    shift on the weights so each weight aligns with the position whose loss we want
    to scale.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("loss_weights")
        # Trainer's default loss path needs `labels` to compute the loss; we'll do
        # it ourselves and discard the model's auto-loss to avoid double work.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous().to(shift_logits.dtype)

        # Replace -100 with 0 so gather doesn't crash on padding; shift_weights is 0
        # there anyway, so the contribution stays zero.
        safe_labels = shift_labels.clamp(min=0)
        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        weighted = per_token_ce * shift_weights
        denom = shift_weights.sum(dim=-1).clamp(min=1e-8)
        per_example_loss = weighted.sum(dim=-1) / denom
        loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss


def build_model(model_dir: str, use_deepspeed: bool):
    kwargs = dict(local_files_only=True, dtype="auto")
    if not use_deepspeed:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    if not use_deepspeed:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    logging.info("Base model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_config = resolve_deepspeed(args.deepspeed)

    if ds_config is not None:
        from transformers.integrations import HfDeepSpeedConfig
        _dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 — must stay alive

    logging.info("Run config: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("is_substantial", True):
                continue
            raw.append(row)
    logging.info("Loaded %d substantial examples from %s", len(raw), args.input)

    if args.max_samples is not None:
        raw = raw[:args.max_samples]
        logging.info("Smoke-test mode: trimmed to %d examples", len(raw))

    dataset = build_dataset(
        raw, tokenizer, args.max_seq_len,
        args.analysis_weight, args.final_weight,
        window_size=args.window_size if args.window_size > 0 else None,
    )
    logging.info("Dataset built: %d examples, padded to %d tokens", len(dataset), args.max_seq_len)

    # Sanity print on example 0.
    first = dataset[0]
    w = torch.tensor(first["loss_weights"])
    logging.info(
        "Sanity dataset[0]: prefix=%d masked, analysis_tokens@%.2f=%d, final_tokens@%.2f=%d, total_w=%.2f",
        int((w == 0).sum()),
        args.analysis_weight,
        int((w == args.analysis_weight).sum()),
        args.final_weight,
        int((w == args.final_weight).sum()),
        float(w.sum()),
    )

    model = build_model(args.model_dir, use_deepspeed=ds_config is not None)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        logging_dir=str(output_dir / "runs"),
        save_strategy="epoch",
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        deepspeed=ds_config,
    )

    trainer = WeightedLossTrainer(model=model, args=training_args, train_dataset=dataset)

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
