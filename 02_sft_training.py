"""LoRA SFT training for the outbound-sales conversation model.

Reads `data/train_examples.jsonl` (produced by `01_preprocessing.ipynb`) and
fine-tunes the base model with LoRA adapters in bf16. Loss is computed only on
the `cleaned_target` field — the prefix is masked with -100.

For GPT-OSS-120B the model is sharded across the node with DeepSpeed ZeRO-3.
The base weights are MXFP4-native, so no extra bnb/QLoRA quantization is used.

Launch:
    # Single-GPU LoRA (debug / small base models):
    python 02_sft_training.py --epochs 3

    # 8-GPU ZeRO-3 (production run for GPT-OSS-120B):
    deepspeed --num_gpus=8 02_sft_training.py --deepspeed zero3 --epochs 3

    # Smoke-test the full pipeline on 30 samples (one short epoch):
    deepspeed --num_gpus=8 02_sft_training.py --deepspeed zero3 \\
        --max-samples 30 --epochs 1

    # Multi-node: use a hostfile.
    deepspeed --hostfile=hostfile --num_gpus=8 02_sft_training.py --deepspeed zero3

Logs:
    - Python `logging` to stdout (status, wall-clock time)
    - TensorBoard event files at OUTPUT_DIR/runs/  (loss / LR / grad-norm)
    - JSON dump of trainer.state.log_history at OUTPUT_DIR/training_log.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Defaults (override via CLI flags below)
# ============================================================
DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_DATA_FILE = "data/train_examples.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints"
DEFAULT_EPOCHS = 3

# Hyperparameters (not exposed via CLI — edit here if needed)
MAX_SEQ_LEN = 4096
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
WARMUP_RATIO = 0.05
LOGGING_STEPS = 10

SPECIAL_TOKENS = [
    "<|system|>", "<|/system|>",
    "<|context|>", "<|/context|>",
    "<|conversation|>", "<|/conversation|>",
    "<|agent|>",
    "<|customer|>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="Local base-model checkpoint directory")
    p.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                   help="Training JSONL with prefix + cleaned_target fields")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where checkpoints, TensorBoard runs, and the adapter land")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--deepspeed", default=None,
                   help="Pass 'zero3' for the built-in ZeRO-3 preset, or a path to a custom "
                        "DeepSpeed JSON config. Omit for single-GPU training.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Train on only the first N examples — for smoke-testing the pipeline.")
    # The `deepspeed` launcher injects --local_rank=N per rank. Absorb it so argparse
    # doesn't reject it; Trainer reads the actual value from the LOCAL_RANK env var.
    p.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def build_zero3_config() -> dict:
    """Inline ZeRO-3 config tuned for LoRA fine-tuning of GPT-OSS-120B on an 8-GPU node.

    Param + optimizer states are CPU-offloaded because the unsharded 120B base model
    plus activations does not fit in 8x80GB. `stage3_gather_16bit_weights_on_model_save`
    is the critical knob for the save-time gather — without it, save_pretrained either
    writes shard fragments or has every rank try to gather full weights at once (OOM).
    """
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            # `offload_param` removed — keeping the partitioned base weights on GPU is
            # faster, and on 8x80GB the per-rank partition (~11GB of 90GB total) fits.
            # `offload_optimizer` kept: harmless because LoRA optimizer state is tiny,
            # and it leaves a bit more GPU room for activations.
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
    """Translate the --deepspeed CLI value into what TrainingArguments expects."""
    if arg is None:
        return None
    if arg == "zero3":
        return build_zero3_config()
    cfg_path = Path(arg)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {cfg_path}")
    return str(cfg_path)


def load_and_tokenize(data_path: Path, tokenizer, max_seq_len: int) -> Dataset:
    """Tokenize JSONL examples with prefix-masked labels and section-aware truncation.

    Truncation drops the oldest turns inside <|conversation|>...<|/conversation|>
    first; the system prompt and context block are preserved unless the budget is
    so tight that even an empty conversation does not fit, in which case the head
    is left-truncated as a last resort.
    """
    conv_start_id = tokenizer.convert_tokens_to_ids("<|conversation|>")
    conv_end_id = tokenizer.convert_tokens_to_ids("<|/conversation|>")

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"No examples found in {data_path}")
    if "cleaned_target" not in examples[0]:
        raise ValueError(
            f"Example missing 'cleaned_target' field in {data_path} — "
            "re-run 01_preprocessing.ipynb to regenerate the JSONL."
        )

    input_ids_list, labels_list, attention_mask_list = [], [], []
    n_truncated_conv = 0
    n_truncated_head = 0

    for ex in examples:
        prefix_ids = tokenizer.encode(ex["prefix"], add_special_tokens=False)
        target_ids = tokenizer.encode(ex["cleaned_target"], add_special_tokens=False)
        target_ids = target_ids + [tokenizer.eos_token_id]

        try:
            conv_start = prefix_ids.index(conv_start_id) + 1
            conv_end = prefix_ids.index(conv_end_id)
        except ValueError:
            conv_start = conv_end = None

        if conv_start is not None and conv_end is not None:
            head_ids = prefix_ids[:conv_start]
            conv_ids = prefix_ids[conv_start:conv_end]
            tail_ids = prefix_ids[conv_end:]

            budget = max_seq_len - len(head_ids) - len(tail_ids) - len(target_ids)
            if budget < 0:
                overflow = -budget
                head_ids = head_ids[overflow:]
                conv_ids = []
                n_truncated_head += 1
            elif len(conv_ids) > budget:
                conv_ids = conv_ids[-budget:]
                n_truncated_conv += 1

            prefix_ids = head_ids + conv_ids + tail_ids
        else:
            if len(prefix_ids) + len(target_ids) > max_seq_len:
                overflow = len(prefix_ids) + len(target_ids) - max_seq_len
                prefix_ids = prefix_ids[overflow:]
                n_truncated_head += 1

        full_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids

        pad_len = max_seq_len - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    logging.info("Truncated conversation (oldest turns dropped): %d", n_truncated_conv)
    logging.info("Truncated head (system/context also clipped): %d", n_truncated_head)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    })


def build_model(model_dir: str, vocab_size: int, use_deepspeed: bool):
    """Load the base model in bf16 and prep for LoRA.

    Under DeepSpeed ZeRO-3, device placement is owned by DS and `device_map="auto"`
    conflicts with sharding, so we drop it. Gradient checkpointing is handled by
    TrainingArguments(gradient_checkpointing=True) in that path too.
    """
    from_pretrained_kwargs = dict(
        local_files_only=True,
        dtype="auto",  # honor the MXFP4-native dtype on disk; don't upcast on load
    )
    if not use_deepspeed:
        from_pretrained_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_dir, **from_pretrained_kwargs)
    model.resize_token_embeddings(vocab_size)

    if not use_deepspeed:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    logging.info("Base model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_config = resolve_deepspeed(args.deepspeed)

    # CRITICAL: must register the DS config BEFORE from_pretrained so ZeRO-3 partitions
    # parameters during model construction (via deepspeed.zero.Init). Without this,
    # every rank materializes the full 120B model and OOMs CUDA. The handle must stay
    # in scope — HfDeepSpeedConfig holds the global registration via a weakref.
    if ds_config is not None:
        from transformers.integrations import HfDeepSpeedConfig
        _dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 — keep alive for the rest of main()

    logging.info("Run config: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Added %d special tokens. Vocab size: %d", num_added, len(tokenizer))

    dataset = load_and_tokenize(data_file, tokenizer, MAX_SEQ_LEN)
    logging.info("Dataset: %d examples, padded to %d tokens", len(dataset), MAX_SEQ_LEN)

    if args.max_samples is not None:
        n = min(args.max_samples, len(dataset))
        dataset = dataset.select(range(n))
        logging.info("Smoke-test mode: trimmed dataset to %d examples", n)

    first = dataset[0]
    mask_boundary = next(i for i, l in enumerate(first["labels"]) if l != -100)
    pad_start = next(
        (i for i, l in enumerate(first["labels"]) if i > mask_boundary and l == -100),
        len(first["labels"]),
    )
    logging.info(
        "Sanity check on dataset[0]: prefix=%d masked tokens, target=%d loss tokens, attention=%d",
        mask_boundary,
        pad_start - mask_boundary,
        sum(first["attention_mask"]),
    )

    model = build_model(args.model_dir, len(tokenizer), use_deepspeed=ds_config is not None)

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

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    logging.info("Starting training. TensorBoard: tensorboard --logdir %s", output_dir / "runs")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    logging.info("Training finished in %.1fs (%.2f min)", elapsed, elapsed / 60.0)

    # Trainer.save_model is ZeRO-3- and PEFT-aware: it triggers the param gather
    # (via stage3_gather_16bit_weights_on_model_save) and writes only on rank 0,
    # so we don't get one full copy per GPU like the old model.save_pretrained did.
    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_path))
        log_path = output_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        logging.info("Adapter + tokenizer saved to %s", adapter_path)
        logging.info("Wrote step-level log history to %s", log_path)


if __name__ == "__main__":
    main()
