"""QLoRA / LoRA SFT training for the outbound-sales conversation model.

Reads `data/train_examples.jsonl` (produced by `01_preprocessing.ipynb`) and
fine-tunes the base model with LoRA adapters. Loss is computed only on the
`cleaned_target` field — the prefix is masked with -100.

Two quantization modes:
    --quantization lora   (default) full bf16 weights, no bitsandbytes
    --quantization qlora  4-bit nf4 + double-quant via bitsandbytes

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

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    p.add_argument("--quantization", choices=["lora", "qlora"], default="lora",
                   help="lora = full bf16 weights; qlora = 4-bit nf4 via bitsandbytes")
    p.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                   help="Training JSONL with prefix + cleaned_target fields")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where checkpoints, TensorBoard runs, and the adapter land")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return p.parse_args()


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


def build_model(model_dir: str, quantization: str, vocab_size: int):
    """Load the base model in the requested quantization mode and prep for LoRA."""
    if quantization == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(vocab_size)
        model = prepare_model_for_kbit_training(model)
        optim = "paged_adamw_8bit"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(vocab_size)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        optim = "adamw_torch"

    logging.info("Base model loaded (%s). Parameters: %s", quantization, f"{model.num_parameters():,}")
    return model, optim


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Run config: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info("Added %d special tokens. Vocab size: %d", num_added, len(tokenizer))

    dataset = load_and_tokenize(data_file, tokenizer, MAX_SEQ_LEN)
    logging.info("Dataset: %d examples, padded to %d tokens", len(dataset), MAX_SEQ_LEN)

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

    model, optim = build_model(args.model_dir, args.quantization, len(tokenizer))

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
        optim=optim,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    logging.info("Starting training. TensorBoard: tensorboard --logdir %s", output_dir / "runs")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    logging.info("Training finished in %.1fs (%.2f min)", elapsed, elapsed / 60.0)

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logging.info("Wrote step-level log history to %s", log_path)

    adapter_path = output_dir / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logging.info("Adapter + tokenizer saved to %s", adapter_path)


if __name__ == "__main__":
    main()
