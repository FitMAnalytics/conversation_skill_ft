# gpt-oss CoT SFT — End-to-End Workflow (0520)

Quick reference for the CoT (teacher-CoT) training + inference pipeline. Reads top-to-bottom: prepare data → train → inference → pick best checkpoint.

| Stage | Script | YAML | Docs |
|---|---|---|---|
| 1. Preprocess (existing) | `01_preprocessing.py` | — | `preprocessing_steps.md` |
| 2. Train | `02_sft_training_cot_0520.py` | `02_train_config_cot_0520.yaml` | `sft_training_cot_0520.md` |
| 3a. Inference (one checkpoint) | `03_inference_cot_0520.py` | reads training YAML | `inference_cot_0520.md` |
| 3b. Inference sweep (all checkpoints) | `03_inference_all_checkpoints_0520.py` | reads training YAML | (this file) |
| Env papercuts | — | — | `gpt_oss_gotchas.md` |

## 0. Prerequisites

- Local gpt-oss-20B (or 120B) weights at `model_dir` — base + tokenizer files together.
- A preprocessed JSONL at `data/preprocessed.jsonl` with these columns: `transcript`, `previous_calls_summary`, `current_call_summary`, `customer_context`, `teacher_cot`, `polished_response`. (See `01_preprocessing.py` for how these are produced.)
- A held-out eval parquet for picking the best checkpoint (same input columns minus the two targets).

## 1. Preprocess (existing pipeline)

Run `01_preprocessing.py` against your raw data to produce `data/preprocessed.jsonl`. This stage builds `previous_calls_summary`, `current_call_summary`, `customer_context`, `polished_response`, `teacher_cot`. Not changed in the 0520 work — see existing docs.

## 2. Train

```bash
# Single-GPU smoke test (20 examples, 1 epoch)
python 02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml \
    --max-samples 20 --epochs 1

# Multi-GPU production via accelerate + FSDP
accelerate launch --config_file <your_fsdp_config.yaml> \
    02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml
```

Before launching for real, update `02_train_config_cot_0520.yaml`:

- `model_dir` — `/path/to/gpt-oss-20b` (or 120B).
- `input` — your preprocessed JSONL.
- `output_dir` — where checkpoints go (default `checkpoints_cot_0520/`).
- `system_prompt` — your task instructions. Auto-routed to the Harmony `developer` block by the chat template.
- `user_prompt_template` — Python format-string with `{customer_context}`, `{previous_calls_summary}`, `{current_call_summary}`, `{transcript}` placeholders, framed by `<<PREV_CALL_SECTION>>` / `<<END>>` markers around the prev-calls block.
- `include_previous_call_summary` — `true` / `false`. When `false`, the marker block (and its body) is stripped from the prompt and the column isn't read.
- `w_analysis`, `w_final` — defaults `0.5` / `0.5`. Per-example, per-channel-mean loss: `loss = 0.5 * mean(ce[analysis]) + 0.5 * mean(ce[final])`, averaged over the batch.
- `save_steps` — checkpoint cadence in **optimizer steps**. With `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, and `N` GPUs: one checkpoint per `save_steps × 8 × N` training rows. `save_steps: 30` → 240 rows/checkpoint on single-GPU, 1440 on 6 GPUs.
- `save_total_limit` — defaults to 10. Raise if you want every checkpoint kept for the multi-checkpoint sweep.

There is **no in-script train/val split** — all rows train. Picking the right checkpoint is a separate step (§4).

What you should see at startup:

```
Loaded N examples from data/preprocessed.jsonl
Dataset built: train=N, padded to 4096 tokens
Sanity train_ds[0]: masked=..., analysis=>0, final=>0
===== Rendered prefix for train_raw[0] (specials visible) =====
<|start|>system<|message|>...Reasoning: medium...<|end|>
<|start|>developer<|message|># Instructions
<your system_prompt here>...
```

Both `analysis>0` and `final>0` in the sanity line confirm the CoT-aware masking is wired up. The render-log block confirms the gpt-oss chat template re-routed `system_prompt` into the developer block.

Output artifacts in `output_dir`:

- `checkpoint-30/`, `checkpoint-60/`, … (every `save_steps` optimizer updates)
- `final_adapter/` (written at the end of training)
- `train_config.yaml` (snapshot of resolved config)
- `training_log.json` (full step-level log)
- `runs/` (TensorBoard logs — `loss`, `loss_a`, `loss_f`)

## 3a. Inference — single checkpoint

```bash
python 03_inference_cot_0520.py \
    --config 02_train_config_cot_0520.yaml \
    --input data/eval_holdout.parquet \
    --output data/eval_predictions.parquet \
    --model-dir /path/to/gpt-oss-20b \
    --adapter-dir checkpoints_cot_0520/final_adapter \
    --batch-size 4 --max-new-tokens 1024 --reasoning-effort medium
```

Input must carry `transcript`, `current_call_summary`, `customer_context`, and — if your training YAML has `include_previous_call_summary: true` — `previous_calls_summary`. The script errors loudly if any are missing.

Output columns added: `raw_response`, `analysis`, `final` (plus `base_raw_response`, `base_analysis`, `base_final` with `--include-base`).

Useful flags:

| Flag | Use |
|---|---|
| `--adapter-dir checkpoints_cot_0520/checkpoint-90` | Evaluate any intermediate checkpoint — tokenizer comes from `--model-dir`, so this is just a path swap. |
| `--include-base` | Also generate with `model.disable_adapter()` for A/B vs. the base model. |
| `--max-samples N` | Smoke cap. |
| `--save-every N` | Partial parquet checkpoint every N batches. Resume-safe — rerunning the same command skips completed rows. |
| `--no-prefix-cache` | Disable the DynamicCache prefix-sharing optimization. Slower; use if you suspect issues with `past_key_values`. |
| `--debug-lcp` | Dump two probe prompts and the exact char/token where they diverge — diagnoses a smaller-than-expected LCP. |
| `--max-transcript-chars N` | Override the YAML's value. Default: take it from the YAML so train and inference stay in lockstep. |

## 3b. Inference — sweep every checkpoint

When you've trained for many epochs and want to A/B all checkpoints against a held-out test set:

```bash
python 03_inference_all_checkpoints_0520.py \
    --config 02_train_config_cot_0520.yaml \
    --input data/eval_holdout.parquet \
    --output-dir data/eval_predictions_by_checkpoint/ \
    --adapter-root checkpoints_cot_0520 \
    --model-dir /path/to/gpt-oss-20b \
    --batch-size 4 --max-new-tokens 1024
```

What it does:

1. Discovers every `checkpoints_cot_0520/checkpoint-N/` directory plus `final_adapter/` (sorted by step number).
2. Loads the base model + tokenizer **once**.
3. For each checkpoint, hot-swaps the LoRA adapter via `model.delete_adapter` + `model.load_adapter` and runs the same per-row inference loop as 3a.
4. Writes one parquet per checkpoint: `predictions_checkpoint-30.parquet`, `predictions_checkpoint-60.parquet`, …, `predictions_final_adapter.parquet`.

Skipping and filtering:

| Flag | Use |
|---|---|
| `--steps 30,60,90` | Restrict to those checkpoint step numbers. |
| `--steps 30-300:30` | Same idea via range with stride (every 30 from 30 to 300). |
| `--no-final-adapter` | Skip `final_adapter/`. |
| `--skip-existing` (default on) | Skip any checkpoint whose output parquet exists and is row-count-complete. |
| `--no-skip-existing` | Always rerun, overwriting. |
| `--debug-lcp-first` | Run the LCP debug printer once, on the first checkpoint only. |

All single-checkpoint flags carry over: `--include-base`, `--max-samples`, `--save-every`, `--no-prefix-cache`, `--max-transcript-chars`, `--reasoning-effort`.

Resume semantics work two levels deep: across checkpoints (re-running skips done parquets) and within a checkpoint (re-running re-uses the partial parquet for that checkpoint).

## 4. Picking the best checkpoint

Score each `predictions_*.parquet` against your held-out gold labels using whatever metric is appropriate (BLEU/ROUGE, embedding similarity, downstream task accuracy, human review, etc.) — that scoring step lives outside this repo.

A common pattern:

```python
import pandas as pd
from pathlib import Path

rows = []
for p in sorted(Path("data/eval_predictions_by_checkpoint").glob("predictions_*.parquet")):
    df = pd.read_parquet(p)
    score = your_metric(df["final"], df["gold_response"])
    rows.append({"checkpoint": p.stem.replace("predictions_", ""), "score": score})
print(pd.DataFrame(rows).sort_values("score", ascending=False).head())
```

Once you've picked a winner, point downstream consumers at that specific `checkpoints_cot_0520/checkpoint-N/` directory.

## 5. Swapping gpt-oss-20B ↔ 120B

One YAML edit: `model_dir`. Both checkpoints share the same Harmony chat template (system → developer routing, analysis/final channels) and the same LoRA target module names (`q_proj/k_proj/v_proj/o_proj`). The 120B needs FSDP across multiple GPUs at training time; the 20B fits single-node with `device_map="auto"`.

Inference (single or sweep) is single-process with `device_map="auto"` for both — do **not** launch the inference scripts under `accelerate launch` or `deepspeed`.

## 6. Files reference

| File | Purpose |
|---|---|
| `02_sft_training_cot_0520.py` | Training script. Reads YAML, builds dataset, trains, saves checkpoints + final adapter. |
| `02_train_config_cot_0520.yaml` | The training contract — prompts, field names, loss weights, optimizer, LoRA, checkpoint cadence. |
| `03_inference_cot_0520.py` | Single-adapter batch inference. Loads the same YAML for prompt rendering. |
| `03_inference_all_checkpoints_0520.py` | Sweep every checkpoint + final_adapter; one parquet per. Loads base once, hot-swaps adapters. |
| `sft_training_cot_0520.md` | Detailed training docs. |
| `inference_cot_0520.md` | Detailed single-checkpoint inference docs. |
| `gpt_oss_gotchas.md` | Environment papercuts (bf16 load, render-then-tokenize, CUDA_VISIBLE_DEVICES ordering, left-padding). |

## 7. Common pitfalls

- **Empty `customer_context` at inference**. The inference scripts error loudly — fix the input parquet upstream rather than masking it in code.
- **Train/inference prompt drift**. Always pass the *same* YAML to training and inference. The YAML is the single source of truth for `system_prompt`, `user_prompt_template`, the prev-call toggle, and all field names.
- **`previous_call_summary_field` spelling**. The data column is `previous_calls_summary` (plural). The YAML default already matches; don't switch it back to singular.
- **`save_total_limit` rolls checkpoints out**. If you want every checkpoint kept for the multi-checkpoint sweep, raise `save_total_limit` accordingly before training (or set it to a number > `total_optimizer_steps / save_steps`).
- **`device_map="auto"` under accelerate**. The training script's `is_distributed_launch()` automatically suppresses `device_map="auto"` when launched under `accelerate launch` / `torchrun`. The inference scripts assume single-process — do not launch them under accelerate.
