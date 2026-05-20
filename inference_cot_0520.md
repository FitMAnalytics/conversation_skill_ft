# `03_inference_cot_0520.py` — Batch inference for the CoT adapter

DataFrame in, parquet out. Loads the **same YAML** used to train the adapter, so the prompt the model sees at inference is byte-for-byte identical to what it saw during training — no drift, no second source of truth for prompt templates.

## Files

| File | Role |
|---|---|
| `03_inference_cot_0520.py` | Inference script. |
| `02_train_config_cot_0520.yaml` | The runtime contract. Same YAML for train and inference. |

## What this script does

For each input row:

1. Build `[{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": <rendered>}]`. The user content is rendered by `render_user_prompt(cfg, row, trimmed_transcript)` — the exact same helper the training script uses.
2. Apply the gpt-oss Harmony chat template with `add_generation_prompt=True` and the requested `reasoning_effort`. The template auto-routes `system` content into the `developer` block under `# Instructions:`.
3. Batched greedy generate (`do_sample=False`) with explicit `input_ids` + `attention_mask` (left-padded).
4. Decode with `skip_special_tokens=False`; parse `analysis` and `final` channels via regex.
5. Optionally repeat under `model.disable_adapter()` for an A/B against the base model.

## Required input columns

The columns the input parquet/csv must carry depend on the YAML toggle:

| Always required | Required only when `include_previous_call_summary: true` |
|---|---|
| `transcript`, `current_call_summary`, `customer_context` | `previous_calls_summary` |

(All names are taken from the YAML's `*_field` keys, so renaming a column is a YAML change, not a code change.) The training targets (`teacher_cot`, `polished_response`) are **not** needed — those are what we generate.

`customer_context` must be a non-empty string (the script errors loudly if it is null/empty). It is the pre-rendered context block produced by `01_preprocessing.py`. No `get_customer_context()` fallback exists in the CoT pipeline — the column must already be there.

## Output columns

Added to every row of the input DataFrame and written to `--output` as parquet:

| Column | Always written | Notes |
|---|---|---|
| `raw_response` | yes | Full decoded model output with channel tokens visible. |
| `analysis` | yes | Parsed analysis channel content (teacher-CoT style reasoning). |
| `final` | yes | Parsed final channel content (the agent's polished turn). |
| `base_raw_response` / `base_analysis` / `base_final` | only with `--include-base` | Same three columns generated under `model.disable_adapter()` for A/B. |

## CLI

```
python 03_inference_cot_0520.py \
    --config 02_train_config_cot_0520.yaml \
    --input data/eval.parquet \
    --output data/eval_predictions_cot.parquet \
    --model-dir /path/to/gpt-oss-20b \
    --adapter-dir checkpoints_cot_0520/final_adapter \
    --batch-size 4 \
    --max-new-tokens 1024 \
    --reasoning-effort medium
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--config` (required) | Same YAML used to train. Drives prompts + input column contract. |
| `--model-dir`, `--adapter-dir` | Override paths without editing the YAML. Use any `checkpoint-N/` to evaluate intermediate snapshots. |
| `--include-base` | Also generate with the LoRA adapter disabled. Writes `base_*` columns. |
| `--max-samples N` | Smoke cap — first N pending rows only. |
| `--save-every N` | Partial-parquet checkpoint cadence. Resume-safe: re-running with the same `--output` skips rows that already have a non-null `final`. |
| `--no-prefix-cache` | Disable the shared-prefix DynamicCache. Slower; useful if your transformers version mishandles `past_key_values` with `generate`. |
| `--debug-lcp` | Print two probe-batch prompts side-by-side and the exact character / token where they diverge. Use this to diagnose a smaller-than-expected LCP (usually caused by a per-row field leaking into the prefix). |
| `--max-transcript-chars N` | Override the YAML's value. Default: use the YAML so train and inference stay in lockstep. |

## Evaluating intermediate checkpoints

The training script writes a checkpoint every `save_steps` optimizer steps into `checkpoints_cot_0520/checkpoint-N/`. Each contains `adapter_config.json` + `adapter_model.safetensors` (the LoRA weights), along with optimizer/scheduler/trainer-state files that inference ignores. To A/B different checkpoints against a held-out test set, just point `--adapter-dir` at the directory you want:

```
python 03_inference_cot_0520.py --config 02_train_config_cot_0520.yaml \
    --input data/eval_holdout.parquet \
    --output data/eval_holdout_checkpoint-90.parquet \
    --adapter-dir checkpoints_cot_0520/checkpoint-90
```

The tokenizer is loaded from `--model-dir` (the base gpt-oss directory), not the adapter directory — SFT doesn't change the tokenizer, so switching checkpoints is purely a `--adapter-dir` flag change. Re-run with different `--adapter-dir` and `--output` values to score each checkpoint independently.

## Prefix-cache speedup

`cfg.system_prompt` and `--reasoning-effort` are fixed for a run, so every prompt shares a long token prefix (the rendered system + developer + everything before the first per-row field). The script:

1. Renders the first batch.
2. Finds the LCP across those prompts (or, if batch size is 1, diffs against a system-only render — passing the same `reasoning_effort` so the template default doesn't poison the probe).
3. If LCP ≥ 64 tokens, runs one forward pass over the prefix to populate a batch=1 `DynamicCache`, then for each subsequent batch expands that cache to batch=B and runs `generate(past_key_values=...)` on just the per-row suffix tokens.

Prefill cost drops from `O(B * |prefix|)` to `O(|prefix|)` for the whole run. The cached path falls back to plain encoding+generate if any row in a batch doesn't start with the cached prefix (logged loudly so you can investigate).

## Swapping gpt-oss-20B ↔ 120B

Change `model_dir` in the YAML (and optionally pass `--model-dir` at the CLI). Tokenizer, Harmony chat template, and LoRA target modules are the same across both checkpoints, so nothing else moves.

## Gotchas

This script already handles the items in `gpt_oss_gotchas.md`:

1. bf16 dtype at load via `dtype="auto"` on a single-process `device_map="auto"`.
2. Render-then-tokenize (the chat template renders to a string; tokenization happens separately to dodge the `BatchEncoding`-vs-tensor inconsistency across transformers versions).
3. Decode with `skip_special_tokens=False` so channel boundaries survive.
4. `padding_side = "left"` set at model load (decoder-only batched generation).

Single-process only — **do not** launch this script under `accelerate launch` or `deepspeed`. Inference uses `device_map="auto"` to shard the model across local GPUs.

## Pointers

- Training script + YAML semantics: `sft_training_cot_0520.md`.
- Environment papercuts (bf16 load, CUDA visibility, decode flags): `gpt_oss_gotchas.md`.
- Reference (nocot) inference: `03_inference_nocot.py` — this script is structurally a fork with the prompt path swapped to read from the YAML.
