# `02_sft_training_cot_0520.py` — Teacher-CoT SFT for gpt-oss

LoRA SFT for the outbound-sales conversation model on gpt-oss-20B / 120B. Supervises both the analysis channel (teacher CoT) and the final channel (polished agent response). Prompt templates live in YAML so you can iterate on prompts without editing Python.

## Files

| File | Role |
|---|---|
| `02_sft_training_cot_0520.py` | Training script. |
| `02_train_config_cot_0520.yaml` | Run configuration (paths, prompts, field names, loss weights, optimizer, LoRA). |

## Differences from `02_sft_training_nocot_0511.py`

| Aspect | `_nocot_0511` | This script |
|---|---|---|
| Analysis-channel target | None (empty span, masked) | Read from `analysis_target_field` (e.g. `teacher_cot`) |
| Final-channel target | `polished_target_field` | `final_target_field` (e.g. `polished_response`) |
| Loss | Mean over final tokens only | Per-example, per-channel mean; `w_a * loss_a + w_f * loss_f` (defaults 0.5/0.5), then mean across batch |
| Prompts | `SYSTEM_CONTENT`, `USER_TMPL` constants in Python | `system_prompt`, `user_prompt_template` in YAML |
| Customer context | `get_customer_context(row)` Python placeholder | Read straight from `customer_context_field` (pre-rendered upstream) |
| `objection_summary` | Used | Dropped |
| Default model | gpt-oss-120B | gpt-oss-20B (swap `model_dir` for 120B) |
| Train/val split | 90/10 customer-level, in-script eval | None — all rows train; A/B checkpoints against a held-out test set via `03_inference_cot_0520.py` |

Everything else — `is_distributed_launch()` gating, `device_map="auto"` for single-GPU, gradient-checkpointing, prefix-trim overflow handling, padding, `TrainingArguments` — is byte-for-byte the same as the working `_nocot_0511` baseline.

## YAML knobs

### Paths
- `model_dir` — local directory containing gpt-oss-20B or 120B weights + tokenizer.
- `input` — JSONL training file produced by `01_preprocessing.py`.
- `output_dir` — checkpoints, TensorBoard runs, and the saved final adapter all go here.

### Prompt templates
- `system_prompt` — the task instructions. The gpt-oss chat template auto-routes this into the `developer` block under `# Instructions:` at render time. You don't need to write `{"role": "developer"}` explicitly. (See **Prompt template authoring** below.)
- `user_prompt_template` — Python format-string with the available placeholders: `{customer_context}`, `{previous_calls_summary}`, `{current_call_summary}`, `{transcript}`.
- `include_previous_call_summary` — boolean. When `false`, the `<<PREV_CALL_SECTION>> ... <<END>>` block (and its body) is stripped from the user prompt and the previous-calls field is not read from the row. When `true`, the marker lines are removed but the body is kept.

### Data field names (JSONL columns)
- `transcript_field`, `previous_call_summary_field`, `current_call_summary_field`, `customer_context_field` — input columns.
- `analysis_target_field` — column holding the teacher CoT (default `teacher_cot`).
- `final_target_field` — column holding the polished response (default `polished_response`).

### Tokenization
- `max_transcript_chars` — char-based cap on the tail of the transcript before tokenization (default 12000 ≈ 2000 words).
- `max_seq_len` — hard cap after tokenization. If the prefix+suffix still overflows after the char trim, tokens are dropped from the **front of the prefix** so the SFT targets are never lost. A warning is logged when this happens.
- `max_samples` — `null` for full training; an int for a smoke test.

### Channel loss weights
- `w_analysis`, `w_final` — defaults `0.5` / `0.5` (literal half/half). See **Loss formulation**.

### LoRA / optimization
- Standard. `q_proj/k_proj/v_proj/o_proj` adapters; LR 1e-5; cosine schedule; grad-accum 8; bf16. Same defaults as the nocot variant.

### Checkpointing & logging

Checkpoint cadence is in **optimizer steps**, not raw rows. One optimizer step processes:

```
per_device_train_batch_size * gradient_accumulation_steps * num_gpus
= 1 * 8 * N rows                  (with the YAML defaults and N GPUs)
```

So `save_steps: 30` writes a checkpoint every `30 * 8 * N = 240 * N` training rows:

| GPUs | Rows per checkpoint |
|---|---|
| 1 (smoke test) | 240 |
| 4 | 960 |
| 6 (production) | 1440 |
| 8 | 1920 |

`save_total_limit: 10` rolls oldest checkpoints out — raise it if you want every checkpoint preserved for later A/B against a held-out test set. Pick the right checkpoint by running:

```
python 03_inference_cot_0520.py --config 02_train_config_cot_0520.yaml \
    --adapter-dir checkpoints_cot_0520/checkpoint-N \
    --input data/eval_holdout.parquet \
    --output data/eval_holdout_checkpoint-N.parquet
```

No in-script eval runs during training. Per-channel diagnostics (`loss_a`, `loss_f`) are logged every `logging_steps` steps and show up in TensorBoard alongside the total `loss`.

## Prompt template authoring

### Available placeholders

| Placeholder | Source |
|---|---|
| `{customer_context}` | `customer_context_field` column |
| `{previous_calls_summary}` | `previous_call_summary_field` column (only if `include_previous_call_summary: true`) |
| `{current_call_summary}` | `current_call_summary_field` column |
| `{transcript}` | `transcript_field` column, after tail-trim to `max_transcript_chars` |

### The `<<PREV_CALL_SECTION>>` marker convention

```yaml
user_prompt_template: |
  CUSTOMER CONTEXT:
  {customer_context}

  <<PREV_CALL_SECTION>>
  PREVIOUS CALL SUMMARY:
  {previous_calls_summary}
  <<END>>

  CURRENT CALL SUMMARY:
  {current_call_summary}

  CURRENT TRANSCRIPT (up to the latest customer turn):
  {transcript}
```

- With `include_previous_call_summary: true`, the two marker lines (`<<PREV_CALL_SECTION>>` and `<<END>>`) are stripped and the body remains.
- With `include_previous_call_summary: false`, the entire block including the body is stripped, and `previous_calls_summary` is **not** read from the row.

### How `system_prompt` becomes Harmony

The script calls `tokenizer.apply_chat_template([{"role":"system",...}, {"role":"user",...}])`. The gpt-oss chat template re-routes the system content into a `developer` role block at render time, so the model sees:

```
<|start|>system<|message|>You are ChatGPT...Reasoning: medium...Valid channels: analysis, commentary, final<|end|>
<|start|>developer<|message|># Instructions
{your system_prompt content}<|end|>
<|start|>user<|message|>{your user_prompt_template content}<|end|>
<|start|>assistant
```

This is the format the base model was post-trained on. To verify visually, look at the `===== Rendered prefix for train_raw[0] =====` block in the training log at startup — it prints the rendered prompt with special tokens visible.

## Loss formulation

For each example `b`:

```
loss_a_b = mean(per_token_ce[channel_mask == 1])     # analysis
loss_f_b = mean(per_token_ce[channel_mask == 2])     # final
loss_b   = w_analysis * loss_a_b + w_final * loss_f_b
```

Batch loss = `mean(loss_b across examples)`. The `.mean()` across the batch is in `ChannelMeanLossTrainer.compute_loss` and is what HF Trainer backprops on.

With defaults `w_analysis=0.5`, `w_final=0.5`, the total magnitude is comparable to a single-channel loss. Skew either direction (e.g. `w_analysis=0.2, w_final=0.8`) to weight the final channel more.

Per-channel diagnostics (`loss_a`, `loss_f`) are logged every `logging_steps` training steps.

## Launch recipes

### Single-GPU smoke test

```
python 02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml \
    --max-samples 20 --epochs 1
```

Use this to verify the prompt renders correctly and the channel masks have non-zero analysis + non-zero final counts. Expect a startup log line like:

```
Sanity train_ds[0]: masked=N, analysis=M, final=K
```

with `M > 0` and `K > 0`. Then the rendered prompt block.

### Multi-GPU FSDP

You supply your own accelerate FSDP config (`fsdp.yaml`) — typical settings: BF16 mixed precision, `FULL_SHARD`, transformer-layer auto-wrap, fp32 optimizer states.

```
accelerate launch --config_file <fsdp.yaml> \
    02_sft_training_cot_0520.py --config 02_train_config_cot_0520.yaml
```

The script detects the launcher via `is_distributed_launch()` (reads `LOCAL_RANK` / `ACCELERATE_USE_FSDP` / `ACCELERATE_USE_DEEPSPEED`) and skips `device_map="auto"` and the in-script `gradient_checkpointing_enable()` call — HF Trainer handles those under FSDP.

CLI overrides that are useful at launch time:
- `--max-samples N` — smoke test cap.
- `--epochs N` — override `num_train_epochs`.
- `--analysis-weight 0.2 --final-weight 0.8` — channel reweighting per run.
- `--max-seq-len`, `--max-transcript-chars` — token / char caps.
- `--output-dir` — keep runs separate without editing YAML.

## Swapping gpt-oss-20B ↔ 120B

Change one line in the YAML:

```yaml
model_dir: /path/to/gpt-oss-120b   # was /path/to/gpt-oss-20b
```

Both checkpoints share the same Harmony chat template (system→developer routing, analysis/final channels), the same LoRA target module names (`q_proj/k_proj/v_proj/o_proj`), and the same tokenizer special-token set. 120B needs FSDP across multiple GPUs; 20B fits single-node with `device_map="auto"`.

## Gotchas

See `gpt_oss_gotchas.md` for the small environment papercuts that this script already handles:

1. bf16 dtype at load (`device_map="auto"` + `dtype="auto"` only when *not* distributed).
2. Render-then-tokenize (used at inference; SFT path tokenizes the rendered string itself).
3. `CUDA_VISIBLE_DEVICES` must be set before any `import torch`.
4. Decode with `skip_special_tokens=False` if you parse channel boundaries downstream.
5. Left-padding for batched generation.

## What to check after a run finishes

- `checkpoints_cot_0520/train_config.yaml` — a snapshot of the resolved config (post CLI overrides).
- `checkpoints_cot_0520/training_log.json` — full step-level log history.
- `checkpoints_cot_0520/runs/` — TensorBoard. `tensorboard --logdir checkpoints_cot_0520/runs` shows `loss`, `loss_a`, `loss_f` curves.
- `checkpoints_cot_0520/final_adapter/` — the LoRA adapter + tokenizer for Stage 03 inference.
