# TSE SFT Preprocessing — What We Built

High-level summary of the preprocessing pipeline. For prompt text and
implementation details, see `0510_preprocessing_plan.md` and
`01_preprocessing.py`. For env quirks hit along the way, see
`gpt_oss_gotchas.md`.

## Goal

Turn `TSE_by_objection.parquet` (one row per customer objection) into
SFT-ready rows for fine-tuning gpt-oss-120b on outbound-sales objection
handling. The teacher used for all preprocessing is gpt-oss-120b itself —
the student is the same architecture, fine-tuned on the teacher's output.

## Inputs the student sees at train and inference

For each objection row, the student is conditioned on:

- `transcript_till_objection`        — raw call up to the objection
- `objection_summary`                — comes from the raw data
- `current_call_summary`             — produced by Stage 4
- `previous_calls_summary`           — produced by Stage 3 (per-customer cache)
- `customer_context`                 — industry, revenue tier, employee count, customer tier
- `product_context`                  — product type, campaign type/subtype

Training targets (training only):

- `analysis_channel  = teacher_analysis`   — Stage 5 CoT
- `final_channel     = polished_response`  — Stage 2 polished agent turn

## Pipeline

```
Raw parquet
  │
  ├─ Stage 1  Substantive turn filter        (low effort)
  │           drop pure-backchannel rows
  │
  ├─ Stage 2  Agent response polish          (medium effort)
  │           remove fillers, collapse repeats, complete obvious truncations,
  │           consolidate `|`-separated fragments → one clean agent turn
  │           secondary drop via `exclude` flag for un-polishable rows
  │
  ├─ Stage 3  Previous-calls summary         (medium effort, cached per customer)
  │           150–300 word prose history of prior calls; skipped when none
  │
  ├─ Stage 4  Current-call summary           (medium effort)
  │           80–150 word prose state-of-call (distinct from objection summary)
  │
  ├─ Stage 5  Teacher CoT + response         (high effort)
  │           full context → analysis channel (CoT) + final channel (agent turn)
  │           consistency check vs. polished_response (deferred to inspection)
  │
  └─ Save preprocessed parquet
```

Filter for SFT-ready rows downstream:
`keep_stage1 & ~exclude_stage2 & consistency_flag`.

## Why each stage exists

- **Stage 1** drops rows where the agent didn't actually engage (pure "yeah,
  uh-huh"). Cheap reasoning — it's a coarse keep/drop decision.
- **Stage 2** is the source of truth for the student's `final_channel` target.
  The raw `agent_response_snippet` is fragmented and full of disfluencies; we
  need a clean single utterance without fabricating content.
- **Stage 3** gives the student long-horizon relationship memory the raw
  transcript doesn't carry. Cached because many objection rows share a
  customer (and therefore share `previous_calls`).
- **Stage 4** is the conversational state at the moment of objection — what
  led here. Distinct from `objection_summary`, which describes the objection
  itself, not the surrounding state.
- **Stage 5** produces the teacher's reasoning. The analysis channel becomes
  the student's CoT target; the final channel is used to sanity-check Stage 2
  (if the teacher with full context picks a totally different move than the
  agent actually said, something's off and we drop the row).

## Code layout

Two files (the .py serves as both importable lib and CLI batch script — the
notebook imports from it via `importlib`):

- **`01_preprocessing.py`**
  - `load_teacher_model` (bf16, `device_map="auto"`, faulty-GPU-aware)
  - `run_and_show` — single-prompt inference with harmony channel splitting
  - `get_customer_context`, `get_product_context` — easy-to-edit context blocks
  - 5 stage prompt constants + per-row functions + per-stage batch runners
  - CLI: `--input/--output/--stages/--limit/--cuda-visible/--resume`
  - Resume-safe checkpoints between stages (atomic parquet replace)
- **`01_preprocessing.ipynb`**
  - Cell-per-stage inspection on a single picked row, both channels printed
  - Side-by-side comparison of `teacher_final` vs `polished_response`
  - 10-row smoke test cell
  - Inspection checklist before launching the full batch

## Env constraints we designed around

- **No MXFP4 packed runtime.** Torch 2.6 / Triton 3.2 forces dequant to bf16
  at load (~240 GB), so we shard across 3–4 H100s with `device_map="auto"`
  and `torch_dtype=torch.bfloat16`. (Background: `environment_limitations.md`.)
- **Faulty GPUs in the cluster.** `--cuda-visible` is plumbed end-to-end;
  notebook sets `CUDA_VISIBLE_DEVICES` before importing torch.
- **No `openai-harmony` library.** Channel splitting done with regex on
  decoded output (`skip_special_tokens=False`).
- **No data leaves the corporate machine.** All preprocessing runs locally.

## Open items (deferred to inspection phase)

1. Stage 5b consistency check — embedding cosine vs. lightweight LLM judge.
2. Per-stage reasoning-effort tuning once we see real outputs.
3. Whether to include more fields in `customer_context` / `product_context`.
4. Whether to hard-filter `cs_great_id == 0` rows up front or post-hoc.
