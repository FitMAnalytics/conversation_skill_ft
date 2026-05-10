# TSE SFT Preprocessing — What We Built

High-level summary of the preprocessing pipeline. For prompt text and
implementation details, see `0510_preprocessing_plan.md` and
`01_preprocessing.py`. Stage 5 went through multiple framings before settling
on the current design — authoritative source is `0510_stage5_design.md`
(current decision: **rationalization CoT in the final channel**, teacher IS
shown `polished_response`). For env quirks hit along the way, see
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
- `current_call_summary`             — produced by Stage 3
- `previous_calls_summary`           — produced by Stage 4 (per-customer cache)
- `customer_context`                 — industry, revenue tier, employee count, customer tier
- `product_context`                  — product type, campaign type/subtype

Training targets (training only):

- `analysis_channel  = teacher_cot`         — Stage 5 rationalization CoT
                                              (sourced from the teacher's
                                              **final** channel — see Stage 5
                                              section below for the inversion)
- `final_channel     = polished_response`   — Stage 2 polished agent turn

So the SFT pair is `(teacher_cot, polished_response)`. Coherent by
construction: the teacher was shown `polished_response` and asked to write a
CoT that reasons its way to it.

Other Stage 5 column: `teacher_thinking` (from the teacher's analysis channel)
— OSS's free-form planning while writing the CoT. Stored as diagnostic only;
not a training target.

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
  ├─ Stage 3  Current-call summary           (medium effort)
  │           80–150 word prose state-of-call (distinct from objection summary)
  │
  ├─ Stage 4  Previous-calls summary         (medium effort, cached per customer)
  │           150–300 word prose history of prior calls; skipped when none
  │
  ├─ Stage 5  Rationalization CoT            (high effort)
  │           teacher sees polished_response → writes a forward-looking
  │           first-person CoT in the FINAL channel that lands at it.
  │           Analysis channel = OSS's own planning (diagnostic only).
  │           teacher_cot → student's analysis-channel SFT target.
  │
  └─ Save preprocessed parquet
```

Filter cascade (enforced in code via `_eligibility_mask`): Stages 3, 4, 5 only
run on rows with `keep_stage1=True & exclude_stage2=False`. Stage 5b
(CoT-quality judge) is still TODO. Final SFT-ready filter under v1:
`keep_stage1 & ~exclude_stage2` (add 5b once implemented).

## Why each stage exists

- **Stage 1** drops rows where the agent didn't actually engage (pure "yeah,
  uh-huh"). Cheap reasoning — it's a coarse keep/drop decision.
- **Stage 2** produces `polished_response`, a clean version of the human
  agent's actual response. This is the student's `final_channel` training
  target and an input to Stage 5.
- **Stage 3** is the conversational state at the moment of objection — what
  led here. Distinct from `objection_summary`, which describes the objection
  itself, not the surrounding state. Runs before Stage 4 because Stage 5 reads
  it directly.
- **Stage 4** gives the teacher (and student) long-horizon relationship
  memory the raw transcript doesn't carry. Cached because many objection rows
  share a customer (and therefore share `previous_calls`).
- **Stage 5** is a rationalization step with an inverted channel mapping.
  The teacher is given the polished response as input and writes a
  forward-looking first-person CoT — the kind of reasoning the agent would
  have done silently *before* speaking, ending exactly at the given response.
  The CoT goes in the **final** channel (so it becomes the student's
  analysis-channel SFT target). The teacher's analysis channel is its own
  free-form planning (don't write the CoT here, plan how to write it).
  Coherence of the (CoT, polished_response) training pair is by construction:
  the teacher was shown the response and asked to reason its way to it.
  See `0510_stage5_design.md` for the framing alternatives weighed.

## Code layout

Two files (the .py serves as both importable lib and CLI batch script — the
notebook imports from it via `importlib`):

- **`01_preprocessing.py`**
  - `load_teacher_model` (bf16, `device_map="auto"`, faulty-GPU-aware)
  - `run_and_show` — single-prompt inference with harmony channel splitting
  - `get_customer_context`, `get_product_context` — easy-to-edit context blocks
  - 5 stage prompt constants + per-row functions + per-stage batch runners
  - CLI: `--input/--output/--stages/--limit/--cuda-visible/--auto-save-batch-size/--batch-size`
  - **Auto-save + auto-resume.** Each stage saves the parquet every
    `--auto-save-batch-size` rows (default 50). On startup, if the output
    parquet already exists, it's loaded and resumed from. Per-row resume
    granularity via `_stage{N}_done` boolean tracker columns — interrupt
    mid-stage and the next run picks up exactly where it stopped.
  - **Batched inference.** Stages 1, 2, 3 run generation in batches of
    `--batch-size` (default 4) for throughput. Stages 4 (cached) and 5
    (long, variable-length outputs) stay singleton. From the notebook the
    runners default to singleton (`batch_size=1`) so cell behavior is
    unchanged.
- **`01_preprocessing.ipynb`**
  - Cell-per-stage inspection on a single picked row, both channels printed
  - Side-by-side: does the CoT (final channel) land at `polished_response`?
  - 10-row smoke test cell + per-row dump of every generated field
  - Inspection checklist before launching the full batch

## Input schema (what your parquet needs)

**Hard requirements** — script crashes without these:

| Column | Used by |
|---|---|
| `full_conversation_pii_rmv_till_obj` | stages 1, 2, 3, 5 (transcript text) |
| `agent_response_snippet` | stages 1, 2 (raw fragmented response) |

**Used if present, otherwise gracefully falls back**:

| Column | Used by | Fallback |
|---|---|---|
| `objection_summary` | stage 5 | empty string |
| `previous_calls` | stage 4 | summary returned as `None` |
| `customer_id` or `cust_id` | stage 4 cache key | hashes `previous_calls` |
| `sic4_industry`, `rev_tier`, `emp_ct`, `tier` | `get_customer_context` | `"unknown"` |
| `cw_opp_type`, `task_type`, `task_subtype` | `get_product_context` | `"unknown"` |

Edit `get_customer_context` / `get_product_context` to plumb more fields into
the prompts.

## Output schema

All input columns are preserved. Stages add:

| Column | Source | Type | Notes |
|---|---|---|---|
| `customer_context` | helper | str | from `get_customer_context` |
| `product_context` | helper | str | from `get_product_context` |
| `keep_stage1` | Stage 1 | bool | filter verdict |
| `stage1_reason` | Stage 1 | str | brief reason |
| `polished_response` | Stage 2 | str | clean agent turn; `""` if filtered out. Student's `final_channel` SFT target. |
| `polish_operations` | Stage 2 | list[str] | which polish ops applied |
| `completeness` | Stage 2 | str | "complete" / "completed_from_context" / "incomplete" |
| `exclude_stage2` | Stage 2 | bool | True if un-polishable |
| `exclude_reason` | Stage 2 | str \| None | |
| `current_call_summary` | Stage 3 | str | `""` if ineligible |
| `previous_calls_summary` | Stage 4 | str \| None | `None` if no prior calls or ineligible |
| `teacher_cot` | Stage 5 (final channel) | str | rationalization CoT — student's `analysis_channel` SFT target |
| `teacher_thinking` | Stage 5 (analysis channel) | str | OSS's free-form planning while writing the CoT — **diagnostic only**, not a training target |
| `_stage1_done` … `_stage5_done` | all stages | bool | per-row resume trackers. Internal; downstream consumers can ignore. |

## Env constraints we designed around

- **No MXFP4 packed runtime.** Torch 2.6 / Triton 3.2 forces dequant to bf16
  at load (~240 GB), so we shard across 3–4 H100s with `device_map="auto"`
  and `torch_dtype=torch.bfloat16`. (Background: `environment_limitations.md`.)
- **Faulty GPUs in the cluster.** `--cuda-visible` is plumbed end-to-end;
  notebook sets `CUDA_VISIBLE_DEVICES` before importing torch.
- **No `openai-harmony` library.** Channel splitting done with regex on
  decoded output (`skip_special_tokens=False`).
- **No data leaves the corporate machine.** All preprocessing runs locally.

## Inference recipe (verified working on this env)

The teacher inference call uses **render-then-tokenize**, not
`apply_chat_template(..., return_tensors="pt")`. Pseudocode:

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    reasoning_effort=effort,           # "low" | "medium" | "high"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=..., do_sample=False,
                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
analysis, final = split_channels(raw)
```

This pattern matches the working `gpt_oss_inspection_notebook` and avoids
two real bugs we hit (`KeyError: 'shape'` from `generate`; the same error
from `inputs["input_ids"].shape` even with `return_dict=True`). All five
stages call through the single `run_and_show` helper that implements this.
Reuse for Stage 02 (SFT) and Stage 03 (inference) — see `gpt_oss_gotchas.md`
entry #1 for the full rationale.

## Open items (deferred to inspection phase)

1. **Stage 5b: CoT-quality judge.** Since the (CoT, polished_response) pair
   is coherent by construction, 5b doesn't need a divergence filter. What it
   should check:
   - Is the CoT actually first-person, forward-looking, and natural prose?
   - Does it reference specific elements of the visible inputs?
   - Does it engage with the substance of the objection?
   - Does it land at the polished response in a way that feels natural rather
     than forced?

   LLM-as-judge with a simple rubric; drop rows that fail multiple checks.
2. Per-stage reasoning-effort tuning once we see real outputs.
3. Whether to include more fields in `customer_context` / `product_context`.
4. Whether to hard-filter `cs_great_id == 0` rows up front or post-hoc.
5. Whether to keep `teacher_thinking` long-term, or drop once we've verified
   CoT quality (it's only diagnostic).
