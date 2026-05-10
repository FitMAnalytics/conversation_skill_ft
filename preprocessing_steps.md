# TSE SFT Preprocessing — What We Built

High-level summary of the preprocessing pipeline. For prompt text and
implementation details, see `0510_preprocessing_plan.md` and
`01_preprocessing.py`. Stage 5 was redesigned after the initial plan —
authoritative source for that stage is `0510_stage5_design.md`. For env
quirks hit along the way, see `gpt_oss_gotchas.md`.

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

- `analysis_channel  = teacher_analysis`   — Stage 5 CoT (Framing B rationale)
- `final_channel     = polished_response`  — Stage 2 polished agent turn

Diagnostic-only column (not a training target): `cot_grounded` ∈ {high, medium,
low}, the teacher's self-reported confidence that its CoT is fully derivable
from the visible inputs. Aggregated to surface info-asymmetry; see
`0510_stage5_design.md`.

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
  ├─ Stage 5  Teacher CoT (Framing B)        (high effort)
  │           shown Stage 2's polished response → reconstruct WHY it's a good move.
  │           analysis = student's CoT target; final = echo of polished response;
  │           ends with [grounded: high|medium|low] tag → cot_grounded column.
  │
  └─ Save preprocessed parquet
```

Filter cascade (enforced in code via `_eligibility_mask`): Stages 3, 4, 5 only
run on rows with `keep_stage1=True & exclude_stage2=False`. Stage 5b (CoT
quality judge per the design doc) is still TODO. Final SFT-ready filter:
`keep_stage1 & ~exclude_stage2 & cot_quality_passed_5b` (or just
`keep_stage1 & ~exclude_stage2` until 5b is implemented).

## Why each stage exists

- **Stage 1** drops rows where the agent didn't actually engage (pure "yeah,
  uh-huh"). Cheap reasoning — it's a coarse keep/drop decision.
- **Stage 2** is the source of truth for the student's `final_channel` target.
  The raw `agent_response_snippet` is fragmented and full of disfluencies; we
  need a clean single utterance without fabricating content.
- **Stage 3** is the conversational state at the moment of objection — what
  led here. Distinct from `objection_summary`, which describes the objection
  itself, not the surrounding state. Runs before Stage 4 because Stage 5 reads
  it directly.
- **Stage 4** gives the student long-horizon relationship memory the raw
  transcript doesn't carry. Cached because many objection rows share a
  customer (and therefore share `previous_calls`).
- **Stage 5** produces the teacher's reasoning under **Framing B**
  (rationalization). The teacher is shown the polished response and asked to
  reconstruct the agent's judgment chain in flowing prose, ending with a
  groundedness tag. Final channel echoes the polished response by construction,
  so there's no consistency-check filter — instead, Stage 5b (TODO) does an
  LLM-as-judge pass on CoT quality. Rationale: with our small corpus the human
  agent has hidden context the teacher doesn't, so a pure-derivation framing
  would systematically drop the most skilled examples. Full reasoning in
  `0510_stage5_design.md`.

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
stages call through the single `_generate` / `run_and_show` helper that
implements this. Reuse it for Stage 02 (SFT data formatting) and Stage 03
(inference) — see `gpt_oss_gotchas.md` entry #1 for the full rationale.

## Open items (deferred to inspection phase)

1. **Stage 5b CoT quality judge.** Under Framing B, 5b is no longer a
   consistency check — it's an LLM-as-judge pass on whether the analysis is
   coherent, references specific inputs, and is honest about groundedness.
   Rubric and rationale in `0510_stage5_design.md`.
2. **`cot_grounded` distribution review.** Once we have a few hundred rows,
   look at the high/medium/low split. A large `low` fraction signals critical
   context fields (account notes, CRM data) are missing — informs v2 schema.
3. Per-stage reasoning-effort tuning once we see real outputs.
4. Whether to include more fields in `customer_context` / `product_context`.
5. Whether to hard-filter `cs_great_id == 0` rows up front or post-hoc.
6. Whether to keep the `[grounded: ...]` tag in the student's analysis-channel
   target (currently stripped — student doesn't learn to emit it).
