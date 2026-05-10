# TSE SFT Data Preprocessing Pipeline

## Context

Build a preprocessing pipeline that transforms `TSE_by_objection.parquet` (one row per objection)
into SFT-ready training data for fine-tuning GPT-OSS-120B on outbound sales objection handling.

The teacher model is **GPT-OSS-120B**, used for all preprocessing stages: filtering, polishing,
summarization, and CoT generation. The student (same architecture) will be fine-tuned on the
output of this pipeline in a downstream SFT step.

## Constraints

- **Model loading**: fp16, no MXFP4 quantization (torch 2.6.0 environment).
  Multi-GPU via `device_map="auto"`. ~240GB for 120B in fp16, expect 3-4 H100s.
- **Format**: standard `transformers` + `peft` + `trl`. No `openai-harmony` library.
  Use `tokenizer.apply_chat_template` to handle harmony channel tokens.
- **Faulty GPUs**: cluster has 2 faulty H100s. Use `CUDA_VISIBLE_DEVICES` to target healthy ones.
- **No data leaves the corporate machine**.

## Deliverables

Two artifacts sharing the same prompt definitions and helper modules:

1. **`preprocess_notebook.ipynb`** (or `.py` cells with `# %%` markers) — example-by-example
   inspection, prompt iteration, side-by-side analysis/final channel display.
2. **`preprocess_batch.py`** — full-dataset batch processing, outputs preprocessed parquet.

Shared module: **`preprocess_lib.py`** containing model loading, prompt definitions,
channel splitting, customer/product context functions, and stage-wise processing functions.

---

## Pipeline Overview

```
Raw parquet
    │
    ├── Stage 1: Substantive turn filter         (drop non-substantive rows)
    │
    ├── Stage 2: Agent response polish           (clean fragments → polished response)
    │       └── secondary filter via exclude flag
    │
    ├── Stage 3: Previous-calls summarization    (cached per customer)
    │
    ├── Stage 4: Current-conversation summary    (per row)
    │
    ├── Stage 5: Teacher CoT + response          (high-effort reasoning)
    │       └── consistency check: teacher_final vs polished_response → drop inconsistent
    │
    └── Save preprocessed parquet
```

**Input/output flow per row** (after preprocessing):

```
Student inputs (at training & inference):
  - transcript_till_objection
  - objection_summary (from raw data)
  - current_call_summary (Stage 4)
  - previous_calls_summary (Stage 3)
  - customer_context (from get_customer_context)
  - product_context (from get_product_context)

Student targets (training only):
  - analysis_channel = teacher_analysis (Stage 5)
  - final_channel = polished_response (Stage 2)
```

---

## Module 1: `preprocess_lib.py`

### 1.1 Model loading

Load GPT-OSS-120B once in fp16, multi-GPU. Return `(model, tokenizer)`.

```python
def load_teacher_model(
    model_name: str = "openai/gpt-oss-120b",
    cuda_visible_devices: str = "0,1,2,3",  # adjust for healthy GPUs
) -> tuple:
    """
    Load teacher model in fp16 with device_map='auto' across visible GPUs.
    Returns (model, tokenizer).
    """
    # set CUDA_VISIBLE_DEVICES before importing torch if not already set
    # AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16, device_map="auto")
    # AutoTokenizer.from_pretrained(...)
    # model.eval()
    pass
```

### 1.2 Generation helper with channel splitting

Core helper that runs the model and splits output into analysis/final channels.

```python
def run_and_show(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "medium",   # "low" | "medium" | "high"
    max_new_tokens: int = 2048,
    verbose: bool = False,
) -> dict:
    """
    Run a single inference. Returns dict with keys:
      - 'analysis': content of analysis channel (CoT)
      - 'final':    content of final channel
      - 'raw':      raw decoded output (for debugging)

    Implementation notes:
      - Build messages list: [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
      - Use tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        reasoning_effort=reasoning_effort, return_tensors="pt")
      - Generate with do_sample=False (or low temperature) for preprocessing reproducibility
      - Decode full output, then split on harmony channel markers:
            "<|channel|>analysis<|message|>" ... "<|end|>"
            "<|channel|>final<|message|>" ... ("<|return|>" or "<|end|>")
      - If verbose: print both channels with separator
    """
    pass
```

Notes on channel splitting:
- The analysis channel ends at `<|end|>` followed by `<|start|>assistant<|channel|>final<|message|>`.
- The final channel ends at `<|return|>` (end of turn) or `<|end|>`.
- Be defensive: if channel markers are missing (model misformatted), fall back to treating
  full output as `final` and log a warning.

### 1.3 Batched generation

For batch processing in Stage X. Single-prompt loop is fine for v1; can optimize later.

```python
def run_batch(
    model,
    tokenizer,
    prompts: list,  # list of (system_prompt, user_prompt) tuples
    reasoning_effort: str = "medium",
    max_new_tokens: int = 2048,
    batch_size: int = 4,
) -> list:
    """
    Returns list of {'analysis', 'final', 'raw'} dicts.
    Use left-padding for batched generation. Set tokenizer.padding_side='left'.
    """
    pass
```

### 1.4 Customer context placeholder

Easy-to-edit function, multi-line string.

```python
def get_customer_context(row: dict) -> str:
    """
    Build a text block of customer information for prompt injection.
    Edit the multi-line string below to add/remove fields.
    Row is a single row from the dataframe (dict).
    """
    text = f"""\
Industry (SIC4): {row.get('sic4_industry', 'unknown')}
Revenue tier: {row.get('rev_tier', 'unknown')}
Employee count: {row.get('emp_ct', 'unknown')}
Customer tier: {row.get('tier', 'unknown')}
"""
    # add more fields as needed:
    # text += f"Other field: {row.get('other_field', 'unknown')}\n"
    return text
```

### 1.5 Product / campaign context placeholder

```python
def get_product_context(row: dict) -> str:
    """
    Build a text block of product / campaign information for prompt injection.
    Edit the multi-line string below to add/remove fields.
    """
    text = f"""\
Product type: {row.get('cw_opp_type', 'unknown')}
Campaign type: {row.get('task_type', 'unknown')}
Campaign subtype: {row.get('task_subtype', 'unknown')}
"""
    # add more fields as needed
    return text
```

---

## Module 2: Stage prompts

Each stage has a system prompt + user prompt template. Define them as module-level
constants so the notebook and batch script share the same definitions.

### Stage 1: Substantive turn filter

**Reasoning effort**: `low`
**Inputs**: `full_conversation_pii_rmv_till_obj`, `agent_response_snippet`
**Output (final channel, JSON)**: `{"keep": bool, "reason": str}`

System prompt focus:
- Define what "substantive" means: agent response addresses the customer's objection or
  hesitation with content beyond pure backchannels (yeah, uh-huh, right).
- Pure backchannels with no substantive content → not substantive.
- Acknowledgment + substantive question or answer → substantive.
- Note: incomplete responses are still potentially substantive — Stage 2 handles completeness.

User prompt template:
```
TRANSCRIPT (up to objection):
{transcript_till_obj}

AGENT RESPONSE TO EVALUATE:
{agent_response_snippet}

Determine whether this agent response is substantive enough to be a training example
for an objection-handling model. Return JSON: {"keep": bool, "reason": "<brief>"}
```

### Stage 2: Agent response polish

**Reasoning effort**: `medium`
**Inputs**: `agent_response_snippet`, `full_conversation_pii_rmv_till_obj`
**Output (final channel, JSON)**:
```json
{
  "polished_response": "<text>",
  "operations_applied": ["filler_removal", "repetition_collapse", "completion", "consolidation"],
  "completeness": "complete" | "completed_from_context" | "incomplete",
  "exclude": false,
  "exclude_reason": null
}
```

System prompt focus:
- Polish operations (uniform handling):
  1. Remove verbal fillers: um, uh, yeah (when not substantive agreement),
     I mean, you know, false starts. Preserve agent's actual phrasing and rhythm.
  2. Collapse repetitions: stutter-style ("the, the call") and phrase-level repeats.
  3. Complete obviously-truncated responses ONLY when trajectory is unambiguous from context.
     Do NOT invent claims, offers, prices, commitments, or product details.
  4. Consolidate multi-turn fragments separated by `|` into one coherent agent turn,
     dropping pure-backchannel fragments and merging substantive ones.
- Goal: produce one clean turn that fits a single agent utterance.
- Preserve agent voice — do NOT rewrite for "professionalism" beyond the operations above.
- If the response is so fragmented or incomplete that polishing would require fabrication,
  set `exclude: true` with a reason.

User prompt template:
```
TRANSCRIPT CONTEXT (for understanding what the agent was responding to):
{transcript_till_obj}

RAW AGENT RESPONSE (possibly fragmented, with fillers, possibly multiple turns merged):
{agent_response_snippet}

Apply minimal polish to produce a single coherent agent turn. Return JSON with the schema specified.
```

### Stage 3: Previous-calls summarization (cached per customer)

**Reasoning effort**: `medium`
**Input**: `previous_calls` (raw concatenated transcripts)
**Output (final channel, plain text)**: structured summary string

Caching: many rows share the same customer (and therefore same `previous_calls`).
Build cache keyed on customer ID (or hash of `previous_calls` if customer ID isn't reliable).
Skip rows where `previous_calls` is empty/null.

System prompt focus:
- Summarize prior interactions chronologically.
- Capture: products discussed, customer concerns raised, objections from prior calls,
  outcomes/next steps agreed on, customer's communication style and engagement level.
- Length target: 150–300 words depending on history depth.
- Output format: prose paragraphs, no bullets (clean for prompt injection).

User prompt template:
```
PREVIOUS CALLS WITH THIS CUSTOMER (chronological, raw transcripts):
{previous_calls_raw}

Summarize the relationship history. Cover products discussed, customer concerns and objections,
outcomes, and the customer's communication style. Output prose only, no bullets or headers.
```

### Stage 4: Current-conversation summary

**Reasoning effort**: `medium`
**Input**: `full_conversation_pii_rmv_till_obj`
**Output (final channel, plain text)**: state-of-call summary

System prompt focus:
- Summarize the call so far up to the objection point.
- Capture: how the call opened, what's been discussed, customer's engagement signals,
  what led to the current objection.
- This is DISTINCT from `objection_summary` (which describes the objection itself).
- This describes the broader conversation state.
- Length target: 80–150 words.
- Output: prose, no bullets.

User prompt template:
```
CURRENT CALL TRANSCRIPT (up to the objection point):
{transcript_till_obj}

Summarize the state of the conversation: how the call opened, what's been discussed,
customer's engagement, and what led to the current point. Output prose only.
```

### Stage 5: Teacher CoT + response generation

**Reasoning effort**: `high`
**Inputs** (everything): transcript_till_obj, objection_summary, current_call_summary,
previous_calls_summary, customer_context, product_context
**Output**:
- `teacher_analysis`: content of analysis channel (CoT) — this becomes the student's
  analysis-channel training target.
- `teacher_final`: content of final channel — used for consistency check against polished_response.

System prompt focus:
- You are an expert outbound sales agent at American Express on the TSE channel.
- Goal: deepen relationship with existing card-holding business customers, increase usage,
  expand spend across vendors/employees/payment types.
- Given the conversation state, customer profile, product, and prior history, reason
  carefully about the customer's objection and produce the ideal next agent turn.
- Reasoning should walk: customer state → underlying concern → strategic options → chosen move → response.
- Final response should be one natural agent turn — no script, no formatting, just what the
  agent should say next.

User prompt template:
```
CUSTOMER CONTEXT:
{customer_context}

PRODUCT / CAMPAIGN CONTEXT:
{product_context}

PREVIOUS RELATIONSHIP HISTORY:
{previous_calls_summary}

CURRENT CALL SO FAR:
{current_call_summary}

CURRENT OBJECTION:
{objection_summary}

FULL TRANSCRIPT UP TO THIS POINT:
{transcript_till_obj}

Reason about the customer's objection and produce the ideal next agent response.
Reasoning goes in the analysis channel; the response goes in the final channel.
The final-channel response should be a single natural agent turn, no formatting.
```

### Stage 5b: Consistency check (no LLM call needed for v1)

For each row, compare `teacher_final` and `polished_response`. v1 implementation:

- Compute simple similarity: ROUGE-L or sentence-embedding cosine.
- Set a threshold (e.g., embedding cosine > 0.5) for "consistent enough".
- Below threshold → set `consistency_flag = False`, exclude from training.

Or, as an alternative for v1: run a lightweight LLM judge call asking
"are these two responses pursuing the same strategic move?" → keep/drop.

Mark this stage as **TODO: pick approach during inspection**. Don't block pipeline build on it.

---

## Module 3: `preprocess_notebook.ipynb`

Cell-by-cell, oriented around prompt iteration.

### Cell 1: Setup
- Import lib
- Set `CUDA_VISIBLE_DEVICES`
- Load model + tokenizer
- Load parquet, sample a few diverse rows for inspection

### Cell 2: Pick an example row
```python
row_idx = 0
row = df.iloc[row_idx].to_dict()
print(row['full_conversation_pii_rmv_till_obj'][:500])
print("---")
print(row['agent_response_snippet'])
```

### Cell 3: Stage 1 — Filter (single example)
- Build prompt
- Call `run_and_show(..., verbose=True)` — prints both channels
- Inspect, iterate prompt

### Cell 4: Stage 2 — Polish (single example)
- Same pattern: build prompt, run, print both channels, parse JSON, inspect output

### Cell 5: Stage 3 — Previous-calls summary (single example)
- Skip if no previous calls; otherwise show input + summary

### Cell 6: Stage 4 — Current-call summary (single example)

### Cell 7: Stage 5 — Teacher CoT + response (single example)
- Print analysis channel and final channel separately
- Compare final channel against polished response from Stage 2

### Cell 8: Run all stages on a small batch (e.g., 10 rows)
- Useful for spotting edge cases before full run
- Print summary stats: how many filtered out, completeness distribution, etc.

The notebook is the prompt iteration loop. Don't move to batch script until Cell 7 looks good
across 5–10 diverse examples.

---

## Module 4: `preprocess_batch.py`

CLI script for full-dataset processing.

### Structure

```python
"""
Usage:
  python preprocess_batch.py \
      --input  TSE_by_objection.parquet \
      --output TSE_preprocessed.parquet \
      --stages 1,2,3,4,5 \
      --resume \
      --batch-size 4 \
      --cuda-visible 0,1,2,3
"""
```

### Argument parsing
- `--input`, `--output`: paths
- `--stages`: comma-separated list of stages to run (default: all)
- `--resume`: pick up from existing partial output (checkpointing)
- `--batch-size`: generation batch size
- `--cuda-visible`: GPU IDs
- `--limit`: optional, process only first N rows (for smoke testing)
- `--reasoning-effort-overrides`: optional, e.g., `"5:medium"` to override Stage 5 to medium

### Main flow

```python
def main(args):
    set_cuda_visible(args.cuda_visible)
    model, tokenizer = load_teacher_model()
    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)

    # Stage 1: filter
    if "1" in args.stages:
        df = run_stage_1_filter(df, model, tokenizer, batch_size=args.batch_size)
        save_checkpoint(df, args.output, stage=1)

    # Stage 2: polish
    if "2" in args.stages:
        df = run_stage_2_polish(df, model, tokenizer, batch_size=args.batch_size)
        save_checkpoint(df, args.output, stage=2)

    # Stage 3: previous-calls summary (cached per customer)
    if "3" in args.stages:
        df = run_stage_3_prev_summary(df, model, tokenizer, batch_size=args.batch_size)
        save_checkpoint(df, args.output, stage=3)

    # Stage 4: current-call summary
    if "4" in args.stages:
        df = run_stage_4_current_summary(df, model, tokenizer, batch_size=args.batch_size)
        save_checkpoint(df, args.output, stage=4)

    # Stage 5: teacher CoT + response
    if "5" in args.stages:
        df = run_stage_5_teacher(df, model, tokenizer, batch_size=args.batch_size)
        save_checkpoint(df, args.output, stage=5)

    # Save final
    df.to_parquet(args.output)
    print(f"Saved {len(df)} preprocessed rows to {args.output}")
```

### Checkpointing
After each stage, save partial parquet to disk. On resume, detect which stages have been
applied (presence of expected columns) and skip them.

### Logging
- tqdm progress bars per stage
- Print stats after each stage: rows in / rows out, time elapsed, avg tokens/sec
- Log failures (parse errors, generation errors) to a sidecar file but don't crash the pipeline
- For failed rows: keep them in the dataframe with a `stage_X_error` flag so they can be retried

### Detached execution
Standard pattern for the cluster:
```bash
nohup python preprocess_batch.py \
    --input TSE_by_objection.parquet \
    --output TSE_preprocessed.parquet \
    --stages 1,2,3,4,5 \
    --batch-size 4 \
    --cuda-visible 0,1,2,3 \
    > preprocess.log 2>&1 &
disown
```

---

## Output schema

Final preprocessed parquet columns (additions to original):

| Column | Source | Type |
|---|---|---|
| `keep_stage1` | Stage 1 | bool |
| `polished_response` | Stage 2 | str |
| `polish_operations` | Stage 2 | list[str] |
| `completeness` | Stage 2 | str |
| `exclude_stage2` | Stage 2 | bool |
| `previous_calls_summary` | Stage 3 | str (or null if no prior) |
| `current_call_summary` | Stage 4 | str |
| `teacher_analysis` | Stage 5 | str |
| `teacher_final` | Stage 5 | str |
| `consistency_flag` | Stage 5b | bool |
| `customer_context` | from get_customer_context | str |
| `product_context` | from get_product_context | str |

Filter for SFT-ready rows: `keep_stage1 & ~exclude_stage2 & consistency_flag`.

---

## Inspection harness checklist (before running full batch)

For each stage prompt, verify on 5–10 diverse rows:

- [ ] Stage 1: filter correctly drops fragmented/empty agent responses, keeps substantive ones
- [ ] Stage 2: polish removes fillers and consolidates fragments without fabricating content
- [ ] Stage 2: completeness flag fires on truncated responses
- [ ] Stage 3: summary is coherent, captures key history, ~150-300 words
- [ ] Stage 4: summary is distinct from objection_summary, focuses on call state
- [ ] Stage 5: analysis channel walks through state → options → move → response
- [ ] Stage 5: final channel produces a single natural agent turn (no headers, no bullets)
- [ ] Stage 5: teacher_final and polished_response are usually similar (most pass consistency)

If any of these fail, iterate the prompt in the notebook before running batch.

---

## Open items / decisions deferred to inspection phase

1. **Stage 5b consistency check approach**: similarity threshold vs LLM judge. Decide
   after seeing a few examples.
2. **Reasoning effort tuning per stage**: defaults are starting points; may need adjustment
   based on output quality and speed.
3. **Customer context fields**: starts minimal, add fields as judgment dictates after
   seeing a few CoT outputs.
4. **Whether to drop rows where `cs_great_id == 0`**: discussed but not committed. Could be
   a Stage 0 hard filter, or could be left to the LLM-based filtering. Default: leave for now,
   filter post-hoc if needed.
