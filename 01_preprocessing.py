"""TSE SFT preprocessing pipeline (5 stages on a per-objection parquet).

Stages
------
  1. Substantive turn filter        keep_stage1, stage1_reason
  2. Agent response polish          polished_response, polish_operations,
                                    completeness, exclude_stage2, exclude_reason
  3. Current-call summary           current_call_summary
  4. Previous-calls summary         previous_calls_summary  (cached per customer)
  5. Teacher rationalization CoT    teacher_cot, teacher_thinking
                                    teacher IS shown polished_response and writes
                                    a forward-looking first-person CoT (in the
                                    final channel) that reconstructs the agent's
                                    reasoning and lands at the given response.
                                    teacher_cot (final channel) is the student's
                                    analysis-channel SFT target; teacher_thinking
                                    (analysis channel) is OSS's free-form planning,
                                    stored as diagnostic only. See
                                    0510_stage5_design.md.
  + customer_context, product_context columns for downstream SFT.

Importable as a library (notebook uses it) AND runnable as a CLI batch script.

Batch usage:
    python 01_preprocessing.py \
        --input  TSE_by_objection.parquet \
        --output TSE_preprocessed.parquet \
        --stages 1,2,3,4,5 \
        --cuda-visible 0,1,2,3

Auto-save + auto-resume: each stage writes the parquet every
--auto-save-batch-size rows (default 50). If the output parquet already
exists at startup, it is loaded and resumed from automatically — each stage
runner tracks per-row completion in a `_stage{N}_done` boolean column and
skips rows that are already done. Killing the script mid-stage and
restarting picks up where it left off.

Filter cascade: stages 3, 4, 5 only run on rows that survive both
`keep_stage1=True` and `exclude_stage2=False`. Stage 2 only runs on rows with
`keep_stage1=True`. Ineligible rows get empty/None values in the new columns
so the schema stays consistent across the parquet.

Internal tracker columns (`_stage1_done` ... `_stage5_done`) live in the
parquet so resume works; downstream consumers can ignore them.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

# NB: callers that want to override CUDA_VISIBLE_DEVICES must do so BEFORE
# importing this module (or before any function here calls torch). The CLI
# entrypoint sets it from --cuda-visible before importing torch lazily.

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"


# ---------------------------------------------------------------------------
# Customer / product context (edit these freely)
# ---------------------------------------------------------------------------

def get_customer_context(row: dict) -> str:
    """Multi-line text block of customer info for prompt injection."""
    return (
        f"Industry (SIC4): {row.get('sic4_industry', 'unknown')}\n"
        f"Revenue tier:    {row.get('rev_tier', 'unknown')}\n"
        f"Employee count:  {row.get('emp_ct', 'unknown')}\n"
        f"Customer tier:   {row.get('tier', 'unknown')}\n"
    )


def get_product_context(row: dict) -> str:
    """Multi-line text block of product / campaign info for prompt injection."""
    return (
        f"Product type:      {row.get('cw_opp_type', 'unknown')}\n"
        f"Campaign type:     {row.get('task_type', 'unknown')}\n"
        f"Campaign subtype:  {row.get('task_subtype', 'unknown')}\n"
    )


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_teacher_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Load GPT-OSS-120B in bf16 sharded across CUDA_VISIBLE_DEVICES.

    `torch_dtype=torch.bfloat16` forces dequantized expert weights through
    `device_map="auto"` instead of staging on cuda:0 (which OOMs the first GPU
    on torch 2.6 / triton 3.2 with `dtype="auto"`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(?P<content>.*?)(?=<\|end\|>|<\|start\|>|<\|return\|>|\Z)",
    re.DOTALL,
)
_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(?P<content>.*?)(?=<\|return\|>|<\|end\|>|\Z)",
    re.DOTALL,
)


def _split_channels(raw: str) -> tuple[str, str]:
    """Pull analysis + final channels out of a harmony-formatted decode."""
    a = _ANALYSIS_RE.findall(raw)
    f = _FINAL_RE.findall(raw)
    analysis = a[-1].strip() if a else ""
    final = f[-1].strip() if f else raw.strip()  # fallback: treat all as final
    return analysis, final


def run_and_show(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "medium",
    max_new_tokens: int = 2048,
    verbose: bool = False,
) -> dict:
    """Single-prompt inference. Returns {'analysis', 'final', 'raw'}."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Two-step rendering: chat-template -> string, then tokenize. Avoids the
    # version-dependent return shape of apply_chat_template(return_dict=True),
    # which on some gpt-oss tokenizers nests dicts under input_ids and breaks
    # downstream `.shape` access.
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0, input_len:], skip_special_tokens=False)
    analysis, final = _split_channels(raw)

    if verbose:
        print("=" * 30, "ANALYSIS", "=" * 30)
        print(analysis or "(empty)")
        print("=" * 30, "FINAL", "=" * 33)
        print(final or "(empty)")
        print("=" * 72)
    return {"analysis": analysis, "final": final, "raw": raw}


def _parse_json_block(text: str) -> dict:
    """Extract outermost {...}; raises ValueError if none found."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in: {text[:300]}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Shared background — prepended to every stage's system prompt so the teacher
# knows what kind of call it's looking at.
# ---------------------------------------------------------------------------

TSE_BACKGROUND = """\
Background — TSE channel:
This data comes from American Express's TSE (Tele Strategic Expansion) channel.
The TSE channel engages existing American Express business customers — companies
that already hold Amex cards but are not yet fully utilizing Amex across their
total business spend. The agent's role is to expand the relationship and
increase business spend on Amex products by surfacing opportunities to grow
share of wallet across vendors, employees, and payment types."""


# ---------------------------------------------------------------------------
# Stage 1: Substantive turn filter
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = """You are a senior sales-call transcript analyst working on an outbound-sales SFT pipeline.
""" + TSE_BACKGROUND + """
Decide whether the (objection, agent response) pair is suitable as a training example
for an objection-handling model.

A pair is SUITABLE ("keep": true) when the agent response contains learnable
content that engages with the customer's objection or hesitation.

A pair is NOT SUITABLE ("keep": false) when ANY of the following apply:

1. PURE BACKCHANNEL: agent response is only acknowledgments (yeah, uh-huh, right,
   mm-hmm, okay) with no substantive content.

2. COMPLIANCE / DISCLOSURE: agent is delivering required disclosures, consent
   language, "this call is recorded," regulatory script.

3. CALL OPENING / CLOSING PLEASANTRIES: greetings, sign-offs, "have a great day."

4. IDENTITY VERIFICATION / PROCEDURAL: account lookup, last-4 confirmation,
   transfer handoffs, escalation language.

A response with substantive content can still be suitable even if it is
incomplete or fragmented — completeness is judged in a later stage. Do not drop
rows just because the response is cut off.

Return ONLY a JSON object in the final channel:
  {"keep": true|false, "reason": "<one short sentence citing which rule applied>"}
"""

STAGE1_USER_TMPL = """TRANSCRIPT (up to objection):
{transcript}

AGENT RESPONSE TO EVALUATE:
{response}

Return JSON: {{"keep": bool, "reason": "<brief>"}}
"""


def stage1_filter_one(row: dict, model, tokenizer) -> dict:
    user = STAGE1_USER_TMPL.format(
        transcript=row["full_conversation_pii_rmv_till_obj"],
        response=row["agent_response_snippet"],
    )
    out = run_and_show(model, tokenizer, STAGE1_SYSTEM, user,
                       reasoning_effort="low", max_new_tokens=256)
    parsed = _parse_json_block(out["final"])
    return {"keep_stage1": bool(parsed["keep"]),
            "stage1_reason": str(parsed.get("reason", ""))}


# ---------------------------------------------------------------------------
# Stage 2: Agent response polish
# ---------------------------------------------------------------------------

STAGE2_SYSTEM = """You are a senior sales-call transcript analyst working on an outbound-sales SFT pipeline.

""" + TSE_BACKGROUND + """

Given the conversation context and a possibly-fragmented raw agent response,
produce ONE polished agent turn that fits a single utterance.

Polish operations (apply uniformly, in order of priority):
  1. filler_removal      Remove um, uh, false starts, "I mean", "you know",
                         and "yeah"/"right" only when used as a filler. Preserve
                         the agent's wording and rhythm; do NOT paraphrase.
  2. repetition_collapse Collapse stutter ("the, the call") and phrase repeats.
  3. completion          Complete obviously-truncated responses ONLY when the
                         trajectory is unambiguous from context. Do NOT invent
                         claims, prices, offers, commitments, or product facts.
  4. consolidation       Multiple agent fragments separated by "|" should be
                         merged into one coherent turn; drop pure-backchannel
                         fragments, merge substantive ones.

Do NOT rewrite for "professionalism" beyond these operations. Preserve voice.
If polishing would require fabrication, set exclude=true with a reason.

Return ONLY JSON in the final channel:
  {
    "polished_response": "<one clean agent turn>",
    "operations_applied": ["filler_removal", "repetition_collapse", ...],
    "completeness": "complete" | "completed_from_context" | "incomplete",
    "exclude": false,
    "exclude_reason": null
  }
"""

STAGE2_USER_TMPL = """TRANSCRIPT CONTEXT (what the agent was responding to):
{transcript}

RAW AGENT RESPONSE (possibly fragmented, with fillers):
{response}

Apply minimal polish per the schema. Return JSON only.
"""


def stage2_polish_one(row: dict, model, tokenizer) -> dict:
    user = STAGE2_USER_TMPL.format(
        transcript=row["full_conversation_pii_rmv_till_obj"],
        response=row["agent_response_snippet"],
    )
    out = run_and_show(model, tokenizer, STAGE2_SYSTEM, user,
                       reasoning_effort="medium", max_new_tokens=1024)
    parsed = _parse_json_block(out["final"])
    return {
        "polished_response": str(parsed.get("polished_response", "")),
        "polish_operations": list(parsed.get("operations_applied", []) or []),
        "completeness": str(parsed.get("completeness", "")),
        "exclude_stage2": bool(parsed.get("exclude", False)),
        "exclude_reason": parsed.get("exclude_reason"),
    }


# ---------------------------------------------------------------------------
# Stage 3: Current-call summary
# ---------------------------------------------------------------------------

STAGE3_SYSTEM = """You are a senior sales-call transcript analyst summarizing the state of an in-progress call.

""" + TSE_BACKGROUND + """

Given the current call transcript up to the point of the customer's objection,
write an 50 - 300 word prose summary of the conversation state:
  - how the call opened
  - what's been discussed
  - the customer's engagement signals
  - what led to the current point

This is DISTINCT from the objection summary (which describes the objection
itself). You are describing the broader conversation state.

Output prose only. No bullets, no headers. Return the summary in the final channel."""

STAGE3_USER_TMPL = """CURRENT CALL TRANSCRIPT (up to the objection point):
{transcript}

Summarize the state of the conversation per the instructions. Prose only.
"""


def stage3_current_summary_one(row: dict, model, tokenizer) -> str:
    user = STAGE3_USER_TMPL.format(
        transcript=row["full_conversation_pii_rmv_till_obj"]
    )
    out = run_and_show(model, tokenizer, STAGE3_SYSTEM, user,
                       reasoning_effort="medium", max_new_tokens=512)
    return out["final"].strip()


# ---------------------------------------------------------------------------
# Stage 4: Previous-calls summary (cached per customer)
# ---------------------------------------------------------------------------

STAGE4_SYSTEM = """You are a senior sales-call transcript analyst summarizing relationship history.

""" + TSE_BACKGROUND + """

Given the raw concatenated transcripts of prior calls with one customer, write
a 150-300 word prose summary covering:
  - products discussed
  - customer concerns and objections raised in prior calls
  - outcomes / next steps that were agreed
  - the customer's communication style and engagement level

Output prose paragraphs only. No bullets, no headers, no preamble.
Return the summary text in the final channel."""

STAGE4_USER_TMPL = """PREVIOUS CALLS WITH THIS CUSTOMER (chronological, raw):
{previous_calls}

Summarize the relationship history per the instructions. Prose only.
"""


def _customer_cache_key(row: dict) -> str:
    prev = row.get("previous_calls") or ""
    if not prev.strip():
        return ""
    cust = row.get("customer_id") or row.get("cust_id") or ""
    if cust:
        return f"cust:{cust}"
    return "hash:" + hashlib.sha1(prev.encode("utf-8")).hexdigest()


def stage4_prev_summary_one(row: dict, model, tokenizer) -> str | None:
    prev = row.get("previous_calls") or ""
    if not prev.strip():
        return None
    user = STAGE4_USER_TMPL.format(previous_calls=prev)
    out = run_and_show(model, tokenizer, STAGE4_SYSTEM, user,
                       reasoning_effort="medium", max_new_tokens=768)
    return out["final"].strip()


# ---------------------------------------------------------------------------
# Stage 5: Rationalization CoT. Teacher IS shown polished_response and writes
# a forward-looking first-person CoT that reconstructs the agent's reasoning
# and lands at the given response. CoT goes in the FINAL channel (becomes the
# student's analysis-channel SFT target); the model's analysis channel is its
# own free-form planning, stored as diagnostic only. SFT pair downstream is
# (teacher_cot, polished_response) — coherent by construction (the teacher
# was shown the response and asked to write a CoT that leads to it). See
# 0510_stage5_design.md.
# ---------------------------------------------------------------------------

STAGE5_SYSTEM = """You are an expert outbound sales agent at American Express on the TSE
(Tele Strategic Expansion) channel.

""" + TSE_BACKGROUND + """

You will be given everything an agent has in front of them at the moment of an
objection — the customer's profile, product/campaign context, prior-call
history, current-call summary, the objection itself, and the transcript up to
that point — PLUS the response the agent actually gave (a polished version of
their actual words).

Your task is to RECONSTRUCT the agent's forward reasoning — the chain of
thought a skilled agent would have run silently between hearing the objection
and speaking their next turn, ending exactly at the given response. You are
not deciding what to say; the agent already chose. Your job is to write the
thinking that leads to that choice. Even if you personally would have picked
a different move, do not propose alternatives — reconstruct the reasoning
that explains THIS response.

The reconstructed CoT goes in the FINAL channel. It must:
  - Be in first person, present tense ("the customer is pushing back on
    price... given their industry, this likely reflects..."). It should
    read as if the agent is thinking it BEFORE speaking, not as a retrospective
    explanation. Do NOT reference the response as something already given;
    write the reasoning that arrives at it.
  - Flow as natural prose. No bullets, headers, JSON, or enumerated steps.
  - Ground in the visible inputs: what does this customer's industry and
    history suggest, what has already happened in this call, what is the
    objection actually signaling beneath its wording, what move best serves
    the relationship and the call objective.
  - Land naturally at the given response — the reasoning should make the
    response feel like the obvious next move, not a leap.
  - Be as long as the case warrants and no more. Simple objections warrant
    brief reasoning; complex multi-signal cases warrant more. Do not pad,
    do not restate inputs, do not second-guess in circles.

The ANALYSIS channel is yours. Use it however helps you plan a good CoT —
draft, weigh framings, check that the reasoning lands at the actual response,
whatever. It will not be used as a training target. Don't worry about its
form."""

STAGE5_USER_TMPL = """CUSTOMER CONTEXT:
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
{transcript}

THE RESPONSE THE AGENT GAVE:
{polished_agent_response}

Reconstruct the agent's forward reasoning that led to this exact response.
The CoT goes in the final channel — first person, present tense, flowing
prose, grounded in the visible inputs, landing at the given response.
Use the analysis channel however helps you plan the CoT.
"""


def stage5_teacher_one(row: dict, model, tokenizer) -> dict:
    polished = row.get("polished_response", "")
    if not polished:
        # Stage 5 needs the polished response as input. The eligibility mask
        # should already exclude rows where Stage 2 didn't produce one.
        raise ValueError("stage 5 requires polished_response from stage 2")
    user = STAGE5_USER_TMPL.format(
        customer_context=row.get("customer_context", ""),
        product_context=row.get("product_context", ""),
        previous_calls_summary=row.get("previous_calls_summary") or "(no prior calls)",
        current_call_summary=row.get("current_call_summary", ""),
        objection_summary=row.get("objection_summary", ""),
        transcript=row["full_conversation_pii_rmv_till_obj"],
        polished_agent_response=polished,
    )
    out = run_and_show(model, tokenizer, STAGE5_SYSTEM, user,
                       reasoning_effort="high", max_new_tokens=4096)
    return {
        # CoT lives in the FINAL channel — this becomes the student's
        # analysis-channel SFT target.
        "teacher_cot": out["final"],
        # Diagnostic: whatever OSS wrote in its analysis channel while
        # planning the CoT. Not a training target.
        "teacher_thinking": out["analysis"],
    }


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

# Output columns each stage adds — used for resume detection.
STAGE_COLS = {
    "1": ["keep_stage1", "stage1_reason"],
    "2": ["polished_response", "polish_operations", "completeness",
          "exclude_stage2", "exclude_reason"],
    "3": ["current_call_summary"],
    "4": ["previous_calls_summary"],
    "5": ["teacher_cot", "teacher_thinking"],
}


def _save_checkpoint(df, output_path: Path, *, stage_label: str = "",
                     verbose: bool = True) -> None:
    """Atomically write df to parquet via a .tmp + os.replace dance."""
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, output_path)
    if verbose:
        logging.info("Checkpoint saved after stage %s -> %s (%d rows)",
                     stage_label, output_path, len(df))
    else:
        logging.debug("Auto-save -> %s (%d rows)", output_path, len(df))


def _maybe_autosave(df, save_path: Path | None, save_every: int | None,
                    processed: int, stage_label: str) -> None:
    """Save the in-progress dataframe every `save_every` processed rows."""
    if save_path is None or not save_every:
        return
    if processed > 0 and processed % save_every == 0:
        _save_checkpoint(df, save_path, stage_label=stage_label, verbose=False)


def _eligible_after_stage_1(df):
    """For Stage 2: row is eligible if it survived Stage 1 (`keep_stage1=True`).

    Stage 2 must NOT use `exclude_stage2` in its eligibility check — that's
    Stage 2's own output, and using it would short-circuit Stage 2 to a no-op
    on resume (since `_init_col` defaults `exclude_stage2=True` for unprocessed
    rows).
    """
    import pandas as pd
    if "keep_stage1" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    return df["keep_stage1"].fillna(False).astype(bool)


def _eligible_after_stage_2(df):
    """For Stages 3/4/5: row eligible if it survived both Stage 1 and Stage 2.

    If `keep_stage1` / `exclude_stage2` aren't present yet (e.g. user ran a
    later stage standalone), defaults to True so we don't silently no-op.
    """
    import pandas as pd
    mask = _eligible_after_stage_1(df)
    if "exclude_stage2" in df.columns:
        mask = mask & ~df["exclude_stage2"].fillna(True).astype(bool)
    return mask


# Back-compat alias — older code/tests may reference the old name.
_eligibility_mask = _eligible_after_stage_2


def _init_col(df, col: str, default):
    """Add `col` filled with `default` if not already present. Object dtype so
    we can hold mixed/Optional values cleanly across parquet round-trips."""
    import pandas as pd
    if col not in df.columns:
        df[col] = pd.Series([default] * len(df), dtype=object)


# Each stage runner uses a `_stage{N}_done` boolean tracker so we can skip
# already-processed rows on resume and on partial mid-stage saves. The tracker
# columns live in the parquet alongside the real outputs.

def run_stage_1(df, model, tokenizer, *,
                save_path: Path | None = None,
                save_every: int | None = None):
    df = df.copy()
    _init_col(df, "_stage1_done", False)
    _init_col(df, "keep_stage1", False)
    _init_col(df, "stage1_reason", "")

    from tqdm import tqdm
    indices = df.index.tolist()
    processed = 0
    for i in tqdm(range(len(df)), desc="stage1-filter"):
        idx = indices[i]
        if bool(df.at[idx, "_stage1_done"]):
            continue
        row = df.iloc[i].to_dict()
        try:
            result = stage1_filter_one(row, model, tokenizer)
            df.at[idx, "keep_stage1"] = bool(result["keep_stage1"])
            df.at[idx, "stage1_reason"] = str(result.get("stage1_reason", ""))
        except Exception as e:  # noqa: BLE001
            logging.warning("stage1 row %d failed: %s", i, e)
            df.at[idx, "keep_stage1"] = False
            df.at[idx, "stage1_reason"] = f"_error: {e}"
        df.at[idx, "_stage1_done"] = True
        processed += 1
        _maybe_autosave(df, save_path, save_every, processed, "1")
    return df


def run_stage_2(df, model, tokenizer, *,
                save_path: Path | None = None,
                save_every: int | None = None):
    df = df.copy()
    _init_col(df, "_stage2_done", False)
    # Compute eligibility BEFORE initializing exclude_stage2 — otherwise the
    # init default contaminates the mask. Also use the stage-1-only check;
    # exclude_stage2 is *our* output, not an input we should gate on.
    eligible = _eligible_after_stage_1(df).tolist()
    _init_col(df, "polished_response", "")
    _init_col(df, "polish_operations", [])
    _init_col(df, "completeness", "")
    _init_col(df, "exclude_stage2", True)
    _init_col(df, "exclude_reason", None)

    from tqdm import tqdm
    indices = df.index.tolist()
    processed = 0
    for i in tqdm(range(len(df)), desc="stage2-polish"):
        idx = indices[i]
        if bool(df.at[idx, "_stage2_done"]):
            continue
        if not eligible[i]:
            # ineligible: leave defaults (polished_response="", exclude_stage2=True)
            df.at[idx, "_stage2_done"] = True
            continue
        row = df.iloc[i].to_dict()
        try:
            result = stage2_polish_one(row, model, tokenizer)
            df.at[idx, "polished_response"] = str(result.get("polished_response", ""))
            df.at[idx, "polish_operations"] = list(result.get("polish_operations", []) or [])
            df.at[idx, "completeness"] = str(result.get("completeness", ""))
            df.at[idx, "exclude_stage2"] = bool(result.get("exclude_stage2", False))
            df.at[idx, "exclude_reason"] = result.get("exclude_reason")
        except Exception as e:  # noqa: BLE001
            logging.warning("stage2 row %d failed: %s", i, e)
            df.at[idx, "polished_response"] = ""
            df.at[idx, "polish_operations"] = []
            df.at[idx, "completeness"] = ""
            df.at[idx, "exclude_stage2"] = True
            df.at[idx, "exclude_reason"] = f"_error: {e}"
        df.at[idx, "_stage2_done"] = True
        processed += 1
        _maybe_autosave(df, save_path, save_every, processed, "2")
    return df


def run_stage_3(df, model, tokenizer, *,
                save_path: Path | None = None,
                save_every: int | None = None):
    df = df.copy()
    _init_col(df, "_stage3_done", False)
    _init_col(df, "current_call_summary", "")

    from tqdm import tqdm
    eligible = _eligible_after_stage_2(df).tolist()
    indices = df.index.tolist()
    processed = 0
    for i in tqdm(range(len(df)), desc="stage3-current-summary"):
        idx = indices[i]
        if bool(df.at[idx, "_stage3_done"]):
            continue
        if not eligible[i]:
            df.at[idx, "_stage3_done"] = True
            continue
        row = df.iloc[i].to_dict()
        try:
            s = stage3_current_summary_one(row, model, tokenizer)
            df.at[idx, "current_call_summary"] = s or ""
        except Exception as e:  # noqa: BLE001
            logging.warning("stage3 row %d failed: %s", i, e)
            df.at[idx, "current_call_summary"] = ""
        df.at[idx, "_stage3_done"] = True
        processed += 1
        _maybe_autosave(df, save_path, save_every, processed, "3")
    return df


def run_stage_4(df, model, tokenizer, *,
                save_path: Path | None = None,
                save_every: int | None = None):
    df = df.copy()
    _init_col(df, "_stage4_done", False)
    _init_col(df, "previous_calls_summary", None)

    from tqdm import tqdm
    eligible = _eligible_after_stage_2(df).tolist()
    indices = df.index.tolist()
    records = df.to_dict("records")

    # Pre-populate per-customer cache from rows already processed in a prior run.
    cache: dict[str, str | None] = {}
    for i in range(len(df)):
        if bool(df.at[indices[i], "_stage4_done"]):
            key = _customer_cache_key(records[i])
            if key and key not in cache:
                cache[key] = df.at[indices[i], "previous_calls_summary"]

    processed = 0
    for i in tqdm(range(len(df)), desc="stage4-prev-summary"):
        idx = indices[i]
        if bool(df.at[idx, "_stage4_done"]):
            continue
        if not eligible[i]:
            df.at[idx, "_stage4_done"] = True
            continue
        key = _customer_cache_key(records[i])
        if key == "":
            df.at[idx, "_stage4_done"] = True
            continue
        if key in cache:
            df.at[idx, "previous_calls_summary"] = cache[key]
            df.at[idx, "_stage4_done"] = True
            # Cache hits are free — don't count toward autosave cadence.
            continue
        try:
            s = stage4_prev_summary_one(records[i], model, tokenizer)
        except Exception as e:  # noqa: BLE001
            logging.warning("stage4 failed for key %s: %s", key, e)
            s = None
        cache[key] = s
        df.at[idx, "previous_calls_summary"] = s
        df.at[idx, "_stage4_done"] = True
        processed += 1
        _maybe_autosave(df, save_path, save_every, processed, "4")
    return df


def run_stage_5(df, model, tokenizer, *,
                save_path: Path | None = None,
                save_every: int | None = None):
    df = df.copy()
    _init_col(df, "_stage5_done", False)
    _init_col(df, "teacher_cot", "")
    _init_col(df, "teacher_thinking", "")
    # Stage 5 reads customer/product context; ensure they exist.
    if "customer_context" not in df.columns:
        df["customer_context"] = [get_customer_context(r) for r in df.to_dict("records")]
    if "product_context" not in df.columns:
        df["product_context"] = [get_product_context(r) for r in df.to_dict("records")]

    from tqdm import tqdm
    eligible = _eligible_after_stage_2(df).tolist()
    indices = df.index.tolist()
    processed = 0
    for i in tqdm(range(len(df)), desc="stage5-teacher"):
        idx = indices[i]
        if bool(df.at[idx, "_stage5_done"]):
            continue
        if not eligible[i]:
            df.at[idx, "_stage5_done"] = True
            continue
        row = df.iloc[i].to_dict()
        try:
            result = stage5_teacher_one(row, model, tokenizer)
            df.at[idx, "teacher_cot"] = str(result.get("teacher_cot", ""))
            df.at[idx, "teacher_thinking"] = str(result.get("teacher_thinking", ""))
        except Exception as e:  # noqa: BLE001
            logging.warning("stage5 row %d failed: %s", i, e)
            df.at[idx, "teacher_cot"] = ""
            df.at[idx, "teacher_thinking"] = ""
        df.at[idx, "_stage5_done"] = True
        processed += 1
        _maybe_autosave(df, save_path, save_every, processed, "5")
    return df


STAGE_RUNNERS = {
    "1": run_stage_1,
    "2": run_stage_2,
    "3": run_stage_3,
    "4": run_stage_4,
    "5": run_stage_5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Input parquet (one row per objection)")
    p.add_argument("--output", required=True, help="Output parquet (resume-safe)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--stages", default="1,2,3,4,5",
                   help="Comma-separated stages to run, e.g. '1,2,3,4,5'")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N rows (smoke testing)")
    p.add_argument("--cuda-visible", default=None,
                   help="Set CUDA_VISIBLE_DEVICES, e.g. '0,1,2,3'")
    p.add_argument("--auto-save-batch-size", type=int, default=50,
                   help="Save the parquet every N processed rows within each "
                        "stage (default: 50). Set to 0 to disable mid-stage saves "
                        "(stage-end checkpoints still happen).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
        logging.info("CUDA_VISIBLE_DEVICES=%s", args.cuda_visible)

    import pandas as pd  # imported here so CUDA_VISIBLE_DEVICES is set first

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGE_RUNNERS:
            raise SystemExit(f"unknown stage {s!r}; valid: {list(STAGE_RUNNERS)}")

    # Auto-resume: if the output parquet already exists, pick up where the
    # previous run left off. Each stage runner skips rows whose `_stage{N}_done`
    # flag is True, so resume works at row granularity, not just stage
    # granularity.
    if out_path.exists():
        df = pd.read_parquet(out_path)
        logging.info("Auto-resuming from %s (%d rows)", out_path, len(df))
    else:
        df = pd.read_parquet(in_path)
        logging.info("Loaded fresh from %s (%d rows)", in_path, len(df))
    if args.limit is not None:
        df = df.head(args.limit).copy()

    # Always materialize context columns up front — they're cheap and Stage 5 reads them.
    df["customer_context"] = [get_customer_context(r) for r in df.to_dict("records")]
    df["product_context"] = [get_product_context(r) for r in df.to_dict("records")]

    model, tokenizer = load_teacher_model(args.model_dir)
    logging.info("Teacher model loaded.")

    save_every = args.auto_save_batch_size if args.auto_save_batch_size > 0 else None
    t0 = time.perf_counter()
    for s in stages:
        logging.info("=== Stage %s ===", s)
        t_stage = time.perf_counter()
        df = STAGE_RUNNERS[s](df, model, tokenizer,
                              save_path=out_path, save_every=save_every)
        logging.info("Stage %s done in %.1fs", s, time.perf_counter() - t_stage)
        _save_checkpoint(df, out_path, stage_label=s)

    logging.info("All stages done in %.1fs. Final output at %s (%d rows)",
                 time.perf_counter() - t0, out_path, len(df))


if __name__ == "__main__":
    main()
