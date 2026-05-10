"""TSE SFT preprocessing pipeline (5 stages on a per-objection parquet).

Stages
------
  1. Substantive turn filter        keep_stage1, stage1_reason
  2. Agent response polish          polished_response, polish_operations,
                                    completeness, exclude_stage2, exclude_reason
  3. Current-call summary           current_call_summary
  4. Previous-calls summary         previous_calls_summary  (cached per customer)
  5. Teacher CoT + response         teacher_analysis, teacher_final, cot_grounded
                                    (Framing B: rationalize Stage 2's polished
                                    response; see 0510_stage5_design.md)
  + customer_context, product_context columns for downstream SFT.

Importable as a library (notebook uses it) AND runnable as a CLI batch script.

Batch usage:
    python 01_preprocessing.py \
        --input  TSE_by_objection.parquet \
        --output TSE_preprocessed.parquet \
        --stages 1,2,3,4,5 \
        --cuda-visible 0,1,2,3

Resume-safe: re-running with the same --output skips stages whose output
columns already exist in the on-disk parquet.

Filter cascade: stages 3, 4, 5 only run on rows that survive both
`keep_stage1=True` and `exclude_stage2=False`. Stage 2 only runs on rows with
`keep_stage1=True`. Ineligible rows get empty/None values in the new columns
so the schema stays consistent across the parquet.
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
# Stage 5: Teacher CoT + response (Framing B — rationalize Stage 2's polished
# response). Design rationale and full prompt rationale: 0510_stage5_design.md.
# ---------------------------------------------------------------------------

STAGE5_SYSTEM = """You are an expert outbound sales agent at American Express on the TSE
(Tele Strategic Expansion) channel.

""" + TSE_BACKGROUND + """

You will be given:
  - the customer's profile and product/campaign context (industry, business size,
    spend patterns, card portfolio, current campaign details)
  - a summary of prior calls with this customer, if any
  - a summary of how the current call has gone so far
  - the current objection summary
  - the full transcript up to the objection point
  - the response the agent actually gave

Your task is to reason about WHY this response is a good move given the situation.
A skilled agent's response is rarely arbitrary; it reflects a chain of judgment
about the customer, the objection, and what will move the conversation forward.
Your job is to reconstruct that judgment in the analysis channel, then echo the
agent's response in the final channel.

In the analysis channel, reason in flowing natural prose — not bullets, not
headers, not JSON, not enumerated steps. Think the way an experienced agent
thinks silently between hearing the objection and choosing what to say. Ground
your reasoning in the specific inputs in front of you: the customer's industry
and what typically matters in that industry, what you know from prior calls,
what has already happened in this call, and what the current objection actually
signals beneath its surface wording. Walk through what response options are
available, weigh them against each other, and arrive at the response the agent
gave.

Important: reason ONLY from the inputs you can see. Real agents often draw on
context that isn't in your inputs — relationship history not captured in the
prior-calls summary, pre-call research, account notes, the customer's tone of
voice. If the agent's response appears to draw on information beyond what's
visible to you, say so honestly in your reasoning rather than inventing context
to justify it. For example: "Given the visible context, this response makes
sense as a way to acknowledge the pricing concern before pivoting to value;
the specific framing about the customer's seasonal cash flow likely also reflects
relationship knowledge from prior interactions that isn't fully captured in the
summary." This honesty is more useful than fabricated justification.

Reason as much as the case warrants and no more. Simple cases warrant brief
reasoning; complex cases warrant more. Do not pad. Do not restate the inputs.
Do not enumerate steps mechanically. Do not second-guess yourself in circles;
once you've weighed the options and chosen, move on.

End your analysis with a single-line tag indicating how well the response is
grounded in the visible inputs:
  [grounded: high]   — response fully derivable from visible context
  [grounded: medium] — response mostly derivable; some elements suggest unstated context
  [grounded: low]    — response likely depends on context not visible in the inputs

In the final channel, output the agent's response exactly as given. The final
channel is not where you reason or rephrase — it's where you produce the target
response that your analysis just explained."""

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

THE AGENT'S RESPONSE:
{polished_agent_response}

In the analysis channel, reason about why this response is a good move given
the situation, working only from the visible inputs and being honest about any
points where the response appears to draw on information beyond what you can see.
End your analysis with a [grounded: high|medium|low] tag. In the final channel,
output the agent's response exactly as given.
"""


_GROUNDED_RE = re.compile(
    r"\[grounded:\s*(high|medium|low)\b[^\]]*\]\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _extract_grounded_tag(analysis: str) -> tuple[str, str]:
    """Strip the trailing `[grounded: high|medium|low ...]` tag from the analysis.

    Returns (analysis_without_tag, tag_value). tag_value is "" if not found.
    The student trains on the stripped version (per 0510_stage5_design.md, v1
    decision is to strip; the tag lives in its own column for diagnostics).
    """
    a = analysis.rstrip()
    m = _GROUNDED_RE.search(a)
    if not m:
        return a, ""
    cleaned = a[:m.start()].rstrip()
    return cleaned, m.group(1).lower()


def stage5_teacher_one(row: dict, model, tokenizer) -> dict:
    polished = row.get("polished_response", "")
    if not polished:
        # Stage 2 must have run and produced a non-empty polished response.
        # Eligibility mask should already exclude these, but guard anyway.
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
    analysis_clean, grounded = _extract_grounded_tag(out["analysis"])
    # Final channel should echo `polished`; log if it drifts so we can
    # spot-check during inspection.
    if out["final"].strip() != polished.strip():
        logging.debug("stage5 final-channel drift from polished_response "
                      "(len diff %d)", len(out["final"]) - len(polished))
    return {
        "teacher_analysis": analysis_clean,
        "teacher_final": out["final"],
        "cot_grounded": grounded,
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
    "5": ["teacher_analysis", "teacher_final", "cot_grounded"],
}


def _save_checkpoint(df, output_path: Path, stage: str) -> None:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, output_path)
    logging.info("Checkpoint saved after stage %s -> %s (%d rows)",
                 stage, output_path, len(df))


def _has_stage_output(df, stage: str) -> bool:
    return all(c in df.columns for c in STAGE_COLS[stage])


def _eligibility_mask(df):
    """True for rows still eligible after stage-1 filter and stage-2 exclusion.

    Used by stages 3/4/5 to skip rows that won't survive into the SFT set —
    no point spending teacher tokens on them. If `keep_stage1` /
    `exclude_stage2` columns aren't present yet (e.g. user ran a later stage
    standalone), defaults to True so we don't silently no-op.
    """
    import pandas as pd
    mask = pd.Series([True] * len(df), index=df.index)
    if "keep_stage1" in df.columns:
        mask = mask & df["keep_stage1"].fillna(False).astype(bool)
    if "exclude_stage2" in df.columns:
        mask = mask & ~df["exclude_stage2"].fillna(True).astype(bool)
    return mask


def _apply_per_row(df, fn, *, desc: str, only_where=None):
    """Apply fn(row_dict) to each row; returns list of result dicts (or None on error).

    `only_where` is an optional boolean Series; rows where False get None.
    """
    from tqdm import tqdm
    results = []
    n_err = 0
    iterator = tqdm(df.to_dict("records"), desc=desc, total=len(df))
    for i, row in enumerate(iterator):
        if only_where is not None and not bool(only_where.iloc[i]):
            results.append(None)
            continue
        try:
            results.append(fn(row))
        except Exception as e:  # noqa: BLE001 — isolated per row
            logging.warning("%s row %d failed: %s", desc, i, e)
            results.append({"_error": str(e)})
            n_err += 1
    if n_err:
        logging.warning("%s: %d rows errored", desc, n_err)
    return results


def run_stage_1(df, model, tokenizer):
    out = _apply_per_row(df, lambda r: stage1_filter_one(r, model, tokenizer),
                         desc="stage1-filter")
    df = df.copy()
    df["keep_stage1"] = [bool(r.get("keep_stage1")) if r and "_error" not in r else False
                        for r in out]
    df["stage1_reason"] = [r.get("stage1_reason", "") if r and "_error" not in r
                           else r.get("_error", "") if r else ""
                           for r in out]
    return df


def run_stage_2(df, model, tokenizer):
    out = _apply_per_row(df, lambda r: stage2_polish_one(r, model, tokenizer),
                         desc="stage2-polish", only_where=_eligibility_mask(df))
    df = df.copy()
    def field(r, k, default):
        if not r or "_error" in r:
            return default
        return r.get(k, default)
    df["polished_response"] = [field(r, "polished_response", "") for r in out]
    df["polish_operations"] = [field(r, "polish_operations", []) for r in out]
    df["completeness"] = [field(r, "completeness", "") for r in out]
    df["exclude_stage2"] = [bool(field(r, "exclude_stage2", True)) for r in out]
    df["exclude_reason"] = [field(r, "exclude_reason", None) for r in out]
    return df


def run_stage_3(df, model, tokenizer):
    out = _apply_per_row(df, lambda r: stage3_current_summary_one(r, model, tokenizer),
                         desc="stage3-current-summary",
                         only_where=_eligibility_mask(df))
    df = df.copy()
    df["current_call_summary"] = [s if isinstance(s, str) else "" for s in out]
    return df


def run_stage_4(df, model, tokenizer):
    df = df.copy()
    eligible = _eligibility_mask(df).tolist()
    cache: dict[str, str | None] = {}
    summaries: list[str | None] = []
    from tqdm import tqdm
    records = df.to_dict("records")
    for i, row in enumerate(tqdm(records, desc="stage4-prev-summary", total=len(df))):
        if not eligible[i]:
            summaries.append(None)
            continue
        key = _customer_cache_key(row)
        if key == "":
            summaries.append(None)
            continue
        if key in cache:
            summaries.append(cache[key])
            continue
        try:
            s = stage4_prev_summary_one(row, model, tokenizer)
        except Exception as e:  # noqa: BLE001
            logging.warning("stage4 failed for key %s: %s", key, e)
            s = None
        cache[key] = s
        summaries.append(s)
    df["previous_calls_summary"] = summaries
    return df


def run_stage_5(df, model, tokenizer):
    df = df.copy()
    # Ensure customer/product context cols exist (cheap to recompute).
    df["customer_context"] = [get_customer_context(r) for r in df.to_dict("records")]
    df["product_context"] = [get_product_context(r) for r in df.to_dict("records")]

    out = _apply_per_row(df, lambda r: stage5_teacher_one(r, model, tokenizer),
                         desc="stage5-teacher",
                         only_where=_eligibility_mask(df))
    df["teacher_analysis"] = [
        r.get("teacher_analysis", "") if r and "_error" not in r else ""
        for r in out
    ]
    df["teacher_final"] = [
        r.get("teacher_final", "") if r and "_error" not in r else ""
        for r in out
    ]
    df["cot_grounded"] = [
        r.get("cot_grounded", "") if r and "_error" not in r else ""
        for r in out
    ]
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
    p.add_argument("--resume", action="store_true",
                   help="Read existing --output and skip stages whose columns exist")
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

    if args.resume and out_path.exists():
        df = pd.read_parquet(out_path)
        logging.info("Resuming from %s (%d rows)", out_path, len(df))
    else:
        df = pd.read_parquet(in_path)
        logging.info("Loaded %s (%d rows)", in_path, len(df))
    if args.limit is not None:
        df = df.head(args.limit).copy()

    # Always materialize context columns up front — they're cheap and Stage 5 reads them.
    df["customer_context"] = [get_customer_context(r) for r in df.to_dict("records")]
    df["product_context"] = [get_product_context(r) for r in df.to_dict("records")]

    model, tokenizer = load_teacher_model(args.model_dir)
    logging.info("Teacher model loaded.")

    t0 = time.perf_counter()
    for s in stages:
        if args.resume and _has_stage_output(df, s):
            logging.info("Stage %s already present — skipping", s)
            continue
        logging.info("=== Stage %s ===", s)
        t_stage = time.perf_counter()
        df = STAGE_RUNNERS[s](df, model, tokenizer)
        logging.info("Stage %s done in %.1fs", s, time.perf_counter() - t_stage)
        _save_checkpoint(df, out_path, stage=s)

    df.to_parquet(out_path, index=False)
    logging.info("All stages done in %.1fs. Wrote %s (%d rows)",
                 time.perf_counter() - t0, out_path, len(df))


if __name__ == "__main__":
    main()
