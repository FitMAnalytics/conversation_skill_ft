"""Batch inference for the no-CoT LoRA SFT adapter — DataFrame in, parquet out.

Self-contained: defines all generation, channel-parsing, and model-loading
helpers internally. The ONLY external project import is
`02_sft_training_nocot_0511.py` for the prompt templates
(`SYSTEM_CONTENT`, `USER_TMPL`, `trim_transcript`,
`DEFAULT_MAX_TRANSCRIPT_CHARS`, `get_customer_context`) — re-exported below
so both this script and `03_inference_nocot.ipynb` share one source of truth
and stay in lockstep with training.

Pipeline per row:
  1. Build messages = [system=SYSTEM_CONTENT, user=USER_TMPL.format(4 inputs)]
     using the same templates the training script uses.
  2. Render via `apply_chat_template(..., tokenize=False)` with the requested
     `reasoning_effort`. We tokenize separately to get a clean BatchEncoding
     (newer transformers versions return BatchEncoding for some chat templates
     and a plain tensor for others when `return_tensors="pt"` is set inside
     apply_chat_template — splitting the call avoids that ambiguity).
  3. Batched greedy generate (`do_sample=False`) with explicit
     `input_ids` + `attention_mask` (required for correct left-padding).
  4. Decode each row with `skip_special_tokens=False`; parse `analysis` and
     `final` channels via `parse_channels`.
  5. (Optional) repeat under `model.disable_adapter()` for an A/B against base.

Input DataFrame columns required:
    transcript, previous_call_summary, current_call_summary, objection_summary
Optional:
    customer_context  (else `get_customer_context(row)` fallback / "")

Output: input DataFrame + columns
    raw_response, analysis, final
    [optionally: base_raw_response, base_analysis, base_final]

Inference runs single-process with `device_map="auto"` (tensor sharding across
local GPUs). Do NOT launch under `accelerate` / `deepspeed`.

Example:
    python 03_inference_nocot.py \\
        --input data/eval.parquet \\
        --output data/eval_predictions.parquet \\
        --model-dir /path/to/gpt-oss-120b \\
        --adapter-dir checkpoints_nocot/final_adapter \\
        --batch-size 4 \\
        --max-new-tokens 1024 \\
        --reasoning-effort medium

To evaluate an intermediate checkpoint, point `--adapter-dir` at it:
    --adapter-dir checkpoints_nocot/checkpoint-90
"""

import argparse
import importlib.util
import logging
import re
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(_HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Only external project dependency: prompt templates from the training script.
_train = _load_module("train02_nocot", "02_sft_training_nocot_0511.py")
SYSTEM_CONTENT = _train.SYSTEM_CONTENT
USER_TMPL = _train.USER_TMPL
trim_transcript = _train.trim_transcript
get_customer_context = _train.get_customer_context
DEFAULT_MAX_TRANSCRIPT_CHARS = _train.DEFAULT_MAX_TRANSCRIPT_CHARS

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_ADAPTER_DIR = "checkpoints_nocot/final_adapter"

REQUIRED_COLS = (
    "transcript",
    "previous_call_summary",
    "current_call_summary",
    "objection_summary",
)


# ============================================================================
# CHANNEL PARSING
# ============================================================================

_CHANNEL_RE = re.compile(
    r"<\|channel\|>(?P<channel>[^<]+?)<\|message\|>(?P<content>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)


def parse_channels(raw_text: str) -> dict:
    """Return {'analysis': str, 'final': str} from a raw Harmony decode."""
    out = {"analysis": "", "final": ""}
    for m in _CHANNEL_RE.finditer(raw_text):
        ch = m.group("channel").strip()
        if ch in out:
            out[ch] += m.group("content")
    return out


# ============================================================================
# TOKENIZATION + GENERATION
# ============================================================================

def render_prompt_text(tokenizer, messages: list[dict],
                       reasoning_effort: str) -> str:
    """Apply the chat template with `reasoning_effort` and `add_generation_prompt`.

    Returns a plain string. Tokenization happens separately so callers always
    get a clean BatchEncoding (with `input_ids` AND `attention_mask`) instead
    of the BatchEncoding-vs-tensor surprise that some transformers versions
    produce when `return_tensors="pt"` is passed inside `apply_chat_template`.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


def encode_one(tokenizer, text: str):
    """Tokenize a single prompt string into a BatchEncoding on CPU."""
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def encode_batch(tokenizer, texts: list[str]):
    """Tokenize a list of prompt strings with left padding (decoder-only generate)."""
    return tokenizer(
        texts, return_tensors="pt", add_special_tokens=False, padding=True,
    )


def generate_from_inputs(model, tokenizer, inputs, max_new_tokens: int) -> list[str]:
    """Greedy batched generate. Returns one decoded suffix string per row.

    `inputs` is a BatchEncoding (or dict) with `input_ids` + `attention_mask`.
    Works for batch size 1 too; always returns a list.
    """
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    prompt_len = input_ids.shape[1]
    suffix_ids = out[:, prompt_len:]
    return tokenizer.batch_decode(suffix_ids, skip_special_tokens=False)


# ============================================================================
# MODEL LOAD
# ============================================================================

def load_model_with_adapter(model_dir: str, adapter_dir: str):
    """Single-process load: base model with `device_map='auto'` + PEFT adapter.

    `adapter_dir` may point at either `.../final_adapter` or any intermediate
    `.../checkpoint-N`; PEFT only needs `adapter_config.json` + the adapter
    weights file.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batched generation
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


# ============================================================================
# PROMPT CONSTRUCTION (mirrors training-script exactly)
# ============================================================================

def build_messages(transcript: str, previous_call_summary: str,
                   current_call_summary: str, objection_summary: str,
                   customer_context: str,
                   max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS) -> list[dict]:
    """Build the 2-message system+user list, trimming transcript like training does."""
    transcript_trimmed, _ = trim_transcript(transcript, max_transcript_chars)
    user_text = USER_TMPL.format(
        customer_context=customer_context,
        previous_call_summary=previous_call_summary,
        current_call_summary=current_call_summary,
        objection_summary=objection_summary,
        transcript=transcript_trimmed,
    )
    return [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_text},
    ]


def generate_response(model, tokenizer, transcript: str,
                      previous_call_summary: str,
                      current_call_summary: str,
                      objection_summary: str,
                      customer_context: str = "",
                      reasoning_effort: str = "medium",
                      max_new_tokens: int = 1024,
                      max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
                      include_base: bool = False) -> dict:
    """Single-row helper.

    Returns {raw, analysis, final}, plus {base_raw, base_analysis, base_final}
    when `include_base=True`.
    """
    messages = build_messages(
        transcript=transcript,
        previous_call_summary=previous_call_summary,
        current_call_summary=current_call_summary,
        objection_summary=objection_summary,
        customer_context=customer_context,
        max_transcript_chars=max_transcript_chars,
    )
    text = render_prompt_text(tokenizer, messages, reasoning_effort)
    inputs = encode_one(tokenizer, text)
    raw = generate_from_inputs(model, tokenizer, inputs, max_new_tokens)[0]
    parsed = parse_channels(raw)
    result = {"raw": raw, "analysis": parsed["analysis"], "final": parsed["final"]}
    if include_base:
        with model.disable_adapter():
            base_raw = generate_from_inputs(model, tokenizer, inputs, max_new_tokens)[0]
        bp = parse_channels(base_raw)
        result["base_raw"] = base_raw
        result["base_analysis"] = bp["analysis"]
        result["base_final"] = bp["final"]
    return result


# ============================================================================
# BATCH PIPELINE
# ============================================================================

def _row_customer_context(row: dict) -> str:
    """Prefer explicit `customer_context` column; fall back to training-script fetcher."""
    val = row.get("customer_context")
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
        return get_customer_context(row) or ""
    return str(val)


def render_batch(tokenizer, rows: list[dict], reasoning_effort: str,
                 max_transcript_chars: int):
    """Render a batch of rows as a left-padded BatchEncoding."""
    texts = []
    for r in rows:
        messages = build_messages(
            transcript=r["transcript"],
            previous_call_summary=r["previous_call_summary"],
            current_call_summary=r["current_call_summary"],
            objection_summary=r["objection_summary"],
            customer_context=_row_customer_context(r),
            max_transcript_chars=max_transcript_chars,
        )
        texts.append(render_prompt_text(tokenizer, messages, reasoning_effort))
    return encode_batch(tokenizer, texts)


# ============================================================================
# I/O + RESUME
# ============================================================================

def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input {path} is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLS)}"
        )
    return df


def init_output_columns(df: pd.DataFrame, include_base: bool) -> pd.DataFrame:
    for c in ("raw_response", "analysis", "final"):
        if c not in df.columns:
            df[c] = pd.NA
    if include_base:
        for c in ("base_raw_response", "base_analysis", "base_final"):
            if c not in df.columns:
                df[c] = pd.NA
    return df


def load_existing_output(out_path: Path, df_in: pd.DataFrame,
                         include_base: bool) -> pd.DataFrame:
    """Resume: if output parquet exists and aligns row-for-row, reuse it."""
    if not out_path.exists():
        return init_output_columns(df_in.copy(), include_base)
    existing = pd.read_parquet(out_path)
    if len(existing) != len(df_in):
        logging.warning(
            "Existing output has %d rows but input has %d — ignoring existing.",
            len(existing), len(df_in),
        )
        return init_output_columns(df_in.copy(), include_base)
    return init_output_columns(existing, include_base)


def pending_indices(df: pd.DataFrame) -> list[int]:
    """Row indices where `final` is still NA — not yet generated."""
    mask = df["final"].isna()
    return df.index[mask].tolist()


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Parquet (or .csv) of input rows.")
    p.add_argument("--output", required=True, help="Parquet destination.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR,
                   help="LoRA adapter dir — final_adapter/ OR any checkpoint-N/.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-transcript-chars", type=int,
                   default=DEFAULT_MAX_TRANSCRIPT_CHARS,
                   help="Char-cap from end of transcript — same default as training.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke cap; processes only first N pending rows.")
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with `model.disable_adapter()` for A/B.")
    p.add_argument("--save-every", type=int, default=10,
                   help="Persist partial parquet every N batches.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_in = read_input(in_path)
    logging.info("Loaded %d rows from %s", len(df_in), in_path)

    df = load_existing_output(out_path, df_in, args.include_base)
    pending = pending_indices(df)
    logging.info("%d rows pending (resume-skipped %d).",
                 len(pending), len(df) - len(pending))

    if args.max_samples is not None:
        pending = pending[: args.max_samples]
        logging.info("Smoke cap: processing only first %d pending rows.", len(pending))

    if not pending:
        logging.info("Nothing to do. Output already complete: %s", out_path)
        return

    model, tokenizer = load_model_with_adapter(args.model_dir, args.adapter_dir)
    logging.info("Model + adapter loaded from %s", args.adapter_dir)

    t0 = time.perf_counter()
    n_batches = 0
    for start in range(0, len(pending), args.batch_size):
        batch_idx = pending[start: start + args.batch_size]
        rows = [df.loc[i].to_dict() for i in batch_idx]

        enc = render_batch(
            tokenizer, rows,
            reasoning_effort=args.reasoning_effort,
            max_transcript_chars=args.max_transcript_chars,
        )
        decoded = generate_from_inputs(model, tokenizer, enc, args.max_new_tokens)

        for i, raw in zip(batch_idx, decoded):
            parsed = parse_channels(raw)
            df.at[i, "raw_response"] = raw
            df.at[i, "analysis"] = parsed["analysis"]
            df.at[i, "final"] = parsed["final"]

        if args.include_base:
            with model.disable_adapter():
                base_decoded = generate_from_inputs(
                    model, tokenizer, enc, args.max_new_tokens
                )
            for i, raw in zip(batch_idx, base_decoded):
                parsed = parse_channels(raw)
                df.at[i, "base_raw_response"] = raw
                df.at[i, "base_analysis"] = parsed["analysis"]
                df.at[i, "base_final"] = parsed["final"]

        n_batches += 1
        done = start + len(batch_idx)
        rate = done / (time.perf_counter() - t0)
        logging.info("Batch %d done | %d/%d rows | %.2f rows/s",
                     n_batches, done, len(pending), rate)

        if n_batches % args.save_every == 0:
            df.to_parquet(out_path, index=False)
            logging.info("Checkpointed → %s", out_path)

    df.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    logging.info("Finished. %d rows in %.1fs (%.2f rows/s). Wrote %s",
                 len(pending), elapsed, len(pending) / elapsed, out_path)


if __name__ == "__main__":
    main()
