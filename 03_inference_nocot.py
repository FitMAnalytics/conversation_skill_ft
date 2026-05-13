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

Prefix-cache speedup (default ON):
  Because `SYSTEM_CONTENT` and `reasoning_effort` are fixed across the whole
  run, every prompt shares a long token prefix (the rendered system block).
  We probe the first batch for the longest common token prefix (LCP), run
  one forward pass on it to populate a `DynamicCache`, then for each batch
  expand that batch=1 cache to batch=B and call generate with only the
  per-row suffix tokens. This cuts the prefill cost from O(B * |prefix|)
  to O(|prefix|) per run. The cache is held in memory throughout the run.

  Override with `--no-prefix-cache` to fall back to the plain path (useful
  if a transformers version mishandles `past_key_values` with `generate`).

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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import DynamicCache
except ImportError:
    from transformers.cache_utils import DynamicCache

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

# Skip prefix caching if the detected shared prefix is shorter than this.
# Below this length the bookkeeping cost outweighs the saved prefill.
MIN_PREFIX_CACHE_TOKENS = 64


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


def texts_for_rows(tokenizer, rows: list[dict], reasoning_effort: str,
                   max_transcript_chars: int) -> list[str]:
    """Render each row's prompt as a chat-template string."""
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
    return texts


# ============================================================================
# PREFIX CACHE — share KV across rows with identical system block
# ============================================================================

def find_lcp_tokens(seqs: list[list[int]]) -> int:
    """Length of the longest common token prefix across all sequences."""
    if not seqs:
        return 0
    min_len = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(min_len):
        for s in seqs[1:]:
            if s[i] != first[i]:
                return i
    return min_len


def precompute_prefix_cache(model, prefix_ids: list[int]) -> DynamicCache:
    """One forward pass over the shared prefix → populated batch=1 DynamicCache."""
    cache = DynamicCache()
    ids = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask,
              past_key_values=cache, use_cache=True)
    return cache


def expand_prefix_cache(cache: DynamicCache, batch_size: int) -> DynamicCache:
    """Replicate a batch=1 cache to batch=B with contiguous copies.

    `.contiguous()` after `.expand()` materializes the broadcast so the
    subsequent `generate` mutates a fresh tensor instead of aliasing the
    original cache.
    """
    new = DynamicCache()
    for k, v in zip(cache.key_cache, cache.value_cache):
        new.key_cache.append(k.expand(batch_size, -1, -1, -1).contiguous())
        new.value_cache.append(v.expand(batch_size, -1, -1, -1).contiguous())
    return new


@dataclass
class PrefixCacheState:
    """Holds the shared-prefix KV caches for adapter and (optionally) base."""
    prefix_ids: list[int] = field(default_factory=list)
    adapter_cache: DynamicCache | None = None
    base_cache: DynamicCache | None = None
    enabled: bool = False


def _log_lcp_debug(tokenizer, sample_texts: list[str],
                   token_lists: list[list[int]], lcp: int) -> None:
    """Print two sample prompts side-by-side and locate where they diverge.

    Helps explain a smaller-than-expected LCP — usually a per-row field
    leaking into the prefix (e.g. customer_context) or a template-injected
    dynamic value (e.g. current date/time).
    """
    if len(token_lists) < 2:
        logging.info("LCP debug: probe batch has only %d rows — nothing to diff.",
                     len(token_lists))
        return

    s0, s1 = sample_texts[0], sample_texts[1]
    char_div = next(
        (i for i in range(min(len(s0), len(s1))) if s0[i] != s1[i]),
        min(len(s0), len(s1)),
    )

    logging.info("=" * 80)
    logging.info("LCP DEBUG  token-LCP=%d  (row0 has %d toks, row1 has %d toks)",
                 lcp, len(token_lists[0]), len(token_lists[1]))
    logging.info("           char divergence at index %d", char_div)
    logging.info("-" * 80)
    logging.info("ROW 0 — rendered prompt (first 800 chars):")
    logging.info("%s", s0[:800])
    logging.info("-" * 80)
    logging.info("ROW 1 — rendered prompt (first 800 chars):")
    logging.info("%s", s1[:800])
    logging.info("-" * 80)

    pre = max(0, char_div - 80)
    post = char_div + 200
    logging.info("DIVERGE near char %d:", char_div)
    logging.info("  ROW 0 [%d:%d] = %r", pre, post, s0[pre:post])
    logging.info("  ROW 1 [%d:%d] = %r", pre, post, s1[pre:post])

    a, b = token_lists[0], token_lists[1]
    lo = max(0, lcp - 5)
    hi_a, hi_b = min(len(a), lcp + 10), min(len(b), lcp + 10)
    logging.info("Tokens around divergence (5 before, up to 10 after):")
    logging.info("  ROW 0 tokens[%d:%d] = %s -> %r",
                 lo, hi_a, a[lo:hi_a],
                 tokenizer.decode(a[lo:hi_a], skip_special_tokens=False))
    logging.info("  ROW 1 tokens[%d:%d] = %s -> %r",
                 lo, hi_b, b[lo:hi_b],
                 tokenizer.decode(b[lo:hi_b], skip_special_tokens=False))
    logging.info("=" * 80)


def _single_sample_lcp(tokenizer, sample_text: str, sample_ids: list[int]
                       ) -> tuple[int, str, list[int]]:
    """Probe-with-1-row helper: compare against a system-only chat-template render.

    Returns (lcp_token_len, sys_text, sys_ids). On failure returns (0, "", []).
    """
    try:
        sys_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_CONTENT}],
            add_generation_prompt=False, tokenize=False,
        )
        sys_ids = tokenizer.encode(sys_text, add_special_tokens=False)
        lcp = 0
        for i in range(min(len(sys_ids), len(sample_ids))):
            if sys_ids[i] != sample_ids[i]:
                break
            lcp = i + 1
        return lcp, sys_text, sys_ids
    except Exception as e:  # noqa: BLE001
        logging.warning("Single-sample prefix probe failed: %s", e)
        return 0, "", []


def run_lcp_debug(tokenizer, sample_texts: list[str]) -> None:
    """Always-on debug printer: dumps two probe prompts and where they diverge.

    Works for single-sample probes too (diffs against the system-only render).
    """
    logging.info("LCP DEBUG: probe batch has %d sample(s).", len(sample_texts))
    if not sample_texts:
        return
    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in sample_texts]
    if len(token_lists) >= 2:
        lcp = find_lcp_tokens(token_lists)
        _log_lcp_debug(tokenizer, sample_texts, token_lists, lcp)
    else:
        lcp, sys_text, sys_ids = _single_sample_lcp(
            tokenizer, sample_texts[0], token_lists[0]
        )
        if not sys_ids:
            return
        logging.info("LCP DEBUG: only 1 sample — diffing against system-only render.")
        _log_lcp_debug(
            tokenizer,
            [sample_texts[0], sys_text],
            [token_lists[0], sys_ids],
            lcp,
        )


def setup_prefix_cache(model, tokenizer, sample_texts: list[str],
                       include_base: bool,
                       min_tokens: int = MIN_PREFIX_CACHE_TOKENS) -> PrefixCacheState:
    """Detect shared prefix from rendered prompts and precompute KV caches."""
    state = PrefixCacheState()
    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in sample_texts]

    if len(token_lists) >= 2:
        lcp = find_lcp_tokens(token_lists)
    elif len(token_lists) == 1:
        lcp, _, _ = _single_sample_lcp(tokenizer, sample_texts[0], token_lists[0])
    else:
        return state

    if lcp < min_tokens:
        logging.info(
            "Shared-prefix LCP=%d tokens (< %d threshold) — prefix cache disabled.",
            lcp, min_tokens,
        )
        return state

    state.prefix_ids = token_lists[0][:lcp]
    logging.info("Precomputing prefix-cache KV for %d shared tokens...", lcp)
    t0 = time.perf_counter()
    state.adapter_cache = precompute_prefix_cache(model, state.prefix_ids)
    logging.info("  adapter prefix-cache built in %.2fs", time.perf_counter() - t0)
    if include_base:
        t0 = time.perf_counter()
        with model.disable_adapter():
            state.base_cache = precompute_prefix_cache(model, state.prefix_ids)
        logging.info("  base prefix-cache built in %.2fs", time.perf_counter() - t0)
    state.enabled = True
    return state


def generate_with_prefix_cache(model, tokenizer, prompt_texts: list[str],
                               state: PrefixCacheState,
                               max_new_tokens: int,
                               use_adapter: bool = True) -> list[str] | None:
    """Greedy batched generate using the precomputed shared-prefix cache.

    Returns one decoded suffix string per row, or None when any row's
    tokenization does not begin with the cached prefix (caller falls back).
    """
    if not state.enabled:
        return None
    source = state.adapter_cache if use_adapter else state.base_cache
    if source is None:
        return None

    prefix_ids = state.prefix_ids
    prefix_len = len(prefix_ids)
    B = len(prompt_texts)

    full_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in prompt_texts]
    for i, ids in enumerate(full_ids_list):
        if ids[:prefix_len] != prefix_ids:
            logging.warning(
                "Row %d does not start with cached prefix — disabling cache for this batch.",
                i,
            )
            return None

    suffixes = [ids[prefix_len:] for ids in full_ids_list]
    max_suffix = max(len(s) for s in suffixes)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    padded, suffix_attn = [], []
    for s in suffixes:
        pad_n = max_suffix - len(s)
        padded.append([pad_id] * pad_n + s)
        suffix_attn.append([0] * pad_n + [1] * len(s))

    suffix_ids = torch.tensor(padded, device=model.device, dtype=torch.long)
    suffix_attn_t = torch.tensor(suffix_attn, device=model.device, dtype=torch.long)
    # Full attention mask covers cached prefix (all 1s) + per-row suffix mask.
    prefix_attn = torch.ones((B, prefix_len), device=model.device, dtype=torch.long)
    full_attn = torch.cat([prefix_attn, suffix_attn_t], dim=1)

    batched_cache = expand_prefix_cache(source, B)

    with torch.no_grad():
        out = model.generate(
            input_ids=suffix_ids,
            attention_mask=full_attn,
            past_key_values=batched_cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    # `out` returned by generate when past_key_values is provided contains
    # [suffix_ids passed in | newly generated tokens] — NOT the cached prefix.
    new_tokens = out[:, suffix_ids.shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=False)


def generate_batch_decoded(model, tokenizer, prompt_texts: list[str],
                           state: PrefixCacheState,
                           max_new_tokens: int,
                           use_adapter: bool = True) -> list[str]:
    """Try cached path; fall back to non-cached encoding+generate on mismatch."""
    cached = generate_with_prefix_cache(
        model, tokenizer, prompt_texts, state, max_new_tokens, use_adapter=use_adapter
    )
    if cached is not None:
        return cached
    enc = encode_batch(tokenizer, prompt_texts)
    return generate_from_inputs(model, tokenizer, enc, max_new_tokens)


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
    p.add_argument("--save-every", type=int, default=20,
                   help="Persist partial parquet every N batches (resume-safe).")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="Disable the shared-prefix DynamicCache optimization. "
                        "Slower; use if your transformers version mishandles "
                        "`past_key_values` with `generate`.")
    p.add_argument("--debug-lcp", action="store_true",
                   help="Print two probe-batch prompts and the exact char/token "
                        "where they diverge, to diagnose a smaller-than-expected LCP.")
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

    # Build probe texts up-front so debug + cache setup can both use them.
    probe_idx = pending[: args.batch_size]
    probe_rows = [df.loc[i].to_dict() for i in probe_idx]
    probe_texts = texts_for_rows(
        tokenizer, probe_rows,
        reasoning_effort=args.reasoning_effort,
        max_transcript_chars=args.max_transcript_chars,
    )

    if args.debug_lcp:
        run_lcp_debug(tokenizer, probe_texts)

    cache_state = PrefixCacheState()
    if not args.no_prefix_cache:
        cache_state = setup_prefix_cache(
            model, tokenizer, probe_texts,
            include_base=args.include_base,
        )

    t0 = time.perf_counter()
    n_batches = 0
    for start in range(0, len(pending), args.batch_size):
        batch_idx = pending[start: start + args.batch_size]
        rows = [df.loc[i].to_dict() for i in batch_idx]
        prompt_texts = texts_for_rows(
            tokenizer, rows,
            reasoning_effort=args.reasoning_effort,
            max_transcript_chars=args.max_transcript_chars,
        )

        decoded = generate_batch_decoded(
            model, tokenizer, prompt_texts, cache_state,
            max_new_tokens=args.max_new_tokens, use_adapter=True,
        )
        for i, raw in zip(batch_idx, decoded):
            parsed = parse_channels(raw)
            df.at[i, "raw_response"] = raw
            df.at[i, "analysis"] = parsed["analysis"]
            df.at[i, "final"] = parsed["final"]

        if args.include_base:
            with model.disable_adapter():
                base_decoded = generate_batch_decoded(
                    model, tokenizer, prompt_texts, cache_state,
                    max_new_tokens=args.max_new_tokens, use_adapter=False,
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
            logging.info("Checkpointed → %s (every %d batches)",
                         out_path, args.save_every)

    df.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    logging.info("Finished. %d rows in %.1fs (%.2f rows/s). Wrote %s",
                 len(pending), elapsed, len(pending) / elapsed, out_path)


if __name__ == "__main__":
    main()
