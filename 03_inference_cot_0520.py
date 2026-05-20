"""Batch inference for the CoT LoRA SFT adapter — DataFrame in, parquet out.

Mirrors `03_inference_nocot.py` for the CoT training pipeline. Loads the SAME
YAML the model was trained with (`02_train_config_cot_0520.yaml`) so the
rendered prompt is byte-for-byte identical to what training saw — no drift
between train and inference.

The ONLY external project import is `02_sft_training_cot_0520.py` for the
shared prompt-rendering helpers (`TrainConfig`, `trim_transcript`,
`render_user_prompt`, `DEFAULT_MAX_TRANSCRIPT_CHARS`).

Pipeline per row:
  1. Build messages = [system=cfg.system_prompt, user=render_user_prompt(cfg, row, trimmed_transcript)]
  2. Render via `apply_chat_template(..., tokenize=False)` with the requested
     `reasoning_effort`; tokenize separately to get a clean BatchEncoding.
  3. Batched greedy generate (`do_sample=False`) with explicit `input_ids` +
     `attention_mask` (required for correct left-padding).
  4. Decode each row with `skip_special_tokens=False`; parse `analysis` and
     `final` channels via `parse_channels`.
  5. (Optional) repeat under `model.disable_adapter()` for an A/B against base.

Prefix-cache speedup (default ON):
  The rendered system + developer block is identical across rows in a run
  (same `cfg.system_prompt`, same `reasoning_effort`), so prompts share a long
  token prefix. We probe the first batch for the longest common token prefix
  (LCP), run one forward pass on it to populate a `DynamicCache`, then for each
  batch expand that batch=1 cache to batch=B and call generate with only the
  per-row suffix tokens. Cuts prefill from O(B * |prefix|) to O(|prefix|).
  Override with `--no-prefix-cache` to fall back to the plain path.

Required input DataFrame columns (config-driven; defaults shown):
    transcript, current_call_summary, customer_context
    previous_calls_summary    — only when `include_previous_call_summary: true`

Output: input DataFrame + columns
    raw_response, analysis, final
    [optionally: base_raw_response, base_analysis, base_final]

Inference runs single-process with `device_map="auto"` (tensor sharding across
local GPUs). Do NOT launch under `accelerate` / `deepspeed`.

Example:
    python 03_inference_cot_0520.py \\
        --config 02_train_config_cot_0520.yaml \\
        --input data/eval.parquet \\
        --output data/eval_predictions_cot.parquet \\
        --model-dir /path/to/gpt-oss-20b \\
        --adapter-dir checkpoints_cot_0520/final_adapter \\
        --batch-size 4 \\
        --max-new-tokens 1024 \\
        --reasoning-effort medium

Evaluate an intermediate checkpoint by pointing `--adapter-dir` at it:
    --adapter-dir checkpoints_cot_0520/checkpoint-90
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


# Only external project dependency: shared prompt helpers from the training script.
_train = _load_module("train02_cot", "02_sft_training_cot_0520.py")
TrainConfig = _train.TrainConfig
trim_transcript = _train.trim_transcript
render_user_prompt = _train.render_user_prompt
DEFAULT_MAX_TRANSCRIPT_CHARS = _train.DEFAULT_MAX_TRANSCRIPT_CHARS

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-20b"
DEFAULT_ADAPTER_DIR = "checkpoints_cot_0520/final_adapter"

# Skip prefix caching if the detected shared prefix is shorter than this.
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
    """Apply the chat template with `reasoning_effort` and `add_generation_prompt`."""
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
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def encode_batch(tokenizer, texts: list[str]):
    return tokenizer(
        texts, return_tensors="pt", add_special_tokens=False, padding=True,
    )


def generate_from_inputs(model, tokenizer, inputs, max_new_tokens: int) -> list[str]:
    """Greedy batched generate. Returns one decoded suffix string per row."""
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
    """Single-process load: base model with `device_map='auto'` + PEFT adapter."""
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
# PROMPT CONSTRUCTION (mirrors training-script render path exactly)
# ============================================================================

def _row_customer_context(row: dict, cfg: TrainConfig) -> str:
    """Read the customer_context column. Errors loudly if missing — training
    expects this column to be pre-rendered by 01_preprocessing.py."""
    val = row.get(cfg.customer_context_field)
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
        raise ValueError(
            f"Row is missing required column '{cfg.customer_context_field}' "
            f"(empty/null). Inference expects this column to be pre-rendered "
            f"upstream; train and inference must see identical prompts."
        )
    return str(val)


def build_messages(row: dict, cfg: TrainConfig) -> list[dict]:
    """Build the 2-message system+user list, trimming transcript like training does."""
    transcript_trimmed, _ = trim_transcript(
        row[cfg.transcript_field], cfg.max_transcript_chars
    )
    # render_user_prompt expects a row-like dict; supply customer_context via the
    # configured field so the existing helper finds it.
    row_for_render = dict(row)
    row_for_render[cfg.customer_context_field] = _row_customer_context(row, cfg)
    user_text = render_user_prompt(cfg, row_for_render, transcript_trimmed)
    return [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_text},
    ]


def texts_for_rows(tokenizer, rows: list[dict], cfg: TrainConfig,
                   reasoning_effort: str) -> list[str]:
    """Render each row's prompt as a chat-template string."""
    return [
        render_prompt_text(tokenizer, build_messages(r, cfg), reasoning_effort)
        for r in rows
    ]


# ============================================================================
# PREFIX CACHE — share KV across rows with identical system+developer block
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


def _cache_num_layers(cache) -> int:
    """Number of populated layers in a DynamicCache, across transformers versions."""
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    n = 0
    while True:
        try:
            _ = cache[n]
        except (IndexError, KeyError, AttributeError):
            return n
        n += 1


def _cache_get_layer_kv(cache, layer_idx: int):
    """Return (keys, values) tensors for one cache layer, across versions."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
        if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
            return layer.key_cache, layer.value_cache
    return cache[layer_idx]


def expand_prefix_cache(cache: DynamicCache, batch_size: int) -> DynamicCache:
    """Replicate a batch=1 cache to batch=B with contiguous copies."""
    new = DynamicCache()
    for layer_idx in range(_cache_num_layers(cache)):
        k, v = _cache_get_layer_kv(cache, layer_idx)
        new.update(
            k.expand(batch_size, -1, -1, -1).contiguous(),
            v.expand(batch_size, -1, -1, -1).contiguous(),
            layer_idx,
        )
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
    """Print two sample prompts side-by-side and locate where they diverge."""
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


def _single_sample_lcp(tokenizer, sample_ids: list[int],
                       cfg: TrainConfig,
                       reasoning_effort: str) -> tuple[int, str, list[int]]:
    """Probe-with-1-row helper: compare against a system-only chat-template render.

    Critical: pass the SAME `reasoning_effort` the row used, otherwise the
    template's default (medium) leaks into the probe and the LCP cuts off
    inside the system block at the `Reasoning: <effort>` line.
    """
    try:
        try:
            sys_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": cfg.system_prompt}],
                reasoning_effort=reasoning_effort,
                add_generation_prompt=False, tokenize=False,
            )
        except TypeError:
            sys_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": cfg.system_prompt}],
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


def run_lcp_debug(tokenizer, sample_texts: list[str], cfg: TrainConfig,
                  reasoning_effort: str) -> None:
    """Always-on debug printer: dumps two probe prompts and where they diverge."""
    logging.info("LCP DEBUG: probe batch has %d sample(s).", len(sample_texts))
    if not sample_texts:
        return
    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in sample_texts]
    if len(token_lists) >= 2:
        lcp = find_lcp_tokens(token_lists)
        _log_lcp_debug(tokenizer, sample_texts, token_lists, lcp)
    else:
        lcp, sys_text, sys_ids = _single_sample_lcp(
            tokenizer, token_lists[0], cfg, reasoning_effort
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
                       cfg: TrainConfig,
                       reasoning_effort: str,
                       include_base: bool,
                       min_tokens: int = MIN_PREFIX_CACHE_TOKENS) -> PrefixCacheState:
    """Detect shared prefix from rendered prompts and precompute KV caches."""
    state = PrefixCacheState()
    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in sample_texts]

    if len(token_lists) >= 2:
        lcp = find_lcp_tokens(token_lists)
    elif len(token_lists) == 1:
        lcp, _, _ = _single_sample_lcp(tokenizer, token_lists[0], cfg, reasoning_effort)
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
    """Greedy batched generate using the precomputed shared-prefix cache."""
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

def required_columns(cfg: TrainConfig) -> list[str]:
    """Input columns the inference script needs, given the YAML toggles."""
    required = [
        cfg.transcript_field,
        cfg.current_call_summary_field,
        cfg.customer_context_field,
    ]
    if cfg.include_previous_call_summary:
        required.append(cfg.previous_call_summary_field)
    return required


def read_input(path: Path, cfg: TrainConfig) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    required = required_columns(cfg)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input {path} is missing required columns: {missing}. "
            f"Required (given the YAML config): {required}"
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
    p.add_argument("--config", required=True,
                   help="Path to the training YAML (e.g. 02_train_config_cot_0520.yaml). "
                        "Same file used by 02_sft_training_cot_0520.py — drives the "
                        "prompt rendering and the input-column contract.")
    p.add_argument("--input", required=True, help="Parquet (or .csv) of input rows.")
    p.add_argument("--output", required=True, help="Parquet destination.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR,
                   help="LoRA adapter dir — final_adapter/ OR any checkpoint-N/.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-transcript-chars", type=int, default=None,
                   help="Override the YAML's max_transcript_chars. Default: take "
                        "from the YAML so train and inference are in lockstep.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke cap; processes only first N pending rows.")
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with `model.disable_adapter()` for A/B.")
    p.add_argument("--save-every", type=int, default=20,
                   help="Persist partial parquet every N batches (resume-safe).")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="Disable the shared-prefix DynamicCache optimization.")
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

    cfg = TrainConfig.from_yaml(args.config)
    if args.max_transcript_chars is not None:
        cfg.max_transcript_chars = args.max_transcript_chars
    if not cfg.system_prompt or not cfg.user_prompt_template:
        raise ValueError(
            f"Config {args.config} is missing system_prompt or user_prompt_template."
        )
    logging.info(
        "Config loaded: include_previous_call_summary=%s, max_transcript_chars=%d, "
        "required columns=%s",
        cfg.include_previous_call_summary, cfg.max_transcript_chars,
        required_columns(cfg),
    )

    df_in = read_input(in_path, cfg)
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

    probe_idx = pending[: args.batch_size]
    probe_rows = [df.loc[i].to_dict() for i in probe_idx]
    probe_texts = texts_for_rows(
        tokenizer, probe_rows, cfg, reasoning_effort=args.reasoning_effort,
    )

    if args.debug_lcp:
        run_lcp_debug(tokenizer, probe_texts, cfg, reasoning_effort=args.reasoning_effort)

    cache_state = PrefixCacheState()
    if not args.no_prefix_cache:
        cache_state = setup_prefix_cache(
            model, tokenizer, probe_texts, cfg,
            reasoning_effort=args.reasoning_effort,
            include_base=args.include_base,
        )

    t0 = time.perf_counter()
    n_batches = 0
    for start in range(0, len(pending), args.batch_size):
        batch_idx = pending[start: start + args.batch_size]
        rows = [df.loc[i].to_dict() for i in batch_idx]
        prompt_texts = texts_for_rows(
            tokenizer, rows, cfg, reasoning_effort=args.reasoning_effort,
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
