"""Run `03_inference_cot_0520.py` over every checkpoint of a training output.

Wraps the per-row inference path from `03_inference_cot_0520.py`. The base
model + tokenizer are loaded ONCE; LoRA adapters are loaded by name and
swapped between sweeps via `model.set_adapter(...)` — so on a 20B / 120B
base, the dominant per-checkpoint cost is the generate loop, not model load.

Discovers every `checkpoint-N/` subdirectory of `--checkpoints-dir` plus
the `final_adapter/` subdirectory (unless `--no-final-adapter` is given),
sorted by step N ascending. Writes one parquet per adapter:

    {output-dir}/{output-prefix}_{adapter_name}.parquet

Per-adapter outputs are resume-safe via the same `load_existing_output` /
`pending_indices` logic as `03_inference_cot_0520.py` — re-running skips
already-completed rows per adapter.

Example:
    python batch_inference_multiple_checkpoints_cot_0520.py \\
        --config 02_train_config_cot_0520.yaml \\
        --input data/eval_holdout.parquet \\
        --output-dir data/sweep_cot_0520 \\
        --checkpoints-dir checkpoints_cot_0520 \\
        --model-dir /path/to/gpt-oss-20b \\
        --batch-size 4 --max-new-tokens 1024

To restrict to specific checkpoints:
    --steps 30,60,90              # only those checkpoint-N dirs
    --no-final-adapter            # skip final_adapter/
    --output-prefix preds_v2      # default is "preds"
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


# Reuse everything from the single-checkpoint inference script verbatim.
_inf = _load_module("inference_cot_0520", "03_inference_cot_0520.py")
TrainConfig = _inf.TrainConfig
texts_for_rows = _inf.texts_for_rows
generate_batch_decoded = _inf.generate_batch_decoded
parse_channels = _inf.parse_channels
setup_prefix_cache = _inf.setup_prefix_cache
PrefixCacheState = _inf.PrefixCacheState
required_columns = _inf.required_columns
read_input = _inf.read_input
load_existing_output = _inf.load_existing_output
pending_indices = _inf.pending_indices


# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================

def discover_adapters(checkpoints_dir: Path,
                      include_final: bool,
                      step_filter: set[int] | None) -> list[tuple[str, Path]]:
    """Find all `checkpoint-N` dirs (sorted by N) and optionally `final_adapter`.

    Each candidate must contain `adapter_config.json` to count. Returns a list
    of (display_name, adapter_dir_path) tuples.
    """
    items: list[tuple[int, str, Path]] = []
    for sub in checkpoints_dir.iterdir():
        if not sub.is_dir():
            continue
        m = re.match(r"^checkpoint-(\d+)$", sub.name)
        if not m:
            continue
        if not (sub / "adapter_config.json").exists():
            logging.warning("Skipping %s — no adapter_config.json", sub)
            continue
        step = int(m.group(1))
        if step_filter is not None and step not in step_filter:
            continue
        items.append((step, sub.name, sub))
    items.sort(key=lambda x: x[0])
    out: list[tuple[str, Path]] = [(name, path) for _, name, path in items]

    if include_final:
        final = checkpoints_dir / "final_adapter"
        if (final / "adapter_config.json").exists():
            out.append(("final_adapter", final))
        else:
            logging.warning("final_adapter/ not found under %s (skipping).",
                            checkpoints_dir)

    return out


# ============================================================================
# MODEL LOAD + ADAPTER SWAP
# ============================================================================

def load_base_with_all_adapters(model_dir: str,
                                adapters: list[tuple[str, Path]]):
    """Load base + tokenizer once; register every adapter under its display name.

    Falls back to the default "default" adapter name if the installed PEFT
    version doesn't support the `adapter_name` kwarg on `from_pretrained`.
    Returns (model, tokenizer, name_alias) where name_alias maps each display
    name to the actual PEFT adapter name (usually the same; "default" is the
    only exception when the fallback is taken for the first adapter).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )

    first_name, first_path = adapters[0]
    name_alias: dict[str, str] = {}
    try:
        model = PeftModel.from_pretrained(
            base, str(first_path), adapter_name=first_name
        )
        name_alias[first_name] = first_name
    except TypeError:
        # Older PEFT: no adapter_name kwarg on from_pretrained → falls back to "default".
        logging.warning(
            "PEFT.from_pretrained() does not accept adapter_name; "
            "first adapter %s registered as 'default'.", first_name,
        )
        model = PeftModel.from_pretrained(base, str(first_path))
        name_alias[first_name] = "default"

    model.eval()

    for name, path in adapters[1:]:
        logging.info("Loading adapter %s from %s", name, path)
        model.load_adapter(str(path), adapter_name=name)
        name_alias[name] = name

    return model, tokenizer, name_alias


# ============================================================================
# PER-ADAPTER INFERENCE LOOP
# ============================================================================

def run_one_adapter(model, tokenizer, cfg: TrainConfig,
                    df_in: pd.DataFrame, out_path: Path,
                    args: argparse.Namespace) -> None:
    """Run inference for the currently-activated adapter; write to `out_path`."""
    df = load_existing_output(out_path, df_in, args.include_base)
    pending = pending_indices(df)
    logging.info("[%s] %d rows pending (resume-skipped %d).",
                 out_path.name, len(pending), len(df) - len(pending))

    if args.max_samples is not None:
        pending = pending[: args.max_samples]
        logging.info("[%s] smoke cap → first %d pending rows.",
                     out_path.name, len(pending))

    if not pending:
        logging.info("[%s] nothing to do.", out_path.name)
        return

    probe_idx = pending[: args.batch_size]
    probe_rows = [df.loc[i].to_dict() for i in probe_idx]
    probe_texts = texts_for_rows(
        tokenizer, probe_rows, cfg, reasoning_effort=args.reasoning_effort,
    )

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
        logging.info("[%s] batch %d | %d/%d rows | %.2f rows/s",
                     out_path.name, n_batches, done, len(pending), rate)

        if n_batches % args.save_every == 0:
            df.to_parquet(out_path, index=False)
            logging.info("[%s] checkpointed → %s", out_path.name, out_path)

    df.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    logging.info("[%s] finished. %d rows in %.1fs (%.2f rows/s). Wrote %s",
                 out_path.name, len(pending), elapsed,
                 len(pending) / elapsed if elapsed > 0 else 0.0, out_path)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True,
                   help="Same YAML used to train (e.g. 02_train_config_cot_0520.yaml).")
    p.add_argument("--input", required=True,
                   help="Parquet (or .csv) of eval / hold-out rows.")
    p.add_argument("--output-dir", required=True,
                   help="Directory where per-checkpoint parquets are written.")
    p.add_argument("--checkpoints-dir", required=True,
                   help="Training output dir containing checkpoint-N/ + final_adapter/.")
    p.add_argument("--model-dir", required=True,
                   help="Base gpt-oss model directory (same one used for training).")
    p.add_argument("--steps", default=None,
                   help="Optional comma-separated list of checkpoint step numbers "
                        "to evaluate (e.g. '30,60,120'). Default: all checkpoint-N "
                        "dirs found.")
    p.add_argument("--no-final-adapter", action="store_true",
                   help="Skip final_adapter/ even if it exists.")
    p.add_argument("--output-prefix", default="preds",
                   help="Output filename prefix → {prefix}_{adapter_name}.parquet")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-transcript-chars", type=int, default=None,
                   help="Override the YAML's max_transcript_chars. Default: use YAML.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke cap (per adapter) — first N pending rows only.")
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with `model.disable_adapter()` for A/B.")
    p.add_argument("--save-every", type=int, default=20,
                   help="Per-adapter: persist partial parquet every N batches.")
    p.add_argument("--no-prefix-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.max_transcript_chars is not None:
        cfg.max_transcript_chars = args.max_transcript_chars
    if not cfg.system_prompt or not cfg.user_prompt_template:
        raise ValueError(
            f"Config {args.config} is missing system_prompt or user_prompt_template."
        )

    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.input)

    step_filter: set[int] | None = None
    if args.steps:
        step_filter = {int(s.strip()) for s in args.steps.split(",") if s.strip()}

    adapters = discover_adapters(
        checkpoints_dir,
        include_final=not args.no_final_adapter,
        step_filter=step_filter,
    )
    if not adapters:
        logging.error("No adapters found under %s (steps filter=%s).",
                      checkpoints_dir, step_filter)
        return
    logging.info("Will evaluate %d adapter(s): %s",
                 len(adapters), [n for n, _ in adapters])

    df_in = read_input(in_path, cfg)
    logging.info("Loaded %d input rows from %s | required cols=%s",
                 len(df_in), in_path, required_columns(cfg))

    model, tokenizer, name_alias = load_base_with_all_adapters(args.model_dir, adapters)
    logging.info("Base model + %d adapter(s) loaded.", len(adapters))

    for display_name, _path in adapters:
        peft_name = name_alias[display_name]
        logging.info("=" * 80)
        logging.info("Activating adapter: %s (peft name: %s)", display_name, peft_name)
        model.set_adapter(peft_name)

        out_path = output_dir / f"{args.output_prefix}_{display_name}.parquet"
        run_one_adapter(model, tokenizer, cfg, df_in, out_path, args)


if __name__ == "__main__":
    main()
