"""Run `03_inference_cot_0520.py` against every checkpoint in a training run.

Thin wrapper: discovers every `checkpoint-N/` directory under `--adapter-root`
(plus the `final_adapter/` if present), runs the same batch inference path as
`03_inference_cot_0520.py` on each, and writes a separate per-checkpoint
parquet to `--output-dir`. Output filenames follow:

    {output_dir}/predictions_checkpoint-30.parquet
    {output_dir}/predictions_checkpoint-60.parquet
    ...
    {output_dir}/predictions_final_adapter.parquet

Loads the base model + tokenizer ONCE, then hot-swaps the LoRA adapter between
runs via `PeftModel.load_adapter` / `delete_adapter`. This avoids paying the
multi-GB base-model load N times when sweeping a long training run.

The prefix-cache KV is rebuilt at the start of each checkpoint (cached KV is
adapter-dependent, so it would be stale across swaps).

Existing per-checkpoint parquets are skipped by default (`--skip-existing`).
Drop `--skip-existing` to overwrite, or delete the per-checkpoint file to
force a re-run for just that one.

Example:
    python 03_inference_all_checkpoints_0520.py \\
        --config 02_train_config_cot_0520.yaml \\
        --input data/eval_holdout.parquet \\
        --output-dir data/eval_predictions_by_checkpoint/ \\
        --adapter-root checkpoints_cot_0520 \\
        --model-dir /path/to/gpt-oss-20b \\
        --batch-size 4 --max-new-tokens 1024 --reasoning-effort medium
"""

import argparse
import importlib.util
import logging
import re
import time
from pathlib import Path

import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(_HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the single-checkpoint inference module verbatim.
_inf = _load_module("inf03_cot", "03_inference_cot_0520.py")
TrainConfig = _inf.TrainConfig
load_model_with_adapter = _inf.load_model_with_adapter
texts_for_rows = _inf.texts_for_rows
generate_batch_decoded = _inf.generate_batch_decoded
setup_prefix_cache = _inf.setup_prefix_cache
PrefixCacheState = _inf.PrefixCacheState
parse_channels = _inf.parse_channels
read_input = _inf.read_input
init_output_columns = _inf.init_output_columns
load_existing_output = _inf.load_existing_output
pending_indices = _inf.pending_indices
required_columns = _inf.required_columns
run_lcp_debug = _inf.run_lcp_debug

DEFAULT_MODEL_DIR = _inf.DEFAULT_MODEL_DIR

# Matches `checkpoint-<int>` so we can sort numerically rather than lexically
# (otherwise checkpoint-100 sorts before checkpoint-90).
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================

def discover_checkpoints(adapter_root: Path,
                         include_final: bool,
                         step_filter: list[int] | None) -> list[tuple[str, Path]]:
    """Return [(label, path)] for all checkpoints to run, sorted by step number.

    `final_adapter` (if present) is appended LAST so the sweep ends at the
    most-trained model. Use `step_filter` to restrict to a specific subset of
    step numbers; ignored for `final_adapter`.
    """
    if not adapter_root.is_dir():
        raise FileNotFoundError(f"--adapter-root does not exist: {adapter_root}")

    pairs = []
    for entry in adapter_root.iterdir():
        if not entry.is_dir():
            continue
        m = _CKPT_RE.match(entry.name)
        if not m:
            continue
        step = int(m.group(1))
        if step_filter is not None and step not in step_filter:
            continue
        if not (entry / "adapter_config.json").exists():
            logging.warning("%s has no adapter_config.json — skipping.", entry)
            continue
        pairs.append((step, entry.name, entry))

    pairs.sort(key=lambda t: t[0])
    result = [(name, path) for _step, name, path in pairs]

    if include_final:
        final = adapter_root / "final_adapter"
        if final.is_dir() and (final / "adapter_config.json").exists():
            result.append(("final_adapter", final))
        else:
            logging.info("No final_adapter found under %s.", adapter_root)

    return result


# ============================================================================
# ADAPTER HOT-SWAP
# ============================================================================

def swap_adapter(model, adapter_dir: Path, name: str = "default") -> None:
    """Replace the currently-active PEFT adapter on `model` with the one in
    `adapter_dir`. Constant-memory: deletes the previous adapter under the same
    name before loading the new one.
    """
    # PEFT raises if the name doesn't exist; ignore that — we want idempotence
    # so the same path works whether or not an adapter was previously loaded.
    try:
        model.delete_adapter(name)
    except (ValueError, KeyError):
        pass
    model.load_adapter(str(adapter_dir), adapter_name=name)
    model.set_adapter(name)


# ============================================================================
# PER-CHECKPOINT INFERENCE LOOP
# ============================================================================

def run_one_checkpoint(model, tokenizer, df_template: pd.DataFrame,
                       cfg: TrainConfig, args: argparse.Namespace,
                       out_path: Path, debug_lcp_this: bool) -> None:
    """Run the full inference loop for one checkpoint, writing to `out_path`.

    Mirrors the body of `03_inference_cot_0520.py:main()` from "load_existing_output"
    onward. `df_template` is the validated input frame (read once at the wrapper
    level); we make a working copy here so resume state per checkpoint is independent.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_existing_output(out_path, df_template, args.include_base)
    pending = pending_indices(df)
    logging.info("[%s] %d rows pending (resume-skipped %d).",
                 out_path.name, len(pending), len(df) - len(pending))

    if args.max_samples is not None:
        pending = pending[: args.max_samples]
        logging.info("[%s] Smoke cap: processing only first %d pending rows.",
                     out_path.name, len(pending))

    if not pending:
        logging.info("[%s] Nothing to do — already complete.", out_path.name)
        return

    probe_idx = pending[: args.batch_size]
    probe_rows = [df.loc[i].to_dict() for i in probe_idx]
    probe_texts = texts_for_rows(
        tokenizer, probe_rows, cfg, reasoning_effort=args.reasoning_effort,
    )

    if debug_lcp_this:
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
        logging.info("[%s] Batch %d done | %d/%d rows | %.2f rows/s",
                     out_path.name, n_batches, done, len(pending), rate)

        if n_batches % args.save_every == 0:
            df.to_parquet(out_path, index=False)
            logging.info("[%s] Checkpointed → %s (every %d batches)",
                         out_path.name, out_path, args.save_every)

    df.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    logging.info("[%s] Finished. %d rows in %.1fs (%.2f rows/s). Wrote %s",
                 out_path.name, len(pending), elapsed, len(pending) / elapsed, out_path)


# ============================================================================
# CLI
# ============================================================================

def _parse_step_filter(s: str | None) -> list[int] | None:
    """Parse e.g. '30,60,90' or '30-120:30' (range with step) into a list of ints."""
    if s is None:
        return None
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            range_part, _, step_part = tok.partition(":")
            lo_s, _, hi_s = range_part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            step = int(step_part) if step_part else 1
            out.update(range(lo, hi + 1, step))
        else:
            out.add(int(tok))
    return sorted(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True,
                   help="Training YAML (same one used by 02_sft_training_cot_0520.py).")
    p.add_argument("--input", required=True, help="Parquet (or .csv) of eval rows.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for per-checkpoint parquets. "
                        "Files written as predictions_<checkpoint-name>.parquet.")
    p.add_argument("--adapter-root", required=True,
                   help="Training output dir (e.g. checkpoints_cot_0520) that holds "
                        "checkpoint-N/ subdirs and optionally final_adapter/.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="Base gpt-oss directory (tokenizer + base weights).")
    p.add_argument("--include-final-adapter", action="store_true", default=True,
                   help="Also evaluate final_adapter/ at the end of the sweep. Default: on.")
    p.add_argument("--no-final-adapter", dest="include_final_adapter",
                   action="store_false",
                   help="Skip final_adapter/ and run only checkpoint-* dirs.")
    p.add_argument("--steps", default=None,
                   help="Restrict to specific checkpoint steps. Comma-separated ints "
                        "and/or ranges with optional stride, e.g. '30,60,90' or "
                        "'30-300:30'. final_adapter is unaffected by this filter.")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip checkpoints whose output parquet already exists. Default: on.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Always re-run, overwriting existing per-checkpoint parquets.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-transcript-chars", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with `model.disable_adapter()` per checkpoint.")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--debug-lcp-first", action="store_true",
                   help="Run the LCP debug printer on the first checkpoint only.")
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
    logging.info("Config: include_previous_call_summary=%s, max_transcript_chars=%d, required=%s",
                 cfg.include_previous_call_summary, cfg.max_transcript_chars,
                 required_columns(cfg))

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(args.adapter_root)

    df_in = read_input(in_path, cfg)
    logging.info("Loaded %d rows from %s", len(df_in), in_path)

    step_filter = _parse_step_filter(args.steps)
    ckpts = discover_checkpoints(adapter_root, args.include_final_adapter, step_filter)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found under {adapter_root} "
            f"(steps={args.steps!r}, include_final={args.include_final_adapter})."
        )
    logging.info("Will evaluate %d checkpoint(s): %s",
                 len(ckpts), [name for name, _ in ckpts])

    # Plan which checkpoints actually need running (skip-existing filter).
    to_run = []
    for label, ckpt_path in ckpts:
        out_path = out_dir / f"predictions_{label}.parquet"
        if args.skip_existing and out_path.exists():
            # Match the resume convention used inside run_one_checkpoint:
            # also skip if every row has a non-null `final`.
            try:
                existing = pd.read_parquet(out_path)
                if "final" in existing.columns and existing["final"].notna().all() \
                        and len(existing) == len(df_in):
                    logging.info("Skip %s — %s already complete.", label, out_path)
                    continue
            except Exception as e:  # noqa: BLE001
                logging.warning("Could not read existing %s (%s); will re-run.", out_path, e)
        to_run.append((label, ckpt_path, out_path))

    if not to_run:
        logging.info("All checkpoints already have complete output parquets. Nothing to do.")
        return

    # Load base + first adapter ONCE; hot-swap for the rest.
    first_label, first_path, first_out = to_run[0]
    logging.info("Loading base model + first adapter (%s)...", first_label)
    t0 = time.perf_counter()
    model, tokenizer = load_model_with_adapter(args.model_dir, str(first_path))
    logging.info("Base + first adapter loaded in %.1fs.", time.perf_counter() - t0)

    for i, (label, ckpt_path, out_path) in enumerate(to_run):
        logging.info("=" * 80)
        logging.info("[%d/%d] Running %s → %s", i + 1, len(to_run), label, out_path)
        logging.info("=" * 80)
        if i > 0:
            t0 = time.perf_counter()
            swap_adapter(model, ckpt_path)
            logging.info("Adapter swapped to %s in %.2fs.", label, time.perf_counter() - t0)

        run_one_checkpoint(
            model, tokenizer, df_in, cfg, args, out_path,
            debug_lcp_this=(args.debug_lcp_first and i == 0),
        )

    logging.info("All checkpoints done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
