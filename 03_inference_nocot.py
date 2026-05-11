"""Batch inference for the no-CoT LoRA SFT adapter — DataFrame in, parquet out.

Mirrors the prompt construction of `02_sft_training_nocot_0511.py` exactly so
adapters saved at any step (final or intermediate `checkpoint-N/`) can be
evaluated with train/inference parity.

Pipeline per row:
  1. Build messages = [system=SYSTEM_CONTENT, user=USER_TMPL.format(4 inputs)]
     using the same templates the training script uses.
  2. Render with `apply_chat_template(..., add_generation_prompt=True)` and the
     requested `reasoning_effort`.
  3. Batched greedy generate (`do_sample=False`).
  4. Decode each row with `skip_special_tokens=False`; parse `analysis` and
     `final` channels out of the raw decode via `infer03.parse_channels`.
  5. (Optional) repeat under `model.disable_adapter()` for an A/B against base.

Input DataFrame columns required:
    transcript, previous_call_summary, current_call_summary, objection_summary
Optional:
    customer_context  (else `train02.get_customer_context(row)` fallback / "")

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

To evaluate an intermediate checkpoint, just point `--adapter-dir` at it:
    --adapter-dir checkpoints_nocot/checkpoint-90
"""

import argparse
import importlib.util
import logging
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


train02 = _load_module("train02_nocot", "02_sft_training_nocot_0511.py")
infer03 = _load_module("infer03", "03_inference.py")

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_ADAPTER_DIR = "checkpoints_nocot/final_adapter"

REQUIRED_COLS = (
    "transcript",
    "previous_call_summary",
    "current_call_summary",
    "objection_summary",
)


def build_prefix_text(tokenizer, row: dict, max_transcript_chars: int) -> str:
    """Replicates `train02.build_prefix_text` minus the truncate-counter side effect."""
    transcript_trimmed, _ = train02.trim_transcript(
        row["transcript"], max_transcript_chars
    )
    user_text = train02.USER_TMPL.format(
        customer_context=row.get("customer_context") or train02.get_customer_context(row),
        previous_call_summary=row["previous_call_summary"],
        current_call_summary=row["current_call_summary"],
        objection_summary=row["objection_summary"],
        transcript=transcript_trimmed,
    )
    messages = [
        {"role": "system", "content": train02.SYSTEM_CONTENT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def render_batch(tokenizer, rows: list[dict], reasoning_effort: str,
                 max_transcript_chars: int) -> dict:
    """Render a batch of rows as a left-padded tokenized batch.

    `reasoning_effort` is injected via the same try/except shim
    `infer03.render_input_ids` uses (tokenizers without that kwarg fall back).
    """
    prefix_texts = [
        build_prefix_text(tokenizer, r, max_transcript_chars) for r in rows
    ]
    # Re-tokenize: apply_chat_template already gave us the strings; tokenize them
    # together with padding so we can batch.
    enc = tokenizer(
        prefix_texts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return enc


def generate_batch(model, tokenizer, enc, max_new_tokens: int) -> list[str]:
    """Greedy batched generate. Returns one decoded string per row (suffix-only)."""
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # left-padded: prompt occupies the last `prompt_len` positions of input_ids,
    # which equals input_ids.shape[1]. The newly generated tokens are appended
    # past that index in `out`.
    prompt_len = input_ids.shape[1]
    suffix_ids = out[:, prompt_len:]
    return tokenizer.batch_decode(suffix_ids, skip_special_tokens=False)


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
                   default=train02.DEFAULT_MAX_TRANSCRIPT_CHARS,
                   help="Char-cap from end of transcript — same default as training.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Smoke cap; processes only first N rows after resume skip.")
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with `model.disable_adapter()` for A/B.")
    p.add_argument("--save-every", type=int, default=10,
                   help="Persist partial parquet every N batches.")
    return p.parse_args()


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
    """Resume: if output parquet exists and aligns row-for-row with input, reuse."""
    if not out_path.exists():
        return init_output_columns(df_in.copy(), include_base)
    existing = pd.read_parquet(out_path)
    if len(existing) != len(df_in):
        logging.warning(
            "Existing output has %d rows but input has %d — ignoring existing "
            "and starting fresh.", len(existing), len(df_in),
        )
        return init_output_columns(df_in.copy(), include_base)
    return init_output_columns(existing, include_base)


def pending_indices(df: pd.DataFrame) -> list[int]:
    """Row indices where `final` is still NA — i.e., not yet generated."""
    mask = df["final"].isna()
    return df.index[mask].tolist()


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

    model, tokenizer = infer03.load_model_with_adapter(args.model_dir, args.adapter_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        decoded = generate_batch(model, tokenizer, enc, args.max_new_tokens)

        for i, raw in zip(batch_idx, decoded):
            parsed = infer03.parse_channels(raw)
            df.at[i, "raw_response"] = raw
            df.at[i, "analysis"] = parsed["analysis"]
            df.at[i, "final"] = parsed["final"]

        if args.include_base:
            with model.disable_adapter():
                base_decoded = generate_batch(model, tokenizer, enc, args.max_new_tokens)
            for i, raw in zip(batch_idx, base_decoded):
                parsed = infer03.parse_channels(raw)
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
