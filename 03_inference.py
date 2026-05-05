"""Stage 03: batch inference with adapter + base-model A/B and selectable reasoning effort.

For each input row with a `transcript`:
  1. Generate `history_summary` via the same Stage 01 prompt (summary-only path).
  2. Build messages with system + summary + transcript and render with the requested
     reasoning_effort (low / medium / high). add_generation_prompt=True.
  3. Generate adapter response, parse channels.
  4. Generate base-model response (same prompt) via PEFT's disable_adapter() context,
     parse channels.
  5. Persist all of the above to `--output` JSONL. Resume-safe by transcript hash.

Launch:
    python 03_inference.py --input data/test.jsonl --output data/inference.jsonl \\
        --model-dir /path/to/gpt-oss-120b --adapter-dir checkpoints/final_adapter \\
        --reasoning-effort medium

Inference runs single-process with `device_map="auto"` for tensor-parallel
sharding across local GPUs — do NOT invoke this script via `accelerate launch`
or `deepspeed`. (Training uses `accelerate launch` per cluster policy; inference
does not, since FSDP/ZeRO-3 sharding is for backward passes that don't run here.)
"""

import argparse
import importlib.util
import json
import logging
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse Stage 01 helpers (summary generation, hash, resume).
_spec = importlib.util.spec_from_file_location(
    "preproc01", str(Path(__file__).resolve().parent / "01_preprocessing.py")
)
preproc01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preproc01)

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_ADAPTER_DIR = "checkpoints/final_adapter"

_CHANNEL_RE = re.compile(
    r"<\|channel\|>(?P<channel>[^<]+?)<\|message\|>(?P<content>.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|channel\|>|\Z)",
    re.DOTALL,
)


def parse_channels(raw_text: str) -> dict:
    """Return {'analysis': str, 'final': str} (empty strings if a channel is absent)."""
    out = {"analysis": "", "final": ""}
    for m in _CHANNEL_RE.finditer(raw_text):
        ch = m.group("channel").strip()
        if ch in out:
            out[ch] += m.group("content")
    return out


def build_messages(transcript: str, history_summary: str, context: str = "") -> list[dict]:
    system_content = preproc01.build_agent_system_content(history_summary, context=context)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"transcript:\n{transcript}"},
    ]


def render_input_ids(tokenizer, messages: list[dict], reasoning_effort: str):
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        return tokenizer.apply_chat_template(
            messages, reasoning_effort=reasoning_effort, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_one(model, tokenizer, input_ids, max_new_tokens: int) -> str:
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)


def load_model_with_adapter(model_dir: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def infer_one(model, tokenizer, transcript: str, reasoning_effort: str,
              max_new_tokens: int = 2048,
              context: str = "",
              window_size: int | None = preproc01.DEFAULT_WINDOW_SIZE) -> dict:
    """Full per-sample flow: summary → adapter response → base response → channel parse."""
    history_summary = preproc01.generate_summary_only(
        transcript, model, tokenizer, reasoning_effort="medium",
        window_size=window_size,
    )
    windowed_transcript = preproc01._windowed(transcript, window_size)
    messages = build_messages(windowed_transcript, history_summary, context=context)
    input_ids = render_input_ids(tokenizer, messages, reasoning_effort)

    model_response = generate_one(model, tokenizer, input_ids, max_new_tokens)
    with model.disable_adapter():
        base_response = generate_one(model, tokenizer, input_ids, max_new_tokens)

    m_parsed = parse_channels(model_response)
    b_parsed = parse_channels(base_response)
    return {
        "history_summary": history_summary,
        "reasoning_effort": reasoning_effort,
        "model_response": model_response,
        "model_response_analysis": m_parsed["analysis"],
        "model_response_final": m_parsed["final"],
        "base_response": base_response,
        "base_response_analysis": b_parsed["analysis"],
        "base_response_final": b_parsed["final"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--window-size", type=int, default=preproc01.DEFAULT_WINDOW_SIZE,
                   help="Keep only the last N transcript turns when summarizing and generating. "
                        "Pass 0 to disable.")
    p.add_argument("--context", default="",
                   help="Default context block content. Per-row `context` column overrides this.")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = preproc01.load_processed_hashes(out_path)
    logging.info("Resume: %d rows already in %s", len(seen), out_path)

    model, tokenizer = load_model_with_adapter(args.model_dir, args.adapter_dir)
    logging.info("Model + adapter loaded.")

    n_done, n_fail = 0, 0
    t0 = time.perf_counter()
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.max_samples is not None and (n_done + n_fail) >= args.max_samples:
                break
            row = json.loads(line)
            if "transcript" not in row:
                logging.warning("row %d missing transcript — skipping", i)
                continue
            if preproc01.transcript_hash(row["transcript"]) in seen:
                continue
            try:
                result = infer_one(
                    model, tokenizer, row["transcript"],
                    reasoning_effort=args.reasoning_effort,
                    max_new_tokens=args.max_new_tokens,
                    context=row.get("context", args.context),
                    window_size=args.window_size if args.window_size > 0 else None,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning("row %d failed: %s", i, e)
                n_fail += 1
                continue
            row.update(result)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % args.log_every == 0:
                rate = n_done / (time.perf_counter() - t0)
                logging.info("Done %d (failed %d) | %.2f rows/s", n_done, n_fail, rate)

    logging.info("Finished. Done=%d Failed=%d in %.1fs",
                 n_done, n_fail, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
