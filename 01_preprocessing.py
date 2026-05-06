"""Stage 01: turn (transcript, target) rows into SFT-ready records.

For each row, calls GPT-OSS once to produce:
  - polished_target           light disfluency cleanup of `target`
  - history_summary           1-3 sentence summary of `transcript` only
  - reasoning                 why a top agent says polished_target here
  - is_substantial            whether the agent had real choice in this turn
  - is_substantial_rationale  one-liner explaining the flag

Output JSONL = original columns + the five fields above.
Resume-safe: skips rows whose `transcript` SHA1 already appears in --output.

Launch:
    python 01_preprocessing.py --input data/raw.jsonl --output data/preprocessed.jsonl
"""

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = "/path/to/gpt-oss-120b"
DEFAULT_WINDOW_SIZE = 50  # max turns of transcript fed to the model

# Fixed system prompt for the SFT model — used by stages 02 and 03. Stage 01
# uses its own preprocessing-specific prompt below.
AGENT_SYSTEM_PROMPT = (
    "You are an outbound sales agent for American Express. Your goal is to "
    "identify customer needs and guide the conversation toward conversion. "
    "Use the Context section, the conversation summary, and the recent transcript "
    "to decide your next response. If the Context section below is non-empty, pay "
    "close attention to it — it may contain product details, offers, or campaign "
    "information that should shape your reply."
)


def build_agent_system_content(history_summary: str, context: str = "") -> str:
    """System message for the SFT agent — system prompt + context block + summary.

    The context block is always present (even when empty) so the model has a
    consistent slot to attend to. The system prompt tells the model to weight it
    heavily when populated.
    """
    ctx_text = context.strip() if context else "(none)"
    return (
        f"{AGENT_SYSTEM_PROMPT}\n\n"
        f"Context: {ctx_text}\n\n"
        f"Conversation summary so far: {history_summary}"
    )


_TURN_START = re.compile(r"^(agent|customer):\s*", re.MULTILINE)


def parse_turns(transcript: str) -> list[tuple[str, str]]:
    """Split a `agent: ... / customer: ...` transcript into ordered (speaker, text) tuples.

    Each match of the speaker prefix starts a new turn; the turn's text runs to the
    next match (or end of string), so utterances spanning multiple lines are kept intact.
    """
    matches = list(_TURN_START.finditer(transcript))
    turns = []
    for i, m in enumerate(matches):
        speaker = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(transcript)
        turns.append((speaker, transcript[start:end].strip()))
    return turns


def render_turns(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in turns)


def window_transcript(transcript: str, window_size: int) -> str:
    """Keep only the last `window_size` turns. Preserves recency."""
    turns = parse_turns(transcript)
    if len(turns) <= window_size:
        return transcript.strip()
    return render_turns(turns[-window_size:])


PREPROC_SYSTEM_PROMPT = """You are a data-prep assistant for an outbound-sales SFT pipeline.
You will receive a call transcript (`transcript`) and the agent's actual next response (`target`).
Produce ONE JSON object — and nothing else — with these fields:

  polished_target          string. The `target` with disfluencies (um, uh, false starts,
                           repeated words) removed. KEEP the original wording, tone,
                           contractions, and sentence structure. Do NOT paraphrase or shorten.

  history_summary          string. 1-3 sentences summarizing what has happened in `transcript`
                           so far. Cover only what is in `transcript` — do not reference the
                           target, since the target hasn't happened yet at this point.

  reasoning                string. Concrete reasoning for why a top-performing agent would say
                           `polished_target` given the conversation so far. Reference specific
                           customer signals from the transcript. Avoid generic platitudes.

  is_substantial           boolean. true if the agent had genuine freedom of choice in how to
                           respond. false if the customer asked a rules-bound question (e.g.
                           specific fee disclosure, compliance script) where the agent had no
                           real latitude — these aren't useful behavioral-cloning examples.

  is_substantial_rationale string. One sentence explaining why is_substantial is true or false.

Return ONLY the JSON object in the final channel. No code fences, no prose around it."""

SUMMARY_ONLY_SYSTEM_PROMPT = """You are a call-summarization assistant.
Given a sales call `transcript`, produce ONE JSON object with a single field:

  history_summary  string. 1-3 sentences describing what has happened in the call so far.

Return ONLY the JSON object in the final channel."""

_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(?P<content>.*?)(?=<\|return\|>|<\|end\|>|\Z)",
    re.DOTALL,
)


def transcript_hash(transcript: str) -> str:
    return hashlib.sha1(transcript.encode("utf-8")).hexdigest()


def load_processed_hashes(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(transcript_hash(json.loads(line)["transcript"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def load_model(model_dir: str):
    """Load tokenizer + model with the inspection-notebook's working recipe.

    `torch_dtype=torch.bfloat16` (not `dtype="auto"`): on torch 2.6 / triton 3.2
    the MXFP4 quantizer can't keep weights packed and dequantizes every expert
    to bf16 at load time (~240 GB). With `dtype="auto"` the dequant intermediates
    land on cuda:0 before `device_map="auto"` dispatches the modules, OOMing the
    first GPU. Forcing `torch_dtype=torch.bfloat16` routes those tensors through
    device_map and shards the load across visible GPUs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _build_user_message(transcript: str, target: str | None) -> str:
    if target is None:
        return f"transcript:\n{transcript}"
    return f"transcript:\n{transcript}\n\ntarget:\n{target}"


def _windowed(transcript: str, window_size: int | None) -> str:
    if window_size is None:
        return transcript.strip()
    return window_transcript(transcript, window_size)


def _generate(model, tokenizer, messages: list[dict], reasoning_effort: str,
              max_new_tokens: int) -> str:
    """Render messages, run generation, return only the final-channel text."""
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, reasoning_effort=reasoning_effort, **kwargs
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(messages, **kwargs)

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)
    matches = _FINAL_CHANNEL_RE.findall(raw)
    if not matches:
        # Last resort: everything stripped of specials.
        return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return matches[-1].strip()


def _parse_json_block(text: str) -> dict:
    """Tolerate stray prose around the JSON — find the outermost {...}."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in: {text[:300]}")
    return json.loads(text[start:end + 1])


def preprocess_one(transcript: str, target: str, model, tokenizer,
                   reasoning_effort: str = "high",
                   max_new_tokens: int = 2048,
                   window_size: int | None = DEFAULT_WINDOW_SIZE) -> dict:
    """Run the full preprocessing prompt on one (transcript, target) pair.

    `window_size` keeps only the last N turns of `transcript` before sending it to
    the LLM — pass None to disable windowing.
    """
    messages = [
        {"role": "system", "content": PREPROC_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(_windowed(transcript, window_size), target)},
    ]
    final_text = _generate(model, tokenizer, messages, reasoning_effort, max_new_tokens)
    parsed = _parse_json_block(final_text)
    required = {"polished_target", "history_summary", "reasoning",
                "is_substantial", "is_substantial_rationale"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"missing fields {missing} in: {final_text[:300]}")
    return parsed


def generate_summary_only(transcript: str, model, tokenizer,
                          reasoning_effort: str = "medium",
                          max_new_tokens: int = 512,
                          window_size: int | None = DEFAULT_WINDOW_SIZE) -> str:
    """Used by stage 03 — only the summary, target not available at inference."""
    messages = [
        {"role": "system", "content": SUMMARY_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(_windowed(transcript, window_size), target=None)},
    ]
    final_text = _generate(model, tokenizer, messages, reasoning_effort, max_new_tokens)
    return _parse_json_block(final_text)["history_summary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="JSONL with at least `transcript` and `target`")
    p.add_argument("--output", required=True, help="JSONL output (resume-safe)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                   help="Keep only the last N turns of `transcript` before sending to the LLM. "
                        "Pass 0 to disable windowing.")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    fail_path = out_path.with_name(out_path.stem + "_failures.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = load_processed_hashes(out_path)
    logging.info("Resume: %d rows already in %s", len(seen), out_path)

    model, tokenizer = load_model(args.model_dir)
    logging.info("Model loaded.")

    n_done, n_fail = 0, 0
    t0 = time.perf_counter()
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "a", encoding="utf-8") as fout, \
         open(fail_path, "a", encoding="utf-8") as fbad:
        for i, line in enumerate(fin):
            if args.max_samples is not None and (n_done + n_fail) >= args.max_samples:
                break
            row = json.loads(line)
            if "transcript" not in row or "target" not in row:
                logging.warning("row %d missing transcript/target — skipping", i)
                continue
            if transcript_hash(row["transcript"]) in seen:
                continue
            try:
                parsed = preprocess_one(
                    row["transcript"], row["target"], model, tokenizer,
                    reasoning_effort=args.reasoning_effort,
                    window_size=args.window_size if args.window_size > 0 else None,
                )
            except Exception as e:  # noqa: BLE001 — preprocessing failures are isolated per row
                fbad.write(json.dumps({"row_index": i, "error": str(e), **row}) + "\n")
                fbad.flush()
                n_fail += 1
                continue
            row.update(parsed)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % args.log_every == 0:
                rate = n_done / (time.perf_counter() - t0)
                logging.info("Done %d (failed %d) | %.2f rows/s", n_done, n_fail, rate)

    logging.info("Finished. Done=%d Failed=%d in %.1fs", n_done, n_fail,
                 time.perf_counter() - t0)


if __name__ == "__main__":
    main()
