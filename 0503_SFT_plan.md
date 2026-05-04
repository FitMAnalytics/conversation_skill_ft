# Plan: Three-Notebook SFT Pipeline for GPT-OSS-120B (Harmony-aware)

## Context

The current LoRA adapter regresses GPT-OSS-120B's analysis channel: at inference, regardless of `reasoning_effort`, output is plain text — no `<|channel|>analysis<|message|>...` block. Root cause: the prior pipeline used custom tokens (`<|system|>`, `<|agent|>`, `<|customer|>`, …) that replaced Harmony entirely, so the model unlearned channel structure. `gpt_oss_inspection.ipynb` cell 7 already flagged this exact failure.

The fix: train on **native Harmony-rendered examples** that include reasoning in the analysis channel, so the structure is reinforced rather than starved.

## Prompt structure (shared across all three stages)

System message has three labelled sections, in this order:
1. Fixed agent system prompt (one paragraph, identical across all examples).
2. **Context block** — always present. Empty by default; populated from a `context` column in the input file when available, or from a `--context` CLI flag at inference. The system prompt instructs the model to pay close attention to it when non-empty (this is a low-effort proxy for full RAG; we'll replace it with retrieval later).
3. **Conversation summary** — the `history_summary` produced by Stage 01 (or by Stage 03's preprocessing call at inference time).

User message holds `transcript:\n<windowed transcript>`. Assistant turn (training only) is rendered as a Harmony two-channel block with `reasoning` in the `analysis` channel and `polished_target` in the `final` channel.

## Look-back windowing

Long transcripts are truncated to the last `--window-size` turns (default **50**) before any token-budget enforcement. Windowing applies in all three stages and is parameter-controllable:
- Stage 01: applied to `transcript` before the preprocessing prompt and the summary-only prompt.
- Stage 02: applied at dataset-build time. If the rendered prefix + assistant suffix still exceed `--max-seq-len` after windowing, the oldest windowed turns are dropped one at a time until it fits. **The system prompt and context block are never truncated** — only transcript turns are dropped. If even an empty transcript overflows, a warning logs the count.
- Stage 03: applied to `transcript` before summary generation AND before model-response generation.

Shared utilities live in `01_preprocessing.py` (`parse_turns`, `window_transcript`, `build_agent_system_content`, `AGENT_SYSTEM_PROMPT`, `DEFAULT_WINDOW_SIZE = 50`) and are imported by Stage 02 and Stage 03 via `importlib`.

## Scope split

- **Initial parsing happens on Amex laptop** (out of scope here). The output is a JSONL/CSV with at least two columns:
  - `transcript` — full prior conversation as plain text (e.g. `agent: ...\ncustomer: ...\nagent: ...`)
  - `target` — the actual agent response we want to clone
- **This plan starts from that file** and produces three notebooks (single-sample, no batching, for testing) + three matching `.py` scripts (batch, production).

## Deliverables

| Stage | Notebook (test) | Script (batch) | Purpose |
|-------|-----------------|----------------|---------|
| 01 Preprocessing | `01_preprocessing.ipynb` | `01_preprocessing.py` | Single GPT-OSS call per sample → polished target + history summary + reasoning + `is_substantial` flag |
| 02 SFT | `02_sft_training.ipynb` | `02_sft_training.py` | Harmony-template SFT with LoRA. Notebook runs ~20 samples × 1 epoch as a smoke test |
| 03 Inference | `03_inference.ipynb` | `03_inference.py` | Apply the same summary preprocessing, then generate adapter + base responses with selectable `reasoning_effort` |

Save final plan also as `0503_SFT_plan.md` in the project root.

---

## 01 — Preprocessing

**Input**: raw file with columns `transcript`, `target` (+ any passthrough cols).

**One GPT-OSS prompt per sample** that returns a structured JSON with four fields:

```json
{
  "polished_target": "<light edit of target — remove um/uh/false starts/repeats; keep wording, tone, contractions>",
  "history_summary": "<1–3 sentence summary of `transcript` only — never references future turns since target isn't in transcript>",
  "reasoning": "<why a top agent would say polished_target given transcript + summary; concrete and grounded, not generic>",
  "is_substantial": true,
  "is_substantial_rationale": "<why true/false — flagged false when customer asked a rules-bound question and agent had no real choice>"
}
```

Prompt design notes:
- Instruct the model to read `transcript` as context and `target` as the answer to explain. Light polishing only — explicitly forbid paraphrasing or shortening.
- `is_substantial=False` examples are kept in the JSONL (with the flag) but **filtered out at training time** by stage 02. This preserves auditability — we can always loosen the filter.
- Use Harmony's `analysis` channel to elicit the reasoning during this prompt too — keeps the reasoner's own thinking high-quality. We capture only the structured JSON from the `final` channel.

**Notebook (`01_preprocessing.ipynb`)**:
- Load the model + tokenizer with `device_map="auto"`, bf16.
- Read first sample from the input file, run the prompt once, pretty-print all four output fields.
- Provide one cell to iterate manually through ~5 samples for spot-checking. No batching, no JSONL writing.

**Script (`01_preprocessing.py`)**:
- CLI flags: `--input`, `--output`, `--model-dir`, `--max-samples`, `--batch-size` (uses HF `pipeline` or manual batched generation — single-node multi-GPU via `device_map="auto"` is fine; no DeepSpeed needed for inference).
- Resume-safe: if output JSONL exists, skip already-processed `transcript` hashes.
- Output JSONL: original columns + `polished_target`, `history_summary`, `reasoning`, `is_substantial`, `is_substantial_rationale`.
- Logs progress every N samples; writes a sidecar `01_failures.jsonl` for parse failures.

---

## 02 — SFT Training

**Input**: JSONL from stage 01.

**Filter**: drop rows where `is_substantial == False`. Log how many remain.

**Build training examples via Harmony chat template**:
- `messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nConversation summary so far: " + history_summary}, {"role": "user", "content": transcript}, {"role": "assistant", "content": <reasoning_in_analysis> + <polished_target_in_final>}]`
- For the assistant turn, render via `tokenizer.apply_chat_template` with the assistant content split across `analysis` and `final` channels — exact mechanism depends on the template's API (need to verify in the inspection notebook output: some forks accept `{"role":"assistant","thinking": reasoning, "content": polished_target}` directly; others require manual construction of `<|start|>assistant<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>`). Lock this down on first sample in the smoke test.
- `add_generation_prompt=False` for training (we want the full assistant turn). Tokenize the full string, then locate `analysis` and `final` channel spans by re-scanning for the framing tokens recorded in `gpt_oss_inspection.ipynb` cell 2.

**Per-token loss weights** (replaces simple `-100` mask):
- `prefix tokens` → 0.0 (everything up to and including the assistant turn header)
- `analysis-channel content` → `analysis_weight` (CLI flag, default **0.2**)
- `final-channel content` → `final_weight` (CLI flag, default **1.0**)
- `framing tokens (<|channel|>, <|message|>, <|end|>, <|return|>)` inside the assistant turn → `final_weight` (these are what we MUST learn to emit — train them at full weight)
- padding → 0.0

Custom `Trainer.compute_loss` override: per-token CE without reduction, multiply by weights, divide by `weights.sum()` per example.

**No special-token registration / `resize_token_embeddings`** — Harmony tokens already exist in the base tokenizer.

**Notebook (`02_sft_training.ipynb`)**:
- Cells: load tokenizer → load 20 samples → build a Harmony example → decode and visually verify framing → set up LoRA (r=64) → run `Trainer.train()` for 1 epoch → save adapter to `checkpoints_smoketest/`.
- Goal: catch format / tokenization bugs in minutes, not hours.

**Script (`02_sft_training.py`)**:
- CLI: `--input`, `--model-dir`, `--output-dir`, `--epochs`, `--analysis-weight` (0.2), `--final-weight` (1.0), `--max-samples`, `--deepspeed` (zero3 preset or path; carry over from existing script's `build_zero3_config`).
- Carry over from existing `02_sft_training.py`: `HfDeepSpeedConfig` registration before `from_pretrained`, ZeRO-3 config with optimizer offload, section-aware truncation (now applied to the user message containing `transcript`), TensorBoard logging, training_log.json dump.
- Drop from existing script: `SPECIAL_TOKENS` block, `add_special_tokens` call, `resize_token_embeddings`.
- Note: the MXFP4 + ZeRO-3 incompatibility documented in `environment_limitations.md` still blocks production runs of GPT-OSS-120B until torch ≥ 2.7 / triton ≥ 3.4. The smoke test in the notebook can run on a smaller base model meanwhile. Plan does not address that env upgrade — it is a separate workstream.

---

## 03 — Inference

**Input**: file with at least `transcript` (and optionally other passthrough columns).

**Per-sample flow**:
1. Run the **same summary-generation prompt** from stage 01 on the transcript → `history_summary`. (We don't generate a target during inference — we generate one response. So a thinner version of the stage-01 prompt that only asks for `history_summary` is sufficient. Implement as `01_preprocessing.py:generate_summary_only(transcript)` and reuse.)
2. Build messages: `[{"role":"system", "content": SYSTEM_PROMPT + summary}, {"role":"user", "content": transcript}]`. Render with `apply_chat_template(add_generation_prompt=True, reasoning_effort=<flag>)`.
3. Generate with adapter loaded → `model_response` (raw text including channel framing).
4. Parse channels using the regex from `gpt_oss_inspection.ipynb` cell 4 → store `model_response_analysis` + `model_response_final` separately.
5. Generate **base model response** for A/B: wrap `model.generate` in `with model.disable_adapter():` (PEFT API) — no need to reload weights. Same parsing → `base_response_analysis`, `base_response_final`.

**Reasoning effort selection**: CLI/notebook flag `--reasoning-effort {low, medium, high}`. Passed through to `apply_chat_template`. Default `medium`.

**Notebook (`03_inference.ipynb`)**:
- Cells: load tokenizer → load base model + attach adapter → define one placeholder `transcript = "agent: ...\ncustomer: ...\n..."` → run preprocessing-summary → generate adapter response → generate base response → display both with channels parsed.
- Cell to swap `reasoning_effort` and re-generate without reloading.

**Script (`03_inference.py`)**:
- CLI: `--input`, `--output`, `--model-dir`, `--adapter-dir`, `--reasoning-effort` (default medium), `--batch-size`, `--max-samples`.
- Loops over input rows in batches. For each row, persists: original columns, `history_summary`, `model_response`, `model_response_analysis`, `model_response_final`, `base_response`, `base_response_analysis`, `base_response_final`, `reasoning_effort`.
- Resume-safe (skip rows whose hash is in the existing output).

---

## Verification

In order, each step is a hard gate before the next:

1. **Stage 01 notebook on 1 sample**: structured JSON parses, all four fields are sensible, `is_substantial` rationale is grounded.
2. **Stage 01 script on 50 samples**: no failures, distribution of `is_substantial` looks right (not 100% true / not 0% true).
3. **Stage 02 notebook smoke test (20 samples × 1 epoch)**:
   - Decoded first example contains exactly one `<|channel|>analysis<|message|>...<|end|>` and one `<|channel|>final<|message|>...<|return|>` block inside the assistant turn.
   - `weights.sum()` matches `0.2 * len(analysis_tokens) + 1.0 * (len(final_tokens) + len(framing_tokens))`.
   - Training loss decreases over the 1-epoch run.
4. **Stage 03 notebook on placeholder transcript** with adapter from step 3:
   - Adapter response **must** contain a populated analysis channel — this is the central regression test.
   - Base response also contains analysis (sanity that the disable_adapter path works).
   - Switching `reasoning_effort` from low→high visibly grows analysis token count.
5. **Stage 03 script on 50 held-out samples**: spot-check 5 — adapter output should sound conversational (not robotic prose), reasoning should be substantive, `final` should resemble a real top-agent response.

If step 4's adapter response has no analysis channel, the most likely culprits are (in order): assistant-turn rendering didn't actually emit two channels (re-check stage 02 first-example decode); `analysis_weight=0.2` was masked to 0 by a bug; framing tokens were given weight 0 instead of `final_weight`.

## Files this plan creates / modifies

- New: `01_preprocessing.ipynb`, `01_preprocessing.py`
- Rewrite: `02_sft_training.ipynb` (currently doesn't exist as a standalone notebook in this scope), `02_sft_training.py` (heavy edits — drop custom tokens, add weighted loss, switch to Harmony rendering)
- New: `03_inference.ipynb`, `03_inference.py` (existing `03_inference.ipynb` is in `archived/`, will not be reused)
- New: `0503_SFT_plan.md` (this plan, copied to project root)
- The old `01_preprocessing.ipynb` and `archived/03_inference.ipynb` stay where they are for historical reference.
