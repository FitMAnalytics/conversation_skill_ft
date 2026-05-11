# TSE SFT Design Doc — Stage 5 Preprocessing + SFT Training

This doc consolidates the design decisions for Layer 1 (SFT) of the TSE
objection-handling model. It covers Stage 5 of preprocessing (teacher CoT
generation), SFT training mechanics (loss masking, channel weighting,
distributed training config, hyperparameters), evaluation, and the
implementation checklist.

---

## Part A — Stage 5 (teacher CoT + response generation)

### A.1 CoT format: free-form natural language prose

Free-form natural prose inside the analysis channel. No JSON, no bullets,
no headers, no enumerated steps. The only structure is the Harmony channel
delimiter (`analysis` vs `final`).

Rationale:
- Schema-constrained CoT degrades reasoning ~10–15% (multiple independent
  studies); models default to "answer-then-rationalize" when schemas allow it.
- DeepSeek-R1 uses only `<think>...</think>` / `<answer>...</answer>` tags
  with free-form prose inside. Internal structure emerged from training,
  not from templating.
- All canonical GPT-OSS fine-tuning examples (OpenAI cookbook, AWS, VESSL,
  HuggingFace Multilingual-Thinking) use free-form prose in the `thinking`
  field.

### A.2 Reasoning style: convergent and deductive

Encourage GPT-OSS's natural convergent-deductive style. Avoid R1-style
"wait, let me reconsider" branchy revisionism.

Rationale:
- Recent comparative SFT study (arxiv 2604.01702) found students trained
  on R1's branchy trajectories generalize ~3–5% worse than students trained
  on GPT-OSS's convergent trajectories — even though R1 data gives lower
  training loss.
- Since we're using GPT-OSS-120B as both teacher and student, the natural
  style transfers cleanly.

### A.3 CoT length: self-modulating

No prescribed length target. Tell the teacher to reason "as much as the
case warrants and no more." Add explicit anti-padding instructions.

Rationale:
- "Longer CoT = better" is a finding from RL with verifiable rewards, not
  from prompting. R1's Thoughtology paper showed performance declines past
  a problem-specific optimal length.
- Sales objection handling is not math; past a point, more reasoning is
  elaboration, not new inference.
- Pushing for longer CoT pushes toward the R1 branchy failure mode.
- Inference latency matters in production deployment.

Diagnostic during inspection: CoT length should correlate with case
complexity. If lengths are flat across complexity, the prompt isn't
inducing modulation.

### A.4 Framing: forward derivation (teacher does NOT see agent response)

The teacher reasons in first person, present tense, from inputs only.
It produces a CoT and a candidate response in one pass. The agent response
is NOT shown to the teacher.

Rationale:
- Train/inference shape match. At inference, the student doesn't have the
  response — it needs to derive it. Training data must be of the same
  shape (forward derivation), not rationalization-shape (third-person,
  knows-the-answer).
- Rationalization framing creates a fundamental mismatch: student learns
  to reason ABOUT a known response, then at inference is asked to PRODUCE
  a response. Different task.

Cost: information asymmetry. Real agents have hidden context (pre-call
research, account notes, relationship feel) that the teacher doesn't see.
Teacher's forward-derived response will often differ from the human
agent's response on a meaningful fraction of cases.

Mitigation (when real teacher API arrives): best-of-N rejection sampling.
Sample N candidate (CoT, response) pairs from the teacher, keep the one
whose response best matches the polished agent response (via LLM-as-judge
or embedding similarity threshold). N=4-8 typical. Drop examples that fail
to match after N tries, or keep with a flag.

**Current state (smoke test):** No best-of-N. Single-sample forward
derivation using GPT-OSS-120B at high reasoning effort. Goal is to eyeball
CoT quality on 15-30 representative examples before scaling. Once GPT-5 or
Claude API access is available (Monday+), switch teacher and add best-of-N.

### A.5 Stage 5b: CoT quality check (not consistency filter)

Under forward-derivation framing, Stage 5b becomes a CoT quality check
rather than a consistency filter. Use LLM-as-judge to assess:
- Does the analysis reference specific elements of the customer context?
- Does it engage with the substance of the objection?
- Does it provide a coherent rationale that leads to the response?

Drop examples that fail on multiple counts.

When best-of-N is added later: 5b also computes the embedding/judge
similarity used for best-of-N selection. Reuses the same matcher.

### A.6 Final Stage 5 prompt (smoke-test version)

```python
TSE_BACKGROUND = """
[PLACEHOLDER — Yi to fill in TSE channel background:
- What TSE is and who it serves
- Channel objectives (deepen relationship, increase usage, expand spend)
- Customer segment (existing card-holding business customers)
- Common objection types
- Compliance/regulatory context relevant to agent responses
]
"""

STAGE5_SYSTEM = f"""You are an outbound sales agent at American Express on the TSE
(Tele Strategic Expansion) channel.

{TSE_BACKGROUND}

You will be given the customer's profile, product/campaign context, prior call
history, current call state, the current objection, and the transcript up to
this point. You need to think about how to respond, then respond.

In the analysis channel, reason as the agent — first person, present tense,
the way you'd think silently between hearing the objection and choosing what
to say. Reason in flowing natural prose: not bullets, not headers, not JSON,
not enumerated steps. Ground your thinking in the specific inputs in front
of you: what does this customer's industry and history tell you, what has
already happened in this call, what is the objection actually signaling
beneath its wording, what response options do you have, and which one best
serves the relationship and the call objective.

Reason as much as the case warrants and no more. Simple cases warrant brief
reasoning; complex cases warrant more. Do not pad. Do not restate the inputs.
Do not enumerate steps mechanically. Do not second-guess yourself in circles;
once you've weighed the options and chosen, move on.

In the final channel, produce ONE natural agent turn — what you would actually
say next, in the words you would speak. No script formatting, no headers,
no bullets, no stage directions. The final-channel response should be the
natural conclusion of the reasoning you just did."""

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

Reason about how to handle this objection in the analysis channel, and
produce the agent turn you would say next in the final channel.
"""
```

---

## Part B — SFT Training

### B.1 Channel loss masking and weighting

**Mask:** user/system tokens are masked (excluded from loss, label=-100).
Both assistant channels (analysis + final) contribute to loss.

**Loss formulation:** sum of per-example per-channel means, then batch mean.

For each example in a batch:
1. `loss_a_i = mean(per-token loss across analysis tokens in example i)`
2. `loss_f_i = mean(per-token loss across final tokens in example i)`
3. `loss_i = w_a * loss_a_i + w_f * loss_f_i`

Batch loss = `mean(loss_i across examples)`.

**Default weights:** `w_a = 1.0`, `w_f = 1.0` (pure sum-of-means).
This naturally neutralizes the length asymmetry (analysis is typically
3-10× longer than final, so unweighted token-level loss would be dominated
by analysis). Per-channel-mean within each example puts equal pressure on
each channel regardless of length.

**Placeholder for adjustment:** weights are exposed as a config knob so we
can shift to e.g. 30/70 (`w_a=0.3, w_f=0.7`) if monitoring suggests one
channel is learning poorly. Diagnostic: log `loss_a` and `loss_f`
separately at every step.

### B.2 Distributed training: FSDP + bf16

**FSDP** with the existing tested config (DeepSpeed comparison deferred to
post-v1 when tech support is set up).

**bf16 mixed precision:** required for GPT-OSS — fp16 diverges due to MoE
down-projection outliers; pure bf16 throughout would cause optimizer-state
drift. Correct config:
- `param_dtype = bfloat16`
- `reduce_dtype = bfloat16`
- `buffer_dtype = bfloat16`
- Optimizer states kept in fp32

**SwiGLU clamp verified:** `self.limit = 7.0` confirmed in installed
`transformers` modeling_gpt_oss.py, with clamp applied to both gate and
up projections. This is the load-bearing numerical safeguard for GPT-OSS
in bf16. No action needed.

**Cluster:** 8×H100, 2 faulty. Use `CUDA_VISIBLE_DEVICES` to target 6
healthy GPUs. Run detached via `nohup ... & disown`.

### B.3 LoRA configuration

**Target modules:** attention projections only (q/k/v/o) for v1. Avoids
the delicate MoE down-projection territory where Unsloth found outliers.
~47.7M trainable params (from prior runs).

**Rank/alpha:** TBD — confirm what previous `sft_20260417.py` runs used.
Conservative defaults if starting fresh: r=16, alpha=32, dropout=0.

### B.4 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 | Conservative. GPT-OSS is strong zero-shot; risk is overwriting, not underlearning. |
| LR schedule | Cosine | 3% warmup ratio |
| Optimizer | AdamW (torch_fused) | Plain AdamW fine for LoRA-size optimizer state |
| Per-device batch size | 2 | bf16 + FSDP constraint |
| Gradient accumulation | TBD | Set to reach effective batch size ~16-32 |
| Max sequence length | TBD | Compute 95th/99th percentile from preprocessed data |
| Epochs | 3 | Small data; watch for overfit by val loss |
| Gradient checkpointing | ON | Trade ~30% compute for activation memory |
| Weight decay | 0.0 | LoRA convention |

### B.5 Checkpointing and logging

- Save checkpoint every 30 steps + at end of each epoch
- `save_total_limit = 10` to bound disk usage (LoRA adapters ~200MB each)
- Load best by val loss at end (`load_best_model_at_end=True`,
  `metric_for_best_model = "eval_loss"`)
- Logging: wandb if allowed in corporate env, else tensorboard
- **Log per-step:** total loss, `loss_a`, `loss_f`, learning rate, grad norm
- **Log per-eval:** `eval_loss`, `eval_loss_a`, `eval_loss_f`

### B.6 Train/val split

**Customer-level split**, 90/10. NOT call-level, NOT row-level.

Rationale: a single customer may appear across multiple calls (prior +
current). Splitting by call ID still leaks customer context. Splitting
by row directly leaks at the call level too. Customer-ID-level split is
the only one that avoids leakage entirely.

Implementation:
1. Get unique customer IDs from the parquet.
2. Random 90/10 split on customer IDs (seed for reproducibility).
3. All objection rows from train customers → train set; same for val.
4. Verify val set has enough rows (~500+) for stable val loss.

### B.7 Evaluation

**Primary:** total val loss + per-channel val loss (val_loss_a, val_loss_f).

**Secondary (per epoch or post-training):**
- Per-scenario val loss breakdown — identify rare/poorly-served scenarios
- Spot-check generation: sample 20-50 val examples, generate completions,
  eyeball CoT and final quality
- LLM-as-judge on response quality with rubric — once stable, automate

**Deferred to later iterations:** scenario-specific rebalancing only if
per-scenario loss reveals both poor quality AND disproportionate business
cost (per earlier decision).

---

## Part C — Implementation Checklist

### Preprocessing
- [ ] Update `STAGE5_SYSTEM` and `STAGE5_USER_TMPL` in `preprocess_lib.py`
- [ ] Fill in `TSE_BACKGROUND` placeholder (Yi)
- [ ] Run Stage 5 smoke test on 15-30 representative examples
- [ ] Eyeball CoT quality, length distribution vs case complexity,
      candidate-response neighborhood vs actual agent response
- [ ] When teacher API available (GPT-5 / Claude): add best-of-N matching
      logic and switch teacher

### Training setup (this design doc)
- [ ] Confirm previous `sft_20260417.py` LoRA hyperparameters
- [ ] Compute token length distribution after preprocessing → set
      `max_seq_length`
- [x] Build custom data collator that tracks channel boundaries and
      emits a `channel_mask` tensor alongside `input_ids`/`labels`
      *(done 2026-05-10 — see Part E.2)*
- [x] Build custom `compute_loss` that splits per-channel and applies
      Version A formulation (per-example per-channel means, summed,
      then batch mean) *(done 2026-05-10 — `ChannelMeanLossTrainer`)*
- [x] Build customer-level 90/10 train/val split *(done 2026-05-10 —
      `customer_level_split()`, falls back to `cust_id`)*
- [x] Set up logging — tensorboard, per-channel diagnostics on both
      train and eval *(done 2026-05-10)*
- [x] Run verification notebook to confirm channel masking and loss
      computation are correct *(merged into `02_sft_training.ipynb`
      cells 5 / 5b / 5c — see Part E.3)*
- [ ] Glue step: write `data/preprocessed.jsonl` from the
      `01_preprocessing.py` parquet so the JSONL fields the script
      reads (`transcript`, `history_summary`, `reasoning`,
      `polished_target`, `context`, `customer_id`, `is_substantial`)
      are populated. Field names are remappable via `TrainConfig` if
      your column names differ.
- [ ] Launch training (see Part E.4 for the commands)

### Post-training
- [ ] Per-channel val loss curves — sanity check
- [ ] Per-scenario val loss breakdown
- [ ] Generation spot-check on 20-50 val examples
- [ ] LLM-as-judge eval if rubric ready

---

## Part E — Implementation Status (2026-05-10)

Translates Parts B and C into the concrete files in this repo and records
the few places where implementation choices needed to be made beyond what
the design doc specified.

### E.1 Files

| File | Role |
|---|---|
| `02_sft_training.py` | Training script. Loads JSONL, splits by customer, tokenizes, applies LoRA, runs `ChannelMeanLossTrainer`. Runnable directly (single GPU) or via `accelerate launch` (multi-GPU). |
| `02_train_config.yaml` | YAML config consumed by `--config`. Mirrors `TrainConfig` dataclass; CLI flags override individual fields. |
| `02_sft_training.ipynb` | Smoke-test notebook (~20 examples × 1 epoch). Imports the script as a module, exercises every component, includes sanity-check cells (5 / 5b / 5c). |
| `01_preprocessing.py` | Stage 1–5 preprocessing (unchanged today). Produces the parquet that feeds the JSONL the trainer reads. |

### E.2 Choices made beyond the design doc

| Topic | Choice | Why |
|---|---|---|
| Tokenization path | Manual Harmony span concat (existing `encode_with_spans`), **not** `apply_chat_template` + state-machine mask | Channel boundaries are known by construction → no off-by-one risk; `fit_prefix_to_budget` (drop oldest turns until prefix+suffix fits) is already validated; respects channel-inverted Stage 5 contract noted in memory |
| Assistant framing tokens (`<\|channel\|>`, `<\|message\|>`, `<\|end\|>`, `<\|start\|>assistant`, `<\|return\|>`) | `channel_mask = 0` (masked from loss) | Base model already knows Harmony from pretraining; LoRA on attention projections doesn't need to relearn structure; keeps `loss_a`/`loss_f` as pure content losses |
| Customer ID column | Configurable via `TrainConfig.customer_id_field` (default `customer_id`, falls back to `cust_id`); hard error if neither present | Matches the dual naming already used in `01_preprocessing.py:453` |
| Per-device batch size | `1` with `gradient_accumulation_steps=8` (effective 8 / GPU) | Design doc says 2 but is FSDP-constraint dependent; 1 is the conservative smoke-test default and grad-accum reaches the target effective batch size. Raise to 2 once the cluster run confirms headroom |
| Save strategy | `save_strategy="steps"` only (every 30 steps), no separate per-epoch save | HF requires `save_strategy == eval_strategy` when `load_best_model_at_end=True`. With `save_total_limit=10` this still keeps the last 10 step-checkpoints; the best by `eval_loss` is reloaded at end |
| Per-channel eval logging | Accumulated inside `ChannelMeanLossTrainer.compute_loss` when `model.training == False`, averaged and emitted via overridden `evaluate()` so tensorboard sees `eval_loss_a` and `eval_loss_f` | HF's default `evaluation_loop` only knows about `eval_loss` |

### E.3 Verification baked into the notebook

`02_sft_training.ipynb` runs end-to-end on ≤20 examples. The critical
checks (run them all before launching the full job):

- **Cell 4** — decode `train_ds[0]` with specials visible; confirm one
  `<|channel|>analysis<|message|>...<|end|>` block and one
  `<|channel|>final<|message|>...<|return|>` block.
- **Cell 5** — `channel_mask` partitions every token into
  {masked, analysis, final}; assertion that
  `labels[i] == -100  ⇔  channel_mask[i] == 0`.
- **Cell 5b** — runs the per-example per-channel-mean loss formula on
  dummy logits across weight sweeps `(1,1) (0.5,0.5) (1,0) (0,1)
  (0.3,0.7)`. Asserts `total == w_a*loss_a + w_f*loss_f` and that
  `loss_a`/`loss_f` are invariant to the weights.
- **Cell 5c** — confirms no customer appears in both `train_raw` and
  `val_raw`.

### E.4 How to use the SFT training script

#### Single-GPU smoke test (no launcher)

```bash
python 02_sft_training.py --config 02_train_config.yaml \
    --max-samples 20 --epochs 1 --output-dir checkpoints_smoketest
```

CLI flags override the matching YAML field. Available overrides:
`--input`, `--model-dir`, `--output-dir`, `--epochs`,
`--analysis-weight`, `--final-weight`, `--max-samples`,
`--max-seq-len`, `--window-size`.

#### Multi-GPU production run

Use the team's existing accelerate config (FSDP, bf16 throughout, fp32
optimizer states — matches B.2). Per cluster policy this is the only
supported multi-GPU launcher; do NOT call `deepspeed` directly.

```bash
nohup accelerate launch --config_file <fsdp_config.yaml> \
    02_sft_training.py --config 02_train_config.yaml \
    > train.log 2>&1 & disown
```

For the 8×H100 / 2-faulty cluster from B.2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup accelerate launch \
    --config_file <fsdp_config.yaml> --num_processes 6 \
    02_sft_training.py --config 02_train_config.yaml \
    > train.log 2>&1 & disown
```

The script auto-detects the distributed launcher (`LOCAL_RANK` or
`ACCELERATE_USE_{FSDP,DEEPSPEED}` env vars) and skips `device_map="auto"`
so it doesn't fight FSDP wrapping.

#### Reproducibility

On startup the script writes the fully-resolved `TrainConfig` to
`<output_dir>/train_config.yaml`. On completion it writes:
- `<output_dir>/final_adapter/` — LoRA adapter + tokenizer
- `<output_dir>/training_log.json` — full `trainer.state.log_history`
- `<output_dir>/runs/` — tensorboard event files

Inspect with: `tensorboard --logdir <output_dir>/runs`. The interesting
scalars are `loss`, `loss_a`, `loss_f` (per-step) and `eval_loss`,
`eval_loss_a`, `eval_loss_f` (every `eval_steps=30`).

#### Notebook smoke test

Open `02_sft_training.ipynb`, fill in `model_dir` and `input` paths in
Cell 1, run top-to-bottom. Cells 4 / 5 / 5b / 5c must pass before Cell 6
is meaningful. Cell 6 runs one short epoch on ≤20 examples (uses
`save_strategy="epoch"` and does not load best — it's a smoke check, not
a real run).

### E.5 What's NOT in scope for the script today

- **Stage 5 prompt rework** (Part A.6). Lives in `01_preprocessing.py`,
  not in the trainer. Update there when the teacher API switches.
- **Best-of-N rejection sampling** (Part A.4). Deferred until the real
  teacher API is available.
- **DeepSpeed-ZeRO-3 A/B** (Part B.2 / D.3). Trainer already supports
  it via the accelerate config (no script changes needed); just point
  `--config_file` at the DeepSpeed accelerate config.
- **MoE LoRA targeting** (Part D.4). `lora_target_modules` is YAML-configurable
  but defaults to attention-only per B.3.

---

## Part D — Open Questions Deferred

1. **Best-of-N N value and similarity threshold.** Decide when real teacher
   API lands. Probably N=4-8 with LLM-as-judge.
2. **Channel weight tuning.** Start at (1.0, 1.0). Only adjust if
   per-channel val loss curves show one channel learning poorly.
3. **DeepSpeed vs FSDP A/B.** After v1 baseline is working.
4. **MoE LoRA targeting.** Adding LoRA to MoE down-projections could
   improve expressivity but is risky on GPT-OSS. Defer to v2.
5. **QAT for production deployment.** NVIDIA's MXFP4-via-QAT pipeline for
   deployment cost reduction. Out of scope for v1.
