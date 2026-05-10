# Stage 5 Design — Teacher CoT + Response Generation

This doc captures the design decisions for Stage 5 of the preprocessing pipeline:
how the teacher (GPT-OSS-120B) generates the chain-of-thought (CoT) and final
response that become training targets for the student's analysis and final
channels.

## Status (current decision, supersedes older sections below)

**v1 uses rationalization-CoT with an inverted channel mapping.** Distinct
from both Framing A (teacher reasons fresh, generates its own response) and
the original Framing B (rationalization with CoT in analysis channel and
response echoed in final channel) discussed below.

The contract:

- **Stage 5 prompt:** teacher sees the full context (customer/product profile,
  prior-call summary, current-call summary, objection, transcript) AND the
  polished agent response that Stage 2 produced. The teacher is told to
  reconstruct the agent's *forward* reasoning — the chain of thought a
  skilled agent would have run silently *before* speaking, written in first
  person, present tense, ending exactly at the given response. The teacher is
  explicitly told NOT to propose alternative moves even if it thinks of a
  better one — the agent already chose; the job is to write the reasoning
  that leads there.
- **Channel mapping (note the inversion):**
  - **Final channel** → the CoT (`teacher_cot`). This is the student's
    `analysis_channel` SFT target.
  - **Analysis channel** → OSS's free-form planning (`teacher_thinking`).
    Unconstrained — let the model draft, weigh framings, check that the CoT
    lands at the response, whatever. Stored as diagnostic only, not a
    training target.
- **Training pair stored:** `(teacher_cot, polished_response)`. The student's
  `analysis_channel` target is the teacher's rationalization CoT; the
  `final_channel` target is `polished_response`.
- **Coherence by construction:** because the teacher was shown
  `polished_response` and asked to write a CoT that leads to it, the
  (CoT, polished_response) pair is coherent — no divergence filter needed.
  Stage 5b becomes a CoT-quality judge (is the CoT first-person, forward-
  looking, grounded in inputs, landing naturally at the response?) rather
  than a divergence filter.

Why this and not the alternatives:

- **vs. pure Framing A (teacher reasons fresh, no `polished_response`):**
  rejected because the teacher's CoT might not lead to `polished_response`
  on individual rows, making the (CoT, polished_response) pair incoherent.
  Even with a 5b divergence filter, we'd lose the most skilled examples
  systematically (the ones where the agent used hidden context the teacher
  doesn't have).
- **vs. original Framing B (CoT in analysis channel, echo response in final):**
  the inverted mapping puts the *training-target content* in the final
  channel where the model can use the analysis channel as scratch space to
  plan it. With the original mapping, the analysis channel had to be the
  training target while the final channel was a near-duplicate of the input,
  wasting that channel's natural use as the model's working scratch.
- **vs. teacher-generates-its-own-response:** rejected because we want the
  student to be aligned with how the human agent actually handled
  objections, not with a synthesized teacher response.

Concrete differences from the original Framing B (sections below):

- No `[grounded: ...]` tag, no `cot_grounded` column.
- CoT lives in the **final** channel (not the analysis channel).
- The final channel is NOT an echo of `polished_response`; it's the CoT.
- The CoT is in *first-person forward-looking* form ("the customer is pushing
  back on price... given their industry...") rather than third-person
  retrospective explanation ("this response makes sense because...").
- 5b is a CoT-quality judge, not a CoT/response consistency filter.

The rest of this doc — the original Framing B decisions and the prompt that
went with them — is retained as historical context for the alternatives
considered. **Do not implement from the sections below as-is.**

---

## Decisions made

### 1. CoT format: free-form natural language prose

**Decision:** Free-form natural prose inside the analysis channel. No JSON, no
bullets, no headers, no enumerated steps. The only "structure" is the Harmony
channel delimiter (`analysis` vs `final`).

**Rationale:**
- Forcing JSON/structured CoT is documented to degrade reasoning quality by
  10–15%, primarily because schema constraints push the model toward
  zero-shot answering rather than genuine step-by-step inference.
- iSELF-DISCOVER (2025) directly compared dynamic JSON CoT vs free-form CoT
  and found a consistent advantage for unstructured reasoning across
  benchmarks.
- DeepSeek-R1 uses only ultra-lightweight `<think>...</think>` /
  `<answer>...</answer>` tags with free-form prose inside; the structure of
  R1's reasoning emerged from RL training, not from imposed templates.
- All canonical GPT-OSS fine-tuning examples (OpenAI cookbook, AWS, VESSL,
  HuggingFace Multilingual-Thinking) use free-form prose in the `thinking`
  field.
- The "separate thinking from formatting" pattern is the standard production
  approach: reason freely first, structure later via constrained decoding if
  needed.

### 2. Reasoning style: convergent and deductive (not exploratory/branchy)

**Decision:** Encourage GPT-OSS's natural convergent-deductive style. Resist
R1-style "wait, let me reconsider" revisionism. Tell the teacher not to
second-guess in circles.

**Rationale:**
- Recent (Apr 2026) study on long-CoT SFT generalization (arxiv 2604.01702)
  found that SFT on R1 trajectories produces lower training loss but
  significantly worse generalization than SFT on GPT-OSS-120B trajectories.
- R1 trajectories contain ~74% "Propose" steps (branchy exploration);
  GPT-OSS trajectories favor deep continuous deductive chains.
- Students inherit the teacher's reasoning shape during SFT, including
  inefficient exploration patterns. Filtering branchy R1 trajectories
  improves generalization by ~3.6% on average.
- Since we're using GPT-OSS-120B as both teacher and student, this aligns
  naturally — the teacher's convergent style transfers cleanly to the
  same-architecture student.

### 3. CoT length: self-modulating, not prescribed

**Decision:** Do not prescribe a length target. Tell the teacher to reason
"as much as the case warrants and no more," with explicit examples of when
more is appropriate (complex multi-signal cases) and when less is (simple
objections, well-understood customer). Add anti-padding instructions: don't
restate inputs, don't enumerate mechanically, don't second-guess in circles.

**Rationale:**
- R1 paper's "longer = better" finding is from RL with verifiable rewards,
  not from prompting. It doesn't transfer to prompted teacher generation.
- Thoughtology paper found R1 has a problem-specific optimal length beyond
  which performance declines, and R1 cannot self-modulate length.
- Sales objection handling is not math: past a certain point, more reasoning
  is elaboration, not new inference.
- Pushing for longer CoT pushes toward R1-style branchy exploration, which
  is the failure mode we want to avoid (see Decision 2).
- Inference latency matters in production. Every analysis token is generated
  before the final response begins.
- Length-modulation (long for complex, short for simple) is itself a
  desirable property for the student to inherit. R1 lacks this; we want it.

**Diagnostic during inspection:** Look at CoT length distribution across
simple vs complex cases. If lengths are flat across complexity, prompt isn't
inducing modulation. If they correlate with complexity, prompt is working.

### 4. Framing: rationalization (Framing B), not pure derivation (Framing A)

**Decision:** Show the teacher the polished agent response and ask it to
generate a CoT that explains why the response is a good move given the
visible context. Final channel echoes the polished response. This is
rationalization-style, not pure-derivation-style.

**Rationale and tradeoffs:**

The pure-derivation framing (teacher reasons from inputs only, generates
both CoT and response, then 5b filters for consistency with polished agent
response) was the initial design. We rejected it for the current data
regime due to a structural information asymmetry:

- Real human agents have access to context that the LLM does not:
  pre-call research, account notes, relationship feel, customer tone of
  voice, manager strategy guidance, prior interactions not captured in the
  summary.
- The teacher reasoning from a strict subset of the human's inputs cannot
  reliably land on the human's response, even with a great prompt.
- This makes 5b consistency filtering systematically biased: it preserves
  cases where the answer was inferable from visible context and drops
  cases where the human used hidden context — exactly the cases where
  the human did something subtle and skilled.
- With ~400–500 transcripts (a few thousand objection rows after
  expansion), we can't afford to filter aggressively.

Costs of rationalization framing we accept:
- Some confabulation risk: teacher invents reasoning to justify a response
  that was actually driven by hidden context. The student learns to
  produce confident post-hoc justification.
- Mitigated by Decision 5 (grounding constraint): teacher is told to
  reason only from visible inputs and to be honest when the response
  appears to draw on hidden context.

Why this is acceptable for Layer 1 SFT specifically:
- SFT goal is to get the student into the right neighborhood of behavior,
  producing coherent analysis-and-response pairs on-distribution.
- Causal reasoning purity is more important for later layers (DPO, RL),
  where genuine optimization pressure on response quality matters.
- The student at inference also won't have hidden context, so training it
  to produce "plausible reasoning given partial context" matches its
  deployment task.

**Framing A is the worst option** (compared to either pure A or pure B):
generating teacher CoT from inputs only, then pairing it with the polished
agent response regardless of consistency. This trains the student on
incoherent (CoT, response) pairs and actively teaches dissociation between
analysis and output. Avoid.

### 5. Grounding constraint inside the rationalization

**Decision:** Instruct the teacher to reason only from visible inputs and to
explicitly flag when the agent's response appears to draw on context not
present in the inputs, rather than inventing justification.

**Rationale:**
- This converts the rationalization framing's main pathology (confabulation)
  into useful, honest training signal.
- When the response IS derivable from visible context, the teacher produces
  clean grounded CoT.
- When it ISN'T, the teacher produces a CoT that openly says so. This
  teaches the student to ground reasoning in available inputs and to be
  honest about what those inputs determine.
- The student learning honest partial-information reasoning is strictly
  better than learning confident confabulation.

### 6. Optional groundedness tag for diagnostics

**Decision:** Have the teacher end its analysis with a tag like
`[grounded: high]`, `[grounded: medium]`, or `[grounded: low — likely uses
unstated context]`.

**Rationale:**
- Turns the information-asymmetry concern from a hypothesis into a
  measurable signal.
- Aggregate distribution tells us whether the visible input fields are
  sufficient (mostly "high") or whether critical context is missing
  (significant "low" fraction).
- Per-scenario breakdown can reveal which scenario types have the largest
  asymmetry, informing where to invest in pulling additional context fields
  (account notes, CRM data) for v2.
- Not used for filtering at this stage — purely diagnostic. Can revisit
  whether to use for downweighting or filtering after seeing the
  distribution.

## Stage 5b implications

The semantic of Stage 5b changes under Framing B.

- **Was:** consistency check between teacher-generated final response and
  Stage 2 polished response, used as filter.
- **Now:** CoT quality check. Final response matches Stage 2 by construction.
  5b becomes an LLM-as-judge pass on the analysis with a simple rubric:
  - Does the analysis reference specific elements of the customer context?
  - Does it engage with the substance of the objection?
  - Does it provide a coherent rationale for the chosen response?
  - Is it honest about groundedness (if it claims grounded:high, is it
    actually grounded; if it claims grounded:low, is the analysis still
    coherent given the partial-information framing)?

  Drop examples that fail on multiple counts.

- The grounded-tag distribution can be reviewed in aggregate as part of 5b,
  separately from per-example filtering.

## Final prompt (Framing B + grounding constraint)

```python
STAGE5_SYSTEM = f"""You are an expert outbound sales agent at American Express on the TSE
(Tele Strategic Expansion) channel.

{TSE_BACKGROUND}

You will be given:
  - the customer's profile and product/campaign context (industry, business size,
    spend patterns, card portfolio, current campaign details)
  - a summary of prior calls with this customer, if any
  - a summary of how the current call has gone so far
  - the current objection summary
  - the full transcript up to the objection point
  - the response the agent actually gave

Your task is to reason about WHY this response is a good move given the situation.
A skilled agent's response is rarely arbitrary; it reflects a chain of judgment
about the customer, the objection, and what will move the conversation forward.
Your job is to reconstruct that judgment in the analysis channel, then echo the
agent's response in the final channel.

In the analysis channel, reason in flowing natural prose — not bullets, not
headers, not JSON, not enumerated steps. Think the way an experienced agent
thinks silently between hearing the objection and choosing what to say. Ground
your reasoning in the specific inputs in front of you: the customer's industry
and what typically matters in that industry, what you know from prior calls,
what has already happened in this call, and what the current objection actually
signals beneath its surface wording. Walk through what response options are
available, weigh them against each other, and arrive at the response the agent
gave.

Important: reason ONLY from the inputs you can see. Real agents often draw on
context that isn't in your inputs — relationship history not captured in the
prior-calls summary, pre-call research, account notes, the customer's tone of
voice. If the agent's response appears to draw on information beyond what's
visible to you, say so honestly in your reasoning rather than inventing context
to justify it. For example: "Given the visible context, this response makes
sense as a way to acknowledge the pricing concern before pivoting to value;
the specific framing about the customer's seasonal cash flow likely also reflects
relationship knowledge from prior interactions that isn't fully captured in the
summary." This honesty is more useful than fabricated justification.

Reason as much as the case warrants and no more. Simple cases warrant brief
reasoning; complex cases warrant more. Do not pad. Do not restate the inputs.
Do not enumerate steps mechanically. Do not second-guess yourself in circles;
once you've weighed the options and chosen, move on.

End your analysis with a single-line tag indicating how well the response is
grounded in the visible inputs:
  [grounded: high]   — response fully derivable from visible context
  [grounded: medium] — response mostly derivable; some elements suggest unstated context
  [grounded: low]    — response likely depends on context not visible in the inputs

In the final channel, output the agent's response exactly as given. The final
channel is not where you reason or rephrase — it's where you produce the target
response that your analysis just explained."""

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

THE AGENT'S RESPONSE:
{polished_agent_response}

In the analysis channel, reason about why this response is a good move given
the situation, working only from the visible inputs and being honest about any
points where the response appears to draw on information beyond what you can see.
End your analysis with a [grounded: high|medium|low] tag. In the final channel,
output the agent's response exactly as given.
"""
```

## Implementation checklist for code changes

- [ ] Update `STAGE5_SYSTEM` and `STAGE5_USER_TMPL` constants in `preprocess_lib.py`
      to the versions above.
- [ ] Add `polished_agent_response` to the Stage 5 input contract. Source: Stage 2
      output.
- [ ] Update Stage 5 call site to pass `polished_agent_response` into the user
      template.
- [ ] Update the parsing logic on Stage 5 output to:
      - Extract analysis channel content
      - Extract the `[grounded: ...]` tag from the end of the analysis
      - Strip the tag from the analysis text before storing (or keep it in,
        depending on whether you want the student to learn to emit it too —
        leaning toward strip for v1)
      - Extract final channel content (should match polished_agent_response;
        verify and log mismatches)
- [ ] Update the per-example output schema to include the grounded tag as a
      separate column (e.g., `cot_grounded: str` in {high, medium, low}).
- [ ] Update Stage 5b:
      - Remove the consistency-check-as-filter logic (no longer applicable)
      - Add LLM-as-judge CoT quality check with the rubric in the doc
      - Add aggregate grounded-tag distribution reporting
- [ ] Update `preprocess_notebook.ipynb` example flow to reflect new Stage 5
      input shape.
- [ ] Update `preprocess_batch.py` checkpointing schema to include the grounded
      tag column.

## Open questions for later

- Whether to use the grounded tag for downweighting `[grounded: low]` examples
  during SFT, or just keep it diagnostic. Decide after seeing the distribution.
- Whether to strip or keep the `[grounded: ...]` tag in the student's training
  target for the analysis channel. v1 leaning: strip. Could revisit if we want
  the student to emit calibrated groundedness at inference.
- Whether to add a second LLM-as-judge pass that re-derives a grounded tag
  independently and checks the teacher's self-reported tag against it.
  Probably overkill for v1.
