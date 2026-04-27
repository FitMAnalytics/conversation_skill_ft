# Fine-Tuning Plan: Outbound Sales Agent (GPT-OSS-120B)

## 1. Objective

Fine-tune a GPT-OSS-120B base model on outbound sales call transcripts from top-performing agents (calls that ended in conversion) to produce a model that can conduct outbound sales conversations — leading the dialogue, identifying customer needs, handling objections, and guiding toward conversion.

This is **behavioral cloning via SFT** — teaching the model to talk like our best agents. It is NOT conversion prediction (separate project) and NOT RL-based strategy optimization (future layer). The SFT model serves as the behavioral backbone that later layers build on.

## 2. Scope and Limitations of SFT Alone

What SFT will accomplish:
- Model learns conversational patterns, tone, and pacing of high-performing agents
- Handles common objections with proven response patterns
- Produces fluent, natural sales dialogue

What SFT will NOT accomplish (requires later layers):
- **Strategic adaptation** to novel/unfamiliar customer objections outside the training distribution
- **Multi-turn planning** — SFT produces reactive, not proactive, behavior
- **Negative signal** — training only on winning calls means the model cannot distinguish between responses that caused the win vs. responses that happened to occur in a winning call
- **Leading the conversation** with a coherent multi-turn strategy (this is the RL/DPO gap)

## 3. Planned Architecture (Full Stack, Phased)

```
┌─────────────────────────────────────────────────┐
│  Layer 4 (Future): DPO / RL                     │
│  Strategic optimization — contrastive pairs of  │
│  good vs. bad agent moves at critical moments.  │
│  Addresses "no negative signal" and multi-turn  │
│  planning. DPO is cheaper than full RL.         │
│  Alternative: discrete action space + DQN/CQL.  │
├─────────────────────────────────────────────────┤
│  Layer 3: Compliance Filter (Parallel)          │
│  Post-generation verification layer.            │
│  NOT injected into the prompt — operates on     │
│  model output independently. Can be rule-based, │
│  classifier, or separate LLM call with          │
│  compliance rulebook. Flags/blocks/regenerates. │
│  Provides audit trail for regulated environment.│
├─────────────────────────────────────────────────┤
│  Layer 2 (Near-term): Product RAG               │
│  Retrieved at inference time, injected into     │
│  <|context|> block. Handles factual accuracy    │
│  for product details, campaigns, offers.        │
│  Separate vector store of product docs.         │
├─────────────────────────────────────────────────┤
│  Layer 1 (NOW): SFT on GPT-OSS-120B            │
│  Behavioral backbone. Learns how to talk like   │
│  a high-performing outbound sales agent from    │
│  winning call transcripts.                      │
└─────────────────────────────────────────────────┘
```

**Key architectural principle:** Layer 1 and 4 learn from data. Layer 2 grounds in facts. Layer 3 enforces constraints. Mixing these concerns into a single model makes each one worse.

## 4. Training Data Format

Each training example is **one agent turn** — the model learns to generate the agent's response given everything before it.

```
<|system|>You are an outbound sales agent for American Express.
Your goal is to identify customer needs and guide the conversation
toward [product/outcome]. Use information provided in the context
block when available to reference relevant offers, campaigns,
or product details. If no context is provided, rely on the
conversation history to guide your response.<|/system|>
<|context|><|/context|>
<|conversation|>
<|agent|>Hi, this is [name] from American Express. I'm reaching out because...
<|customer|>Yeah, I'm kind of busy right now, what's this about?
<|agent|>I completely understand, I'll be brief. Based on your company's spending profile...
<|customer|>We already have a corporate card program with Chase, why would we switch?
<|/conversation|>
<|agent|>[MODEL GENERATES THIS — loss computed only on this part]
```

### 4.1 Special Tokens

```
<|system|> <|/system|>       — System prompt boundaries
<|context|> <|/context|>     — RAG context block (placeholder for v1)
<|conversation|> <|/conversation|> — Conversation history boundaries
<|agent|>                    — Agent turn marker
<|customer|>                 — Customer turn marker
```

### 4.2 The `<|context|>` Block — RAG Placeholder Strategy

For v1, the context block is mostly empty but structurally present.

**Why include it now:** If the model only sees empty context during SFT, attention heads won't learn to attend to that position. When RAG is added later, the model will have weak priors on how to integrate retrieved content.

**v1 strategy:** Randomly populate 10-20% of training examples with basic metadata extracted from transcript metadata (no LLM call needed):

```
<|context|>
Product: Corporate Platinum Card
Campaign: Q2 2025 Spend Bonus
<|/context|>
```

The system prompt explicitly instructs the model that empty context is normal ("If no context is provided, rely on the conversation history"), establishing the behavioral contract for both training and inference.

### 4.3 Loss Masking

Compute loss **only on agent response tokens** after the final `<|agent|>` marker. Everything else (system prompt, context, conversation history, customer turns) is input context — masked from the loss.

## 5. Raw Data Format

Each transcript is a sequence of alternating turns in the format:

```
agent(u1): [agent utterance text]
customer(u2): [customer utterance text]
agent(u3): [agent utterance text]
customer(u4): [customer utterance text]
...
```

- `agent` and `customer` are speaker labels
- `uN` is the utterance index (1-indexed, sequential)
- The conversation is outbound, so `u1` is always the agent's opening
- Turns strictly alternate agent → customer → agent → ...
- Each transcript has an associated binary label: 1 = converted, 0 = did not convert
- **For v1 SFT, we only use transcripts where label = 1 (converted calls from top-performing agents)**

## 6. Transcript Preprocessing Pipeline

### 6.1 From Raw Transcript to Training Examples

One conversation with N agent turns produces N training examples (one per substantive agent turn).

Steps:

1. **Parse turns:** Regex-parse each line to extract speaker (`agent`/`customer`), utterance index, and utterance text. Since the raw format is strictly alternating, no diarization cleanup or same-speaker merging is needed.

2. **Filter trivial agent turns:** Skip agent backchannels ("mhm," "right," "I see") as prediction targets. Apply a minimum token length threshold (~15 tokens). These turns remain in conversation history as context, just not used as generation targets.

3. **Construct training examples:** For each substantive agent turn at position `i`, the input is: system prompt + context block + all turns before position `i`. The target is the agent's utterance at position `i`.

### 6.2 Truncation Strategy at Training Time

Each training example is packed as `prefix + target + EOS` and normalized to a fixed `MAX_SEQ_LEN` (right-padded with `-100` labels on pad tokens). Section-aware truncation handles over-budget examples:

- **Preserve the system prompt and `<|context|>` block intact.** They define the model's identity and factual grounding — truncating them trains the model on inputs that don't match inference-time prompts.
- **Preserve the target (next agent utterance) + EOS verbatim.** Keeps the loss boundary clean and always trains on a complete stopping signal.
- **Drop the oldest turns from inside `<|conversation|>...<|/conversation|>` first.** Recency dominates next-turn prediction, so the earliest turns lose the least signal per token removed.
- **Last-resort fallback:** if `system + context + target` alone already exceed `MAX_SEQ_LEN`, the head (system + context) is left-truncated. This should be essentially never triggered with a sensibly-chosen `MAX_SEQ_LEN`; the preprocessing notebook prints a counter for this case.
- **Pick `MAX_SEQ_LEN` from the data.** Compute the 95-99th percentile of actual `prefix + target + EOS` token lengths and set `MAX_SEQ_LEN` just above that. Going larger wastes compute on padding (attention is super-linear in sequence length); going smaller truncates real content.

Sequence packing (concatenating multiple short examples per row with per-segment masks) is an optional future optimization when length variance is high; v1 uses plain right-padding for simplicity.

### 6.3 System Prompt Consistency

Use a **single fixed system prompt** across all training examples. Do not customize per-conversation. Conversation-specific details belong in `<|context|>`, not the system prompt. This ensures that at inference time, one fixed system prompt works and all variability flows through the context block and conversation history.

## 7. What to Build First (v1 Scope)

Implemented as a four-notebook pipeline:

1. **`00_generate_fake_data.ipynb`** — synthetic outbound-sales transcripts in the raw `agent(uN): ... / customer(uN): ...` format described in §5, each with a binary `label`. Used while real transcripts are unavailable and as a smoke test for the preprocessing / training code paths.

2. **`01_preprocessing.ipynb`** — Raw transcripts → training examples. Parses the alternating-turn format, filters trivial agent turns (<15 tokens) from being used as generation targets, emits one example per substantive agent turn in the `<|system|> / <|context|> / <|conversation|> / <|agent|>` template with loss masking on everything before the final `<|agent|>`. Randomly populates `<|context|>` on 10-20% of examples with metadata-derived text so the model learns to attend to that slot for Layer 2.

3. **`02_sft_training.ipynb`** — QLoRA on GPT-OSS-120B. Includes:
   - **Token-length distribution check:** histograms + percentiles (p50/p75/p90/p95/p99/max) over `prefix + target + EOS` lengths to pick `MAX_SEQ_LEN` from the data (§6.2).
   - **Section-aware truncation:** preserves system prompt, `<|context|>` block, and target verbatim; drops oldest turns from inside `<|conversation|>…<|/conversation|>` first; counts how many examples hit conversation-level vs. head-level truncation.
   - 8 registered special tokens with `resize_token_embeddings`, loss masked to the agent-response span, LoRA adapter saved to `checkpoints/final_adapter/`.

4. **`03_inference.ipynb`** — Loads the base model in 4-bit + attaches the LoRA adapter. Exposes `generate_agent_response(system_prompt, context, conversation_history)` using the exact training-time template, with stop tokens on `<|customer|>`, `<|/conversation|>`, and EOS. Also provides an interactive chat cell and a **base-model-only** variant (`generate_base_response`) that wraps `model.generate` in `with model.disable_adapter():` for direct base-vs-fine-tuned A/B comparison without reloading weights.

**Evaluation (v1):** manual qualitative review of generated responses on held-out conversations — fluency, on-topic responses, appropriate tone, reasonable objection handling — plus side-by-side inspection against the base-model baseline to confirm the adapter is actually changing behavior in the intended direction.

## 8. Future Layers (Not In Scope for v1)

### 8.1 DPO (Layer 4, Phase 2)

Construct contrastive pairs: at critical conversation moments (objection raised, closing attempt, need discovery), pair the winning agent's response (chosen) with a weaker response (rejected). The rejected response can come from lower-performing agents on similar calls, or from the SFT model's own generations that a human reviewer rates as inferior.

DPO trains directly on preference pairs without needing a reward model, making it cheaper than full RL while still providing negative signal.

**Infrastructure reuse:** the multi-scale state-retrieval system described in `rag_architecture.md` is the mechanism for finding "similar states" across transcripts. Given a state from a winning call, retrieve the closest states from losing calls; the winning agent's response becomes `chosen` and the losing agent's response becomes `rejected`. This turns the retrieval system into a preference-pair-mining pipeline.

### 8.2 Product RAG (Layer 2, Phase 2)

Build a vector store of product fact sheets, campaign details, and benefit summaries. At inference time, retrieve top-k relevant docs based on recent conversation turns, populate the `<|context|>` block, and generate. The model already knows how to use this block from SFT.

### 8.3 Compliance Filter (Layer 3, Phase 2)

Post-generation parallel filter. Evaluates agent response against compliance rulebook before delivery. Operates independently of the generation model. Provides audit trail. Can be rule-based, classifier, or separate LLM call.
