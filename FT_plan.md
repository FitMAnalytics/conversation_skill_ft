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

### 6.2 Conversation History Windowing

For long conversations that exceed context limits:
- Include **first 2 turns** (the opening framing matters)
- Include **last K-2 turns** as the immediate context window (K ≈ 20 should cover most calls)
- This preserves both the call opening and recent conversational context

### 6.3 System Prompt Consistency

Use a **single fixed system prompt** across all training examples. Do not customize per-conversation. Conversation-specific details belong in `<|context|>`, not the system prompt. This ensures that at inference time, one fixed system prompt works and all variability flows through the context block and conversation history.

## 7. What to Build First (v1 Scope)

1. **Preprocessing script:** Raw transcripts → cleaned, turn-segmented, formatted training examples as described above.
2. **Data validation:** Verify token distributions, check for truncation at max context length, confirm loss masking is correct.
3. **SFT training run:** QLoRA on GPT-OSS-120B with the formatted data.
4. **Evaluation:** Manual qualitative review of generated responses on held-out conversations. Check for: fluency, on-topic responses, appropriate tone, reasonable objection handling.

## 8. Future Layers (Not In Scope for v1)

### 8.1 DPO (Layer 4, Phase 2)

Construct contrastive pairs: at critical conversation moments (objection raised, closing attempt, need discovery), pair the winning agent's response (chosen) with a weaker response (rejected). The rejected response can come from lower-performing agents on similar calls, or from the SFT model's own generations that a human reviewer rates as inferior.

DPO trains directly on preference pairs without needing a reward model, making it cheaper than full RL while still providing negative signal.

### 8.2 Product RAG (Layer 2, Phase 2)

Build a vector store of product fact sheets, campaign details, and benefit summaries. At inference time, retrieve top-k relevant docs based on recent conversation turns, populate the `<|context|>` block, and generate. The model already knows how to use this block from SFT.

### 8.3 Compliance Filter (Layer 3, Phase 2)

Post-generation parallel filter. Evaluates agent response against compliance rulebook before delivery. Operates independently of the generation model. Provides audit trail. Can be rule-based, classifier, or separate LLM call.
