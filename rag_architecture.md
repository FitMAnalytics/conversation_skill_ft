# RAG-based Conversational State Retrieval System — Architecture

## 1. Objective

Build a retrieval system that finds historically similar conversational states across outbound sales call transcripts. "Similar" means: the ideal agent response would be similar in both situations. This system serves two purposes:

1. **Standalone value**: Retrieve what high-performing agents said in similar situations as real-time agent guidance.
2. **DPO pipeline component**: Construct preference pairs by comparing agent responses (and their conversion deltas) across similar states.

---

## 2. Transcript Format & Turn Segmentation

### Raw format
```
amex agent (u1): ... \n customer (u2): ... \n amex agent (u3): ... \n customer (u4): ...
```

### Segmentation rule
Divide each transcript into **states**, where each state boundary is defined as **immediately after a customer utterance**. This is the decision point where the agent must choose what to say next.

For a transcript with utterances u1 (agent), u2 (customer), u3 (agent), u4 (customer), ..., the states are:
- State after u2: agent must decide u3
- State after u4: agent must decide u5
- etc.

Each state captures "what has happened so far up to this customer turn."

---

## 3. Multi-Scale Embedding Strategy

For each state, construct multiple look-back windows to capture both immediate context and broader conversational dynamics:

| Window ID | Content | Captures |
|-----------|---------|----------|
| `w1` | Last customer utterance only | Immediate customer intent/request |
| `w2` | Last 2 turns (1 agent + 1 customer) | Recent exchange dynamics |
| `w3` | Last 3 turns | Short-term context |
| `w5` | Last 5 turns | Medium-term topic flow |
| `w10` | Last 10 turns | Broader conversation arc |
| `w_summary` | GPT-4.1 structured summary of full context up to this point | Global state |

### Structured summary prompt (for `w_summary`)
Use GPT-4.1 to produce a structured state description before embedding:

```
Summarize the following sales call context in this exact format:
- Conversation phase: (opening / discovery / pitch / objection_handling / closing / follow_up)
- Treatment: (e.g., AP automation, employee cards, cross-sell, underwriting, etc.)
- Customer intent: (interested / skeptical / comparing_alternatives / ready_to_commit / disengaged / unclear)
- Key objection or topic: (one sentence)
- Sentiment trajectory: (warming / cooling / neutral / volatile)
- Customer type signals: (any inferred seniority, role, company size indicators)
```

This compensates for ada-2's general-purpose nature by translating conversational dynamics into descriptive text that ada-2 handles well.

### Design note on overlapping windows
Windows w1 ⊂ w2 ⊂ w3 ⊂ w5 ⊂ w10 — they overlap. This is intentional for now (simpler implementation). The Optuna weight optimization should learn to downweight redundant signals. If retrieval quality is poor, consider switching to non-overlapping segments (e.g., last utterance, last 2-3 turns, turns 4-10, full summary).

---

## 4. Embedding Model

### Primary: `ada-2` (text-embedding-ada-002)
- Available via `safechain.lcel.model('ada-2')` 
- 1536-dimensional embeddings
- 8191 token context window
- General-purpose; not optimized for dialogue similarity

### Calling convention
```python
from safechain.lcel import model
embeddings = model('ada-2')
vector = embeddings.embed_query(text)  # returns list of 1536 floats
```

### Known limitations
- May latch onto surface lexical patterns rather than conversational dynamics
- The structured summary path (`w_summary`) is the primary mitigation
- If retrieval quality is insufficient after optimization, fallback plan is to pre-train a domain-specific BERT on transcripts (see Section 9)

### Other available models (not yet accessible via safechain, may require direct API calls)
- `text-embedding-3-large` (Azure OpenAI): 3072-dim, Matryoshka support, superior quality — available in Amex Azure but not registered in safechain config
- `bge-large-en` (Launchpad): 512-dim, open-source — accessible at `/genai/google/v1/launchpad/models/bge-large-en/embeddings`
- `bge-reranker-large` (Launchpad): cross-encoder reranker — could be used as a second-stage reranker if retrieval precision needs improvement

---

## 5. Composite Similarity Metric

Given a query state Q and a candidate state C, the similarity score is a weighted sum of per-window cosine similarities:

```
sim(Q, C) = α1 * cos(Q_w1, C_w1) 
          + α2 * cos(Q_w2, C_w2) 
          + α3 * cos(Q_w3, C_w3) 
          + α5 * cos(Q_w5, C_w5) 
          + α10 * cos(Q_w10, C_w10) 
          + αs * cos(Q_wsummary, C_wsummary)
```

Where `α1 + α2 + α3 + α5 + α10 + αs = 1` (normalized).

### Implementation approach
- Store each window's embedding as a separate vector in a FAISS index (one index per window scale)
- At query time, compute cosine similarity per window, then combine with weights
- Alternatively, concatenate weighted embeddings into a single vector for single-index retrieval (but this loses the ability to tune weights without re-indexing)

**Recommended approach**: Separate indices per window, combine scores at query time. This allows weight tuning without re-embedding.

---

## 6. Vector Database

### Choice: Pure numpy (brute-force cosine similarity)

**Scale**: ~400-500 transcripts × 10-200 utterances × ~50% are customer turns = roughly 10K-50K states, each with 6 embedding windows.

At this scale, brute-force `np.dot` on L2-normalized vectors is near-instant and requires no additional dependencies.

### Index structure
```python
import numpy as np

EMBEDDING_DIM = 1536  # ada-2
WINDOW_NAMES = ['w1', 'w2', 'w3', 'w5', 'w10', 'w_summary']

# Store as dict of numpy arrays, one per window scale
# Each array has shape (n_states, 1536)
embedding_store = {name: [] for name in WINDOW_NAMES}

def add_state(window_embeddings: dict):
    """Append one state's embeddings. Call build_index() after all states added."""
    for name in WINDOW_NAMES:
        vec = np.array(window_embeddings[name], dtype='float32')
        vec = vec / (np.linalg.norm(vec) + 1e-8)  # L2 normalize
        embedding_store[name].append(vec)

def build_index():
    """Stack lists into matrices for fast retrieval."""
    for name in WINDOW_NAMES:
        embedding_store[name] = np.vstack(embedding_store[name])  # (n_states, 1536)

def retrieve_top_k(query_embeddings: dict, weights: dict, k: int = 5):
    """
    query_embeddings: dict of window_name -> 1D vector (1536,)
    weights: dict of window_name -> float (should sum to 1)
    Returns: indices of top-k most similar states
    """
    composite_scores = np.zeros(embedding_store['w1'].shape[0])
    for name in WINDOW_NAMES:
        q = np.array(query_embeddings[name], dtype='float32')
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = embedding_store[name] @ q  # (n_states,) cosine similarities
        composite_scores += weights[name] * sims
    return np.argsort(composite_scores)[::-1][:k]
```

### Metadata storage
Use a parallel pandas DataFrame where row index aligns with the numpy array index:
- `transcript_id`
- `state_index` (which customer turn within the transcript)
- `agent_response` (the agent's next utterance — the response we care about for DPO)
- `treatment` (topic label)
- `agent_id` (for filtering by performance tier)
- Conversion outcome

### Persistence
```python
# Save
np.savez('embeddings.npz', **embedding_store)
metadata_df.to_parquet('metadata.parquet')

# Load
loaded = np.load('embeddings.npz')
embedding_store = {name: loaded[name] for name in WINDOW_NAMES}
metadata_df = pd.read_parquet('metadata.parquet')
```

---

## 7. Weight Optimization via Optuna

### Objective
Find the weight vector `[α1, α2, α3, α5, α10, αs]` that maximizes retrieval quality as judged by an LLM.

### Procedure
1. Sample N evaluation states (target: 50-100, stratified by treatment and conversation length)
2. For each Optuna trial, propose a weight vector
3. For each evaluation state, retrieve top-5 matches using the proposed weights
4. Use GPT-4.1 to rate each (query_state, retrieved_state) pair on a 1-5 scale
5. Objective = mean LLM similarity rating across all evaluations

### LLM judge prompt (critical design choice)
Frame around downstream utility, not abstract similarity:

```
You are evaluating whether two sales call states are similar enough that 
the same agent response strategy would be appropriate for both.

State A:
{state_a_context}

State B:
{state_b_context}

Rate the similarity on a 1-5 scale:
1 - Completely different situations, same response would be inappropriate
2 - Some surface similarity but different underlying dynamics
3 - Moderately similar, same general approach might work but details differ
4 - Very similar, the same response strategy would likely be effective
5 - Nearly identical situations

Respond with only the number.
```

### Optuna configuration
```python
import optuna

def objective(trial):
    # Propose weights (Dirichlet-like via softmax of unbounded params)
    raw = [trial.suggest_float(f'w_{name}', 0, 10) for name in window_names]
    weights = [r / sum(raw) for r in raw]
    
    # Retrieve and evaluate
    scores = []
    for state in eval_states:
        top5 = retrieve_top_k(state, weights, k=5)
        for match in top5:
            score = llm_judge(state, match)
            scores.append(score)
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Cost estimate
- 50-100 eval states × 5 matches × 50 Optuna trials = 12,500-25,000 LLM judge calls
- At GPT-4.1 with short prompts, this is manageable but not trivial
- Consider caching: if the same (query, candidate) pair appears across trials (because only the ranking changed), reuse the rating

### Validation after optimization
- Take optimized weights, retrieve top-5 for 30+ new states (not in eval set)
- Manual inspection + LLM re-evaluation on held-out set
- **Negative validation**: find cases where retrieval says "very similar" but outcomes diverged significantly — these reveal missing dimensions

---

## 8. Retrieval Pipeline (end-to-end)

```
Input: a customer utterance in an ongoing call

1. Segment: extract windows (w1, w2, w3, w5, w10) from conversation so far
2. Summarize: call GPT-4.1 to produce structured summary (w_summary)
3. Embed: call ada-2 on each window text → 6 vectors
4. Search: query each FAISS index, get top-K candidates per window
5. Score: compute weighted composite similarity for all unique candidates
6. Rank: return top-5 by composite score
7. (Optional) Rerank: if bge-reranker-large becomes accessible, rerank top-20 → top-5

Output: top-5 similar historical states + their agent responses + conversion outcomes
```

---

## 9. Fallback: Domain-Specific BERT Pre-training

If ada-2 retrieval quality is insufficient (diagnosed by low LLM-judge scores even after weight optimization):

1. Start from `bert-base-uncased` (or a dialogue-pretrained variant)
2. MLM pre-train on the full transcript corpus to learn domain vocabulary
3. Fine-tune as a bi-encoder with contrastive loss on state pairs
4. Training pairs: reuse the LLM-judged similarity data from the Optuna evaluation pipeline
   - Pairs rated 4-5 → positive pairs
   - Pairs rated 1-2 → negative pairs (hard negatives)

This way the evaluation infrastructure built for ada-2 becomes training data for the BERT — nothing is wasted.

---

## 10. Open Questions / TODO

- [ ] Confirm whether `text-embedding-3-large` or `bge-large-en` can be called directly (bypassing safechain)
- [ ] Determine if bge-reranker-large is accessible via direct API call for two-stage retrieval
- [ ] Decide on non-overlapping vs overlapping windows based on initial retrieval quality
- [ ] Define the evaluation state sampling strategy (stratification criteria)
- [ ] Determine how to handle states early in calls where w10 or w5 may not exist (pad? skip? use only available windows?)
- [ ] Build conversion probability predictor (separate component) to enable ΔP-based preference pair construction
