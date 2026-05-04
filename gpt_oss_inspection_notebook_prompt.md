# GPT-OSS Output Format Inspection Notebook — Generation Prompt

## Purpose

This document contains the full prompt to give to Claude (or another capable code-generating LLM) to produce a Jupyter notebook that loads GPT-OSS-120B from a local directory and inspects its raw output format in detail.

The goal is to understand the exact tokenization and channel structure of the model's responses so that SFT training data can be constructed to match the format precisely — including reasoning traces, not just final answers. Getting this wrong risks reasoning collapse during fine-tuning.

---

## Prompt to paste

I need a Jupyter notebook (`.ipynb`) that loads GPT-OSS-120B from a local directory and inspects its raw output format in detail. The purpose is to understand the exact tokenization and channel structure of the model's responses so I can construct SFT training data that matches the format precisely — including reasoning traces, not just final answers. Getting this wrong risks reasoning collapse during fine-tuning, so I need to *see* every special token, every channel boundary, and every formatting artifact the model produces natively.

### Hard constraints

- **transformers package only** for model/tokenizer loading. Do NOT use `openai-harmony`, `gpt-oss`, or any OpenAI-published helper package — they are not installable in my environment. Use `AutoModelForCausalLM` and `AutoTokenizer`.
- **torch == 2.6.0**. Don't assume torch 2.7+ APIs.
- **No DeepSpeed for loading** in this notebook — keep it simple. Single-node multi-GPU via `device_map="auto"` is fine. If the model is too large for the available GPUs in bf16, use `device_map="auto"` with `torch_dtype=torch.bfloat16` and let HF Accelerate shard it. Note: model weights are stored in **MXFP4** but loading via transformers will materialize compute in bf16 — that's expected and fine for inspection purposes.
- **Local weights**: model lives at a path like `/path/to/gpt-oss-120b/` (parameterize this as a variable at the top of the notebook). No HuggingFace Hub downloads.
- **No external network calls** anywhere in the notebook.
- Environment is an Amex GCP project — keep dependencies minimal and standard.

### What the notebook needs to do

#### Cell 1: Setup and config
- Imports: `torch`, `transformers` (AutoModelForCausalLM, AutoTokenizer, GenerationConfig), `json`, `pprint`.
- Print `torch.__version__`, `transformers.__version__`, `torch.cuda.device_count()`, and per-GPU memory.
- Define `MODEL_PATH` as a variable at the top.

#### Cell 2: Load tokenizer and dump every special token
- Load tokenizer with `AutoTokenizer.from_pretrained(MODEL_PATH)`.
- Print `tokenizer.special_tokens_map`, `tokenizer.additional_special_tokens`, and the full list of added tokens with their IDs (`tokenizer.get_added_vocab()` filtered to special tokens, sorted by ID).
- Specifically look for and print the IDs of Harmony-format tokens: `<|start|>`, `<|end|>`, `<|message|>`, `<|channel|>`, `<|return|>`, `<|call|>`, `<|constrain|>`, and any role tokens (`system`, `developer`, `user`, `assistant`, `tool`). If a name differs in the actual tokenizer, find the closest match and print what's there.
- Print `tokenizer.chat_template` (the raw Jinja template) — this is the ground truth for how messages are formatted. Pretty-print it.

#### Cell 3: Load model
- `AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)`.
- Set `model.eval()`.
- Print `model.config` (or relevant fields: `model_type`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`, `vocab_size`, anything MoE-related like `num_local_experts` / `num_experts_per_tok`, and any `sliding_window` / `rope_scaling` fields).
- Print model device map (`model.hf_device_map` if present).

#### Cell 4: Define inspection helpers
Write three small helper functions:

1. `apply_template_and_show(messages, reasoning_effort=None)` — runs `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, ...)` and returns the rendered string. If `reasoning_effort` is supported by the template, pass it through. Print the rendered prompt verbatim inside a clearly delimited block so I can see every special token.

2. `tokenize_and_show(text)` — tokenizes and returns a list of `(token_id, repr(token_str))` tuples, printed one per line for the first ~200 tokens and last ~50 tokens, so I can see exactly how special tokens are encoded.

3. `generate_and_dissect(messages, max_new_tokens=2048, reasoning_effort="medium", temperature=1.0, top_p=1.0)`:
   - Apply template, tokenize, generate with `model.generate(...)`.
   - Decode the **generated portion only** (slice off input length) with `skip_special_tokens=False` — this is critical, I need to see the raw special tokens.
   - Also produce a `skip_special_tokens=True` version for readability comparison.
   - Return a dict with: raw decoded string (specials visible), clean decoded string, list of generated token IDs, list of `(id, repr(piece))` for the first 100 generated tokens, and a parsed channel breakdown (see next bullet).
   - **Channel parser**: write a regex- or scan-based parser that splits the raw output by `<|channel|>...<|message|>...<|end|>` (or whatever the actual delimiters turn out to be — discover this from cell 2). Return a list of `{"channel": "analysis"|"commentary"|"final"|..., "content": str}` segments. Print this list.

#### Cell 5: Run a controlled experiment with my actual use case
Use a realistic prompt that mirrors my Amex sales-call use case. Provide it as a `messages` list:

```python
SYSTEM_PROMPT = """You are an expert sales analyst evaluating outbound calls. 
Given a call transcript, predict whether the call will result in a successful 
conversion (customer accepting the offer). Respond with your reasoning and 
then a final answer of either CONVERT or NO_CONVERT."""

CONTEXT_TRANSCRIPT = """[Agent]: Hi, this is Sarah from American Express. Am I 
speaking with Mr. Johnson?
[Customer]: Yes, this is he. What is this about?
[Agent]: I'm calling about a pre-approved offer for our Platinum card with a 
$200 statement credit after your first purchase...
[Customer]: I already have a Platinum card. I've had it for years.
[Agent]: Oh, I see. Well, we have an upgrade path that could give you 
additional benefits...
[Customer]: I'm not interested in changing anything right now. I'm happy with 
what I have.
[Agent]: I understand. Could I at least send you some information by email?
[Customer]: Sure, fine. Send the email. I have to go now.
[Agent]: Thank you, have a great day."""

USER_PROMPT = f"Analyze the following call transcript and predict conversion outcome.\n\n{CONTEXT_TRANSCRIPT}"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]
```

Run `generate_and_dissect` on this at three reasoning effort levels: `"low"`, `"medium"`, `"high"`. For each:
- Print the rendered prompt (specials visible).
- Print the raw output (specials visible).
- Print the parsed channel breakdown.
- Print a count of tokens per channel.
- Print final-channel-only content (clean).

#### Cell 6: Comparison table
Print a small summary table (just `print` formatted lines, no pandas needed) showing for each effort level: total generated tokens, analysis-channel tokens, final-channel tokens, ratio, and final answer extracted.

#### Cell 7: Notes for SFT data construction
End the notebook with a markdown cell that I will fill in myself — leave it as a template with these section headers:
- `## Special tokens to preserve verbatim in SFT targets`
- `## Channel structure observed`
- `## Loss masking strategy implications` (specifically: should I compute loss on analysis channel, final channel, or both? at what weight?)
- `## Format risks if Harmony structure is broken`

### Style requirements

- Heavy inline comments explaining *why* each step matters, not just what it does — this notebook is partly documentation for my team.
- When printing raw outputs with special tokens, wrap them in clearly visible delimiters like `===== RAW OUTPUT (specials visible) =====` so they're easy to find when scrolling.
- No silent fallbacks. If a special token isn't found, print a warning so I know.
- Don't use `print(tokenizer.decode(...))` without `skip_special_tokens=False` somewhere visible — the whole point is to see the specials.

### What I do NOT want

- Any use of `openai-harmony` or `gpt_oss` packages.
- Any `pip install` cells.
- Any abstraction layers — keep it flat and inspectable.
- Quantization config, bitsandbytes, or 4-bit loading — bf16 only.
- Generation with `skip_special_tokens=True` as the only output — I need to see specials.

Generate the notebook now as a single `.ipynb` file.

---

## Notes on what to watch for after running

The chat template in `tokenizer.chat_template` is the real ground truth for SFT data construction — more authoritative than any blog post or doc, because it's what the model was trained against. Once cell 2 runs, that Jinja template tells you exactly how to assemble training examples.

On the loss-masking question — once the channel structure is visible, the practical decision space narrows to:

- **(a)** train on final channel only with analysis masked — safest, but loses the ability to steer reasoning
- **(b)** train on both with full weight — highest collapse risk
- **(c)** train on both with analysis weighted 0.1–0.3x — middle path, reasonable default for conversion prediction in a regulated environment given that the task is closer to pattern recognition than multi-step reasoning

The notebook gives the format; the weighting decision is downstream and worth running an ablation on.
