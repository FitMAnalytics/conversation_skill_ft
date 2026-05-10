# gpt-oss-120b Gotchas

Small, easy-to-miss issues we've hit while running gpt-oss-120b inference and
fine-tuning in this environment. Add new entries as they come up — each entry
should give the symptom, the cause, and the fix. The systemic env-floor issue
(MXFP4 + ZeRO-3) lives in `environment_limitations.md`; this file is for the
papercuts.

---

## 1. `KeyError: 'shape'` from `model.generate` (and from `inputs['input_ids'].shape`)

> **Use render-then-tokenize for any gpt-oss inference call.** This is the
> known-good pattern matching `gpt_oss_inspection_notebook` — verified working
> in this environment. Don't let `apply_chat_template` tokenize for you.

**Symptom.** One of two errors, depending on what you tried:
- `model.generate(input_ids, ...)` raises `KeyError: 'shape'` deep inside
  transformers when `input_ids` came from `apply_chat_template(...,
  return_tensors="pt")`.
- After "fixing" by adding `return_dict=True` and splatting with `**inputs`,
  the same `KeyError: 'shape'` reappears at `inputs["input_ids"].shape[1]` —
  because `inputs["input_ids"]` is itself a dict, not a tensor.

**Cause.** `apply_chat_template`'s return shape when it does its own
tokenization is version- and template-dependent on gpt-oss:
- With `return_tensors="pt"` alone, it returns a dict-like `BatchEncoding`
  (not a tensor). Passing that positionally as `input_ids` makes `generate`
  try `inputs['shape']` and crash.
- With `return_dict=True` added, some transformers/gpt-oss combinations nest
  structured content under `"input_ids"` (token dicts rather than a tensor),
  so `inputs["input_ids"].shape` crashes the same way.

The chat template's *string* rendering is stable — only the tokenization
layer is flaky. So render to a string, then tokenize yourself.

**Fix (verified working).** Two-step: `tokenize=False` to get the
harmony-formatted string, then standard `tokenizer(...)` to get a clean
`BatchEncoding` whose `input_ids` is always a real tensor.

```python
try:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        reasoning_effort="medium",
    )
except TypeError:  # older tokenizer rejects reasoning_effort
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

out = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
new_tokens = out[0, input_len:]
```

**Why this preserves the analysis channel.** `apply_chat_template(...,
tokenize=False, reasoning_effort=...)` still injects the harmony control
tokens (`<|start|>system<|message|>...Reasoning: high...<|channel|>analysis...`)
into the rendered string. The model still emits `<|channel|>analysis<|message|>`
and `<|channel|>final<|message|>` blocks; we still parse them with regex on the
decoded output (see gotcha #4). End-to-end behavior is unchanged — only the
tokenization step moved out of `apply_chat_template`.

**Re-use this pattern.** Stage 02 (SFT) and Stage 03 (inference) should use
the same render-then-tokenize path. `_generate` / `run_and_show` in
`01_preprocessing.py` is the reference implementation.

---

## 2. `dtype="auto"` OOMs cuda:0 at model load

**Symptom.** `from_pretrained(..., dtype="auto", device_map="auto")` OOMs the
first GPU during load, well before generation.

**Cause.** On torch 2.6 / triton 3.2 the MXFP4 quantizer can't keep experts
packed and dequantizes them to bf16 at load (~240 GB total). With `dtype="auto"`
the dequant intermediates land on `cuda:0` *before* `device_map="auto"`
dispatches modules across GPUs, so cuda:0 fills up first.

**Fix.** Force `torch_dtype=torch.bfloat16` so the dequant tensors flow through
`device_map` and shard across visible GPUs:

```python
AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=True,
)
```

Background on the underlying MXFP4 / ZeRO-3 incompatibility: see
`environment_limitations.md`.

---

## 3. Setting `CUDA_VISIBLE_DEVICES` after importing torch is silently ignored

**Symptom.** You wanted to use GPUs `[0,1,2,3]` but the model loads onto a
faulty GPU anyway, or `nvidia-smi` shows the process on different devices than
expected.

**Cause.** `torch` snapshots `CUDA_VISIBLE_DEVICES` at import time. Setting it
later via `os.environ` has no effect on the running process.

**Fix.** Set it before importing torch (and before any module that imports
torch transitively, including `transformers`). The notebook does this with an
assertion at the top:

```python
import os
assert "torch" not in globals(), "torch already imported — restart kernel"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

In the batch script we set it from `--cuda-visible` *before* the first
`import torch` line.

---

## 4. Harmony channel markers must be parsed with `skip_special_tokens=False`

**Symptom.** You decode the model's output and the `<|channel|>analysis...`
and `<|channel|>final...` markers are gone, so you can't split CoT from the
final response.

**Cause.** The default `tokenizer.decode(...)` uses `skip_special_tokens=True`,
which strips harmony channel tokens. The CoT/final split is then impossible
without re-running.

**Fix.** Decode with `skip_special_tokens=False`, then split with regex:

```python
ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(?P<c>.*?)(?=<\|end\|>|<\|start\|>|<\|return\|>|\Z)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(?P<c>.*?)(?=<\|return\|>|<\|end\|>|\Z)",
    re.DOTALL,
)
raw = tokenizer.decode(out[0, input_len:], skip_special_tokens=False)
analysis = (ANALYSIS_RE.findall(raw) or [""])[-1].strip()
final = (FINAL_RE.findall(raw) or [raw])[-1].strip()
```

Be defensive: if the model misformats and channel markers are missing, fall
back to treating the whole decode as `final` and log a warning.

---

## 5. `apply_chat_template` rejects `reasoning_effort` on older tokenizers

**Symptom.** `TypeError: apply_chat_template() got an unexpected keyword
argument 'reasoning_effort'`.

**Cause.** The `reasoning_effort` kwarg is plumbed through the gpt-oss chat
template specifically; older / non-gpt-oss tokenizers don't accept it.

**Fix.** Try-with-fallback so the same code works across tokenizers:

```python
try:
    inputs = tokenizer.apply_chat_template(
        messages, reasoning_effort=effort, **kwargs)
except TypeError:
    inputs = tokenizer.apply_chat_template(messages, **kwargs)
```

---

## 6. Left-padding required for batched generation

**Symptom.** Batched generation produces garbage for shorter prompts in the
batch.

**Cause.** Default `tokenizer.padding_side` is "right". For decoder-only
generation, padding tokens on the right of the prompt push the next-token
prediction off the end of real content.

**Fix.** Set padding side to left at load time:

```python
if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"
```

(`load_teacher_model` in `01_preprocessing.py` does this.)

---

## 7. Module name can't start with a digit — notebooks need `importlib`

**Symptom.** `import 01_preprocessing` is a SyntaxError.

**Cause.** Python identifiers can't begin with a digit, and our convention is
to prefix scripts with stage numbers (`01_`, `02_`, …).

**Fix.** Load via `importlib.util` in the notebook:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "01_preprocessing.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
```

Then iterate prompts in the .py file and `importlib.reload(p)` to pick up
edits without restarting the kernel.
