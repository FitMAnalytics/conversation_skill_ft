# Environment Limitations: gpt-oss-120b + DeepSpeed ZeRO-3

## TL;DR

The current environment (`torch 2.6.0` / `triton 3.2.0`) is **below the floor** required to run gpt-oss-120b's MXFP4 packed runtime. Packed MXFP4 inference needs Hopper-class hardware (H100 / H200 / MI300X) **plus** `torch >= 2.7` / `triton >= 3.4`. Without that combination, the MXFP4 quantizer falls back to load-time dequantization, which does not compose with DeepSpeed ZeRO-3 — every rank materializes a partial 120B model on `cuda:0` and OOMs.

## What we tried

Running `02_sft_training.py` with `deepspeed --num_gpus=8 ... --deepspeed zero3` on the cluster. The script was already configured with:

- `dtype="auto"` so the MXFP4 weights would be honored on disk
- `HfDeepSpeedConfig` instantiated before `from_pretrained` (so `deepspeed.zero.Init` would partition tensors during construction)
- `device_map` removed under DeepSpeed (it conflicts with ZeRO-3 sharding)
- `stage3_gather_16bit_weights_on_model_save: true` for the save path
- `offload_optimizer: cpu` retained, `offload_param` removed

## What failed

OOM during model load, before training started. Stack trace (rank 7):

```
File ".../transformers/quantizers/quantizer_mxfp4.py", line 246, in create_quantized_param
    dequantize(module, param_name, param_value, target_device, ...)
File ".../transformers/integrations/mxfp4.py", line 139, in convert_moe_packed_tensors
    idx_lo = (blk & 0x0F).to(torch.long)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
GPU 0 has a total capacity of 79.10 GiB of which 2.91 GiB is free.
Process 622729 has 12.51 GiB memory in use.
Process 622730 has 8.51 GiB memory in use.
... (8 distinct PIDs all on GPU 0)
```

Two observations from this:

1. **All 8 ranks landed on `cuda:0`.** Eight different PIDs each holding 4–12 GB on the same GPU. ZeRO-3 partitioning never engaged during the load.
2. **MXFP4 weights are being unpacked to bf16 at load time** (`convert_moe_packed_tensors`, the `(blk & 0x0F)` line is 4-bit unpacking). The MoE expert weights expand from ~60 GB packed to ~240 GB bf16 — and each rank attempts to materialize that on cuda:0.

## Root cause

Two compounding issues, both downstream of the env being below the gpt-oss software floor.

### 1. MXFP4 weights are being dequantized at load time

The HF MXFP4 quantizer keeps weights packed at runtime **only** when the matching Triton kernels are available. Those kernels require:

- `triton >= 3.4` (for FP4 mma and packed-dtype support)
- a Hopper-class GPU (H100 / H200 / MI300X — A100 has no FP4 hardware)
- the `kernels` package installed (or `triton_kernels` directly)

When any of those is missing, the quantizer falls through to `convert_moe_packed_tensors(...)` during `_load_state_dict_into_meta_model`, dequantizing every expert tensor to bf16 in GPU memory. The 120B → 240 GB blow-up is the direct consequence.

### 2. The MXFP4 quantizer bypasses `deepspeed.zero.Init`

`HfDeepSpeedConfig` registers a hook so that `from_pretrained` allocates parameters via `deepspeed.zero.Init` and they get partitioned across ranks at construction time. The MXFP4 quantizer's `create_quantized_param` materializes real tensors directly on a target device — it does not go through the meta-device path that the hook intercepts. So:

- `zero.Init` sharding never engages for the dequantized expert tensors
- Without `device_map`, every rank's "target_device" defaults to `cuda:0`
- Eight ranks each try to put a (partially-dequantized) 120B model on `cuda:0`
- The first rank to overflow throws CUDA OOM

This is the same class of incompatibility as QLoRA + ZeRO-3: a runtime quantizer that allocates outside the ZeRO sharding context defeats the partitioning.

## Required vs. current environment

| Component | Required for packed MXFP4 + ZeRO-3 | Current |
|---|---|---|
| GPU architecture | Hopper or newer (H100 / H200 / MI300X) | _to confirm via `nvidia-smi`_ |
| `torch` | >= 2.7 (bundles triton 3.3+) | **2.6.0** |
| `triton` | >= 3.4 | **3.2.0** |
| `transformers` | >= 4.55 (with gpt-oss MXFP4 quantizer) | OK |
| `kernels` | latest (>= 0.13.x) | not installed |

`torch 2.6.0` ships with `triton 3.2.0` — both are below the floor. Upgrading `kernels` alone does not help: old triton cannot compile the MXFP4 ops, and no software upgrade rescues an A100 because MXFP4 needs FP4 silicon that A100 lacks.

## Paths forward

### A. If the cluster has Hopper-class GPUs (H100 / H200 / MI300X)

Upgrade the environment. Easiest is a fresh conda env following the gpt-oss install:

```bash
pip install -U torch transformers kernels
```

This pulls `torch 2.7+` (which brings `triton 3.4+`) and the `kernels` package. With that combination:

- MXFP4 stays packed at ~60 GB
- ZeRO-3 partitions to ~7.5 GB/rank across 8 GPUs
- The existing `02_sft_training.py` should run without further changes

### B. If the cluster is A100 (or any non-Hopper GPU)

Packed MXFP4 is not available on this hardware regardless of software. The pragmatic path is to dequantize the checkpoint to bf16 **once** and train against the bf16 copy, which DeepSpeed ZeRO-3 partitions normally.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

m = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    torch_dtype="bfloat16",
    device_map="cpu",
    quantization_config={"dequantize": True},
)
m.save_pretrained("/path/to/gpt-oss-120b-bf16", safe_serialization=True)
AutoTokenizer.from_pretrained("openai/gpt-oss-120b").save_pretrained(
    "/path/to/gpt-oss-120b-bf16"
)
```

The bf16 copy will be ~240 GB on disk. ZeRO-3 partitions it to ~30 GB/rank on 8x80 GB, which fits comfortably alongside activations and LoRA state. After dequant, point `--model-dir` at the bf16 directory and run the existing script — no other changes needed.

Run the dequant on a machine with at least ~256 GB of CPU RAM (or stream shard-by-shard on a smaller box).

## Open question

`nvidia-smi --query-gpu=name --format=csv,noheader` is the deciding factor — it tells us whether path A or path B applies. The 79.10 GiB capacity reported in the OOM is consistent with both A100-80GB and H100-80GB, so the symptom alone is not diagnostic.
