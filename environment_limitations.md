# Environment Limitations: gpt-oss-120b + DeepSpeed ZeRO-3

## TL;DR

The current environment (`torch 2.6.0` / `triton 3.2.0`) is **below the floor** required to run gpt-oss-120b's MXFP4 packed runtime. Packed MXFP4 inference needs Hopper-class hardware (H100 / H200 / MI300X) **plus** `torch >= 2.7` / `triton >= 3.4`. Without that combination, the MXFP4 quantizer falls back to load-time dequantization, which does not compose with DeepSpeed ZeRO-3 — every rank materializes a partial 120B model on `cuda:0` and OOMs.

## What we tried

Running `02_sft_training.py` via the now-mandated `accelerate launch --config_file <ds_zero3.yaml> 02_sft_training.py` (the accelerate config carries `deepspeed_config` with stage 3, `zero3_init_flag: true`, and cpu optimizer offload). Earlier attempts with the direct `deepspeed --num_gpus=8 ... --deepspeed zero3` invocation hit the same wall — the failure mode is launcher-agnostic. With either launcher the script ran with:

- `dtype="auto"` so the MXFP4 weights would be honored on disk
- `zero3_init_flag: true` in the accelerate config (formerly `HfDeepSpeedConfig`) so `deepspeed.zero.Init` partitions tensors during construction
- `device_map` skipped under any distributed launcher (it conflicts with ZeRO-3 sharding and with FSDP wrapping)
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

| Component  | Required for packed MXFP4 + ZeRO-3 | Current         |
| ---------- | ---------------------------------- | --------------- |
| `triton` | >= 3.4                             | 3.2.0           |
| `torch`  | >= 2.7 (bundles triton 3.3+)       | **2.6.0** |
