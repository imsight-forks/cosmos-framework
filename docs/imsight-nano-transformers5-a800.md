# Imsight Cosmos3-Nano / Transformers 5 on A800

This fork's `imsight/cosmos3-nano-transformers5-a800` branch carries the bounded Nano generation-LoRA compatibility changes based on NVIDIA commit `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. The originating project tracks this repository as a Git submodule at `extern/tracked/cosmos-framework`.

## Changes

- Package metadata admits Transformers `>=4.57.1,<5.2.0`. The tested candidate is exactly **5.1.0**; other 5.x versions are not qualified.
- FlashAttention 2 accepts top-left causal masking for equal-length dense Q/K sequences, where it matches the supported bottom-right mask. Unequal-length top-left masks and varlen requests remain rejected.
- `COSMOS_ALLOW_SINGLE_SAMPLE_DENSE_TRAINING=1` opts single-sample training into the existing dense attention path. The default remains unchanged. Multiple samples and caption subdivisions still require varlen attention.

The attention changes address a runtime limitation also present with Transformers 4.57.1. Upstream training always selects varlen attention, its FlashAttention 2 adapter rejects varlen for instability, and its cuDNN adapter requires a newer runtime than the tested Torch build provides.

For the tested eager single-sample path, set these variables on the training process:

```bash
COSMOS_ALLOW_SINGLE_SAMPLE_DENSE_TRAINING=1
I4_ATTN_BACKENDS=flash2
```

Use process-scoped environment variables or export them in the launch shell. This is not a complete training recipe. FlashAttention 2 must match the Python, Torch, CUDA, and GPU versions in the runtime.

## Validation and scope

The 2026-10-16 A800 experiment used Python 3.13, Torch 2.10.0 / CUDA 12.8, PyAV 18.1.0, and NVIDIA FlashAttention 2.7.4.post1. It compared Transformers 4.57.1 with 5.1.0 using the same saved fixture and native `Qwen3VLTextForCausalLM` MoT implementation, reduced to two layers and 722,688 base parameters.

Native rank-16 / alpha-32 LoRA targeted generation Q/K/V/output projections: 32,768 trainable parameters across sixteen tensors. Three AdamW updates on synthetic generation-output MSE gave identical losses: `1.8363336325 -> 1.8309371471 -> 1.8246470690`. First outputs, all first-step adapter gradients, and final adapters were bitwise equal across the two Transformers versions. Frozen parameters stayed unchanged.

Native dense attention also matched Torch math SDPA in forward and Q/K/V backward, including padded and unpadded packs. Package-manager resolution passed for the originating project's complete `infer-x64` requirements plus the training additions. The native training CLI help and eight native module imports passed under 5.1.0.

This evidence does not qualify full-size Nano training, the video denoising objective, segmentation conditioning, distributed execution, compilation, checkpoint/resume, or adapter export/reload. Multi-sample packing still needs a qualified A800 varlen backend. The optional reasoner HF monkey patches retain their `4.57.` version guard; the relaxed package constraint does not port those paths.

The originating project's report and ignored evidence bundle are located at:

```text
context/campaigns/debug/2026-10-15-cosmos3-nano-style-lora/
  reports/2026-10-16-a800-transformers5-compatibility.md
  data/iteration-01/attempt-001-transformers5-compat/
```

The fork preserves the upstream locked dependency versions, including Transformers 4.57.6, and synchronizes the lock's package requirement metadata with the relaxed constraint. The tested Transformers 5.1.0 environment is pinned separately by Pixi in that evidence bundle. Do not infer that plain `uv sync` selects or qualifies Transformers 5.
