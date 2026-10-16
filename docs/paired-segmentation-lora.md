# Nano paired segmentation LoRA

The `segmentation_sft_nano` experiment uses the native trainer, native attention LoRA, the existing paired transfer sample processor, and the existing flow matching objective. Its mechanics recipe uses 93 RGB target frames at 1280x704 with paired colored segmentation controls, rank 16, alpha 32, AdamW at `1e-4`, BF16, one sample per rank, gradient accumulation of one, and no compilation or EMA. The selected topology uses two FSDP ranks and an effective global batch of two. Twenty updates and a checkpoint at update ten are qualification defaults; they do not establish style quality or the memory fit of the full recipe.

The TOML explicitly sets `[job].experiment_module` to `cosmos_framework.configs.base.experiment.sft.vision_sft_nano`. This uses the existing native config loader's optional module-selection interface to register the paired recipe without importing unrelated robotics experiments and their runtime dependencies. Omitting the field preserves the complete shipped experiment catalog; the selector is metadata and is never emitted as a Hydra `job` override.

## Source and checkpoint boundary

The integration starts from framework commit `be9737c72bb17479e96c6edb7b2449ee857d3b02`, including the project Transformers 5 port. The source license is OpenMDW-1.1; retain `LICENSE` and origin notices when distributing this integration. Record the final patched revision and effective runtime inventory for every attempt.

`checkpoint.load_path` consumes a native distributed checkpoint containing `model/`, not a Diffusers directory. The supported CPU conversion entrypoint is:

```bash
python -m cosmos_framework.inference2.convert_model_to_dcp \
  --checkpoint "$COSMOS3_DIFFUSERS_PATH" \
  --output-path "$COSMOS3_DCP_PATH"
```

The converter reads the staged root `config.json["model"]`, restores native model targets, uses the inverse key mapping in `cosmos_framework/inference2/_model_io.py`, and writes the native network state through `torch.distributed.checkpoint`. It also initializes frozen tokenizers from native artifact registrations. A staged Diffusers VAE directory alone is not a verified replacement for the native `WAN_VAE_PATH` file. The paired recipe retains the staged Nano action and sound heads, uses the native AVAE with `normalization_type="none"`, and matches its untied text embeddings. Point `AVAE_PATH` at the verified native audio-tokenizer artifact; a Diffusers audio-tokenizer directory is not directly accepted by `AVAEInterface`. Conversion and key names alone do not establish numerical base parity.

The published native Nano configuration uses `Qwen3VLTextForCausalLM` with `include_visual=False`. Its checkpoint contains the text/generation network and action/sound heads; the separate Diffusers `vision_encoder` is outside that native architecture. On the qualified staged base, 814 native BF16 tensors correspond to source tensors, while exactly 351 `language_model.visual.*` tensors belong to the separate vision encoder and are absent from the native network. Retain the complete excluded-key inventory and vision-component hash, and reject any other unexplained omission. Segmentation controls enter the Wan VAE in native training. Portable adapters bind the original full Diffusers base, whose separate vision encoder remains present. This explicit component boundary does not justify relaxed loading of any required native key.

The four native attention generation projections are mapped by the existing forward converter and its inverse:

| Native leaf | Diffusers leaf |
| --- | --- |
| `q_proj_moe_gen` | `add_q_proj` |
| `k_proj_moe_gen` | `add_k_proj` |
| `v_proj_moe_gen` | `add_v_proj` |
| `o_proj_moe_gen` | `to_add_out` |

The source mapping preserves tensor orientation. Native LoRA stores sibling `<projection>.lora_A.weight` with shape `[rank, in_features]` and `<projection>.lora_B.weight` with shape `[out_features, rank]`, and computes `base(x) + (alpha / rank) * B(A(x))`. Parameter names, counts, shapes, scaling, and optimizer groups must still be enumerated on the actual staged model. Zero-adapter parity and fresh-process exported-adapter parity are execution gates, not consequences of the mapping table.

The inspected staged BF16 Nano transformer has 36 layers and 144 generation projections: Q/output weights are `[4096,4096]`, and K/V weights are `[1024,4096]`. The declared rank-16 adapter therefore has 288 tensors and 15,335,424 parameters. These counts come from the staged safetensors headers, not an allocated native model. Its root config SHA-256 is `c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6`; retain the complete per-layer mapping alongside runtime evidence.

## Paired data route

Each frozen training JSONL record supplies `uuid`, `vision_path`, `control_path`, `width`, `height`, `duration`, `nb_frames`, `framerate`, and `t2w_windows`. Each window supplies `start_frame`, `end_frame`, and a grounded structured object under `caption_json`. Relative media paths resolve relative to the JSONL. No original media or latent cache is modified. `caption_json` takes priority over dense captions and is serialized with the canonical `caption_json_to_prompt` function; the model's existing tokenizer and embeddings are reused.

`TransferSFTDataset.process_one_sample` decodes real RGB, applies the selected source window, temporal interval, resize, and center crop, then loads the segmentation control with the same indices and crop. Segmentation resizing uses nearest neighbor. The decoded streams must have equal complete shapes. Both become float `[3,T,H,W]` tensors in `[-1,1]`, ordered `[control, real_RGB_target]`. The 49-frame diagnostic uses the same processor with a separately frozen geometry override; it is no longer excluded by an unrelated 61-frame minimum. It does not qualify 93-frame geometry.

`SequencePlan.share_vision_temporal_positions=True` aligns control and target token positions. The native packer's `resolve_item_condition_frames` marks all latent control frames clean and leaves target frames unconditioned for this recipe. The existing VAE encodes both streams. The model's noise construction multiplies control sigmas by zero through the condition mask; only RGB target latents receive the denoising objective. The recipe explicitly excludes fully conditioned items from scalar loss reduction. Caption CFG dropout remains `0.1`; controls are always present, target initial-RGB conditioning is zero, and no segmentation-derived loss mask is introduced. Perturbing a control under fixed other inputs is a required runtime proof that the model consumes these tokens.

## Native training and recovery

```bash
export DATASET_PATH=/absolute/path/to/frozen/view
export BASE_CHECKPOINT_PATH=/absolute/path/to/native/dcp
export WAN_VAE_PATH=/absolute/path/to/native/Wan2.2_VAE.pth
export AVAE_PATH=/absolute/path/to/native/avae_48k_noncausal_25hz_64ch.ckpt
export TEXT_TOKENIZER_PATH=/absolute/path/to/verified/base/text_tokenizer
export LORA_AUDIT_DIR=/absolute/path/to/attempt/adapter-audit
export IMAGINAIRE_OUTPUT_ROOT=/absolute/path/to/attempt/training
export CUDA_VISIBLE_DEVICES="$ADMITTED_GPU_UUID_RANK0,$ADMITTED_GPU_UUID_RANK1"
export COSMOS_ALLOW_SINGLE_SAMPLE_DENSE_TRAINING=1
export I4_ATTN_BACKENDS=flash2
export FLASH_ATTENTION_DETERMINISTIC=1
torchrun --nproc-per-node=2 -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/segmentation_sft_nano.toml
```

Set `ADMITTED_GPU_UUID_RANK0` and `ADMITTED_GPU_UUID_RANK1` to the two admitted GPU UUIDs in their fixed rank order. The selected topology has data-parallel sharding degree two, replication/context/CFG parallel degrees one, and one sample per rank with `grad_accum_iter=1`, giving an effective global batch of two. Preserve the same ordered UUIDs and topology for the recovery comparison.

The selected A800 runtime requires the dense-path settings from the [Transformers 5/A800 port](imsight-nano-transformers5-a800.md): `COSMOS_ALLOW_SINGLE_SAMPLE_DENSE_TRAINING=1` admits eager single-sample training, and `I4_ATTN_BACKENDS=flash2` selects the available backend. Without the dense opt-in, gradient-enabled execution requests varlen attention, which this backend rejects. `FLASH_ATTENTION_DETERMINISTIC=1` is resolved by the native attention frontend on each call and requests deterministic backward from the selected backend; an explicit caller request also remains honored. Keep global deterministic algorithms disabled for this recovery probe so the inherited trainer does not reseed after loading checkpoint RNG. Kernel dispatch evidence and actual continuation comparisons remain separate gates.

The measured single-GPU attempt `training-single-gpu-004` evaluated 93-frame 1280x704 paired inputs with rank-16 adapters and BF16 on one A800 and failed at the first gradient-enabled MLP forward: a 992MiB allocation exceeded the reported 979.38MiB free memory, before any optimizer update. The required 3GiB Torch reservation was resident. This is a scoped capacity observation for that recipe/runtime. The paired task selects the measured two-rank topology. Multiple samples per rank and per-caption subdivisions still require a separately qualified varlen backend. Keep compilation disabled and one sample per rank; additional data-parallel ranks do not increase the number of samples within each rank's pack.

The paired TOML sets `checkpoint.dcp_per_thread_copy_ahead=0`, selecting the installed PyTorch public filesystem writer's serial CPU staging path. The framework default remains `None`, which returns the original `FileSystemWriter(path=...)` constructor. An explicit value also enables a small local writer wrapper that logs the original `write_data` or future-wait traceback as plain text on the failing rank and rethrows the same error, preserving the returned future on success. The measured component qualification covers local filesystem saving with `checkpoint.dcp_async_mode_enabled=false`. This setting does not qualify the asynchronous producer's pinned staging or object-store writers. Full Nano checkpoint/recovery remains a separate execution gate.

`TransferSFTMapDataset` exposes explicit paired windows to the existing `MapDistributor`, which owns seeded shuffling, rank partitioning, and cursor positions. Failed frozen samples raise instead of silently changing the observed data order. `CosmosDataLoaderStateCallback` saves consumed epoch/index positions through the native DCP dataloader component. Native DCP already saves adapters together with model state, optimizer, scheduler, trainer iteration, scaler, and per-rank Python/NumPy/PyTorch/CUDA RNG state. Same-job auto-resume restores all components; adapter-loading exclusions apply only to the initial base warm start.

For the recovery comparison, retain the uninterrupted run's update-ten checkpoint and start a separate job with a fresh output directory. Apply these trailing overrides to the same TOML:

```bash
job.name=resume-from-ten \
checkpoint.load_path=/absolute/path/to/reference/checkpoints/iter_000000010 \
checkpoint.load_training_state=true \
'checkpoint.keys_to_skip_loading=[]' \
'checkpoint.keys_not_to_resume=[]' \
trainer.max_iter=20
```

The explicit checkpoint path must identify the directory containing `model/`, `optim/`, `scheduler/`, `trainer/`, and `dataloader/`. An existing destination `latest_checkpoint.txt` takes precedence over `checkpoint.load_path`, so verify that the separate job has no resumable checkpoint before launch. The native loader treats this explicit path as a warm start even when loading every training component; clearing both exclusion lists therefore restores the trained adapters and all training state. This route uses the supported loader without copying the full checkpoint or manufacturing a latest-checkpoint pointer. Verify the logged source path and restored iteration, all five component inventories, and each rank's actual dataloader pickle; a missing dataloader file otherwise logs a skip.

`trainer.max_iter` is the absolute number of completed optimizer updates. Loading iteration ten with `max_iter=20` executes ten additional updates. Keep the same twenty-update scheduler configuration in both reference and resumed jobs, including its ten-update warmup, and preserve world size, model/optimizer settings, loader seed, and dataset bindings. The consumed cursor is recorded before each checkpoint save. Avoid the script's `--deterministic` option for this recovery probe: its global deterministic-algorithms setting triggers the inherited trainer's post-load seed reset, replacing the restored RNG trajectory. Explicit `trainer.seed=42`, `trainer.cudnn.deterministic=true`, `trainer.cudnn.benchmark=false`, and `FLASH_ATTENTION_DETERMINISTIC=1` preserve the checkpoint RNG loading path; actual continuation parity still requires measurement.

The native `GenerationLoraAudit` callback records the exact adapter allowlist and optimizer groups, per-update losses and gradient norms, timings, allocated/reserved memory, sample cursors, adapter deltas, and complete per-rank frozen-parameter fingerprints. Give each uninterrupted or resumed process a fresh `LORA_AUDIT_DIR` so evidence is retained independently. This callback observes the upstream objective and optimizer; it does not replace them. `TEXT_TOKENIZER_PATH` binds the verified base tokenizer directly, keeping prompt construction independent of mutable Hub downloads and adding no trainable token embeddings.

This replay recipe requires `num_workers=0`, one-sample batches and `grad_accum_iter=1`. Worker RNG state and packing lookahead are outside its supported replay boundary; the paired map adapter rejects worker execution. `grad_accum_iter=4` is outside the qualified recipe and has no measured memory, schedule or recovery guarantee here. A dedicated loader seed prevents iterator reconstruction from consuming the trainer's next diffusion-noise draw. The campaign must separately establish an uninterrupted 20-update reference, fresh-process resume at ten with the original tolerances, complete frozen-weight fingerprints and actual adapter updates, pretrained export/reload parity, and fully decoded matched videos. Mark each gate passed only from its actual result; checkpoint presence, a component proof or a prepared request does not establish those outcomes.

## Focused checks

```bash
python -m pytest -c /dev/null --noconftest -p no:cacheprovider -q \
  cosmos_framework/configs/toml_config/experiment_module_test.py \
  cosmos_framework/data/generator/local_datasets/transfer_sft_dataset_test.py \
  cosmos_framework/data/generator/dataflow/resume_test.py \
  cosmos_framework/utils/generator/lora_export_test.py \
  cosmos_framework/callbacks/generation_lora_audit_test.py \
  cosmos_framework/model/generator/tokenizers/audio/avae_test.py \
  cosmos_framework/model/attention/frontend_determinism_test.py \
  cosmos_framework/checkpoint/dcp_writer_test.py
```

These focused CPU unit checks run independently of the upstream global pytest configuration and conftest plugin stack. They cover frame selection and nearest-neighbor flags, control/target roles and shared positions, 49-frame admission, caption budget preservation, strict failed-sample behavior, worker-RNG rejection, cursor recovery, preservation of the next diffusion-noise draw, adapter mapping, audit calculations, freezing final materialized AVAE parameters after weight-norm removal, deterministic attention dispatch, serial-writer selection and CPU DCP value restoration. They are separate from the complete upstream test suite. They do not load the full base model or prove full-model backward execution, actual trained-adapter export, GPU memory fit, multi-rank behavior, video decode quality, style learning, or mask adherence.
