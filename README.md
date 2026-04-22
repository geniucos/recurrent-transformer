# Recurrent OLMo

A fork of [OLMo](https://github.com/allenai/OLMo) that adds **block-recurrent transformer** layers. Each recurrent block recomputes its keys and values from the previous timestep's output (rather than from the original input), making attention structurally causal without a causal mask.

Two recurrent block types are provided:

| `block_type` | Implementation | Notes |
|---|---|---|
| `recurrent` | Tiled kernel with a custom `torch.autograd.Function` (`OLMoRecurrentBlockTiledFunction`) | Production path. O(L log L)tiled forward; memory-efficient backward via chunked MLP recomputation (`bwd_mlp_chunks`). |
| `recurrent_autograd` | Plain PyTorch loop (`OLMoRecurrentAutogradBlock`) | Reference implementation. Numerically equivalent to `recurrent`; used for correctness verification. |

The standard OLMo block types (`sequential`, `llama`) are unchanged and can be used normally.

## Setup

```bash
conda create -n recurrent python=3.11
conda activate recurrent
pip install -e .
```

The C4/T5 tokenized dataset is expected at the path configured in `configs/kempner/base-c4-t5.yaml` (`data.paths`). Adjust this to point to your local copy.

## Training

Training uses the standard OLMo entrypoint. A base config plus a model-size overlay defines the architecture; a sweep YAML adds recurrent-specific overrides.

**Single-GPU example** (150M params, seq_len=512, recurrent blocks):

```bash
export CHECKPOINTS_PATH=/path/to/checkpoints

python scripts/train.py \
    configs/kempner/base-c4-t5.yaml+configs/kempner/models/150m.yaml \
    --model.block_type=recurrent \
    --model.rope=false \
    --model.alibi=true \
    --model.max_sequence_length=512 \
    --model.init_device=cuda \
    --compile=null \
    --cuda_graph=whole \
    --global_train_batch_size=512 \
    --device_train_microbatch_size=512 \
    --distributed_strategy=single \
    --run_name=recurrent_150m_test
```

**SLURM sweep** (hyperparameter grid via `--array`):

```bash
# Edit scripts/kempner/launch_sweep.sh to set your account, partition, and paths.
# Adjust --array to match the number of grid points in the sweep config.
sbatch scripts/kempner/launch_sweep.sh
```

See `sweep_recurrent_150m_512.yaml` for a sweep config example. The sweep runner (`scripts/kempner/run_sweep.py`) expands the grid and maps `SLURM_ARRAY_TASK_ID` to a specific hyperparameter combination.

### Key config options

| Config field | Values | Description |
|---|---|---|
| `model.block_type` | `sequential`, `llama`, `recurrent`, `recurrent_autograd` | Block implementation. |
| `model.alibi` | `true` / `false` | ALiBi positional encoding. Recurrent blocks support ALiBi; they do **not** support RoPE (`model.rope` must be `false`). |
| `model.bwd_mlp_chunks` | integer (default 1) | Chunks for MLP recomputation in the tiled backward pass. Higher = less memory, slightly slower. 4–8 is typical. |
| `cuda_graph` | `null`, `per_layer`, `whole` | CUDA graph capture mode. `whole` captures all layers in one graph (best throughput for fixed shapes). |
| `compile` | `null` or compile config | `torch.compile` settings. Set to `null` when using `cuda_graph`. |
| `model.n_kv_heads` | integer | Must equal `model.n_heads` for recurrent blocks (GQA is not supported). |

### Model sizes

Model-size overlays are provided in `configs/kempner/models/`. All configs target roughly the stated non-embedding parameter count at different depth/width trade-offs:

| Config | d_model | n_heads | mlp_hidden_size | n_layers | bwd_mlp_chunks |
|---|---|---|---|---|---|
| **150m.yaml** | 1024 | 16 | 4096 | 12 | 4 |
| **150m_6.yaml** | 1408 | 22 | 5632 | 6 | 6 |
| **300m.yaml** | 1024 | 16 | 4096 | 24 | 8 |
| **300m_12.yaml** | 1408 | 22 | 5632 | 12 | 8 |
| **300m_6.yaml** | 2048 | 32 | 8192 | 6 | 8 |

## Verifying correctness

`debug_test_recurrent.py` compares the `recurrent` (tiled) block against the `recurrent_autograd` (reference) block by initializing both from the same sequential weights and comparing logits, loss, and per-parameter gradient R2.

```bash
# Basic run (all 4 alibi x norm_after configs, quiet output)
python debug_test_recurrent.py

# Verbose per-parameter breakdown
python debug_test_recurrent.py -v

# Also test the CUDA-graphed path
python debug_test_recurrent.py --cuda-capture

# Forward-only (skip backward/gradient comparison)
python debug_test_recurrent.py --no-backward
```

Expected output: `max_grad_r2` on the order of 1e-13 (fp32), 0 params skipped.

## Benchmarking block latency

`benchmark_block_latencies.py` measures per-block forward (and optionally backward) latency across sequence lengths, batch sizes, and block types. Outputs a CSV.

```bash
# Default sweep: recurrent vs recurrent_autograd vs sequential
python benchmark_block_latencies.py

# Specific configuration
python benchmark_block_latencies.py \
    --modes recurrent sequential \
    --seq-lens 128 256 512 1024 \
    --batch-sizes 256 512 \
    --d-model 1024 --n-heads 16 \
    --backwards \
    --alibi
```

Results are written to `block_implementation_latencies.csv`.

## Repository structure

```
olmo/
  config.py          # TrainConfig, ModelConfig, BlockType, CudaGraphMode
  model.py           # OLMoBlock hierarchy, OLMoRecurrentBlockTiled, OLMoRecurrentAutogradBlock,
                     #   OLMoRecurrentBlockTiledFunction (custom autograd)
  efficient_utils.py # CUDA graph capture, benchmarking helpers, alibi_attention_bias

configs/kempner/
  base-c4-t5.yaml    # Base training config (optimizer, data paths, eval)
  models/            # Model-size overlays (150m, 300m)

sweep_*.yaml         # Sweep configs for hyperparameter grids

scripts/
  train.py                   # Training entrypoint
  kempner/launch_sweep.sh    # SLURM job template
  kempner/run_sweep.py       # Sweep grid expansion + launch

debug_test_recurrent.py      # Numerical equivalence test
debug_utils.py               # Model creation, weight transfer, comparison helpers
benchmark_block_latencies.py # Block latency profiling
```
