#!/usr/bin/env python3
"""Benchmark one transformer block (n_layers=1) across block types, sequence
lengths, and batch sizes.  Outputs a CSV with vanilla and graphed/compiled
latencies.
"""

from debug_utils import environment_setup

environment_setup()

import argparse
import traceback
from contextlib import nullcontext
from itertools import product
from typing import Optional

import pandas as pd
import torch

from debug_utils import aggressive_cleanup, create_model, get_debug_base_config, mode_to_block_type
from olmo.efficient_utils import (
    BlockWrapperForModel,
    alibi_attention_bias,
    create_test_config,
    cuda_capture_block,
    profile_block,
)

DEFAULT_LATENCY = 999999.0


def run_benchmark(
    max_seq_len: int,
    batch_size: int,
    mode: str,
    device: torch.device,
    precision: str = "amp_bf16",
    include_backwards: bool = False,
    d_model: int = 1024,
    n_heads: int = 16,
    mlp_hidden_size: Optional[int] = None,
    use_alibi: bool = False,
):
    torch.compiler.reset()
    base_cfg = get_debug_base_config()
    if "sequential" not in mode:
        base_cfg.compile = None
    base_cfg.model.n_layers = 1
    base_cfg.model.rope = False
    if mlp_hidden_size is None:
        mlp_hidden_size = 4 * d_model
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    base_cfg.model.d_model = d_model
    base_cfg.model.n_heads = n_heads
    base_cfg.model.n_kv_heads = n_heads
    base_cfg.model.mlp_hidden_size = mlp_hidden_size

    overrides = {
        "model.max_sequence_length": max_seq_len,
        "model.block_type": mode_to_block_type(mode),
        "global_train_batch_size": batch_size,
        "precision": precision,
        "model.bwd_mlp_chunks": 4,
    }
    if use_alibi:
        overrides["model.alibi"] = True

    cfg = create_test_config(base_cfg, overrides)

    vanilla_latency = DEFAULT_LATENCY
    graphed_latency = DEFAULT_LATENCY
    max_memory_mb = None
    status = "failed"

    aggressive_cleanup()

    try:
        model = create_model(cfg, device)
        block = model.transformer.blocks[0]

        attention_bias = alibi_attention_bias(model, cfg, max_seq_len)

        vanilla_latency = profile_block(
            block, cfg, include_backwards=include_backwards, attention_bias=attention_bias,
        )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if cfg.compile is None:
            ctx = nullcontext() if include_backwards else torch.no_grad()
            with ctx:
                graphed_callable = cuda_capture_block(block, cfg, attention_bias)
            graphed_block = BlockWrapperForModel(graphed_callable, model.device)
        else:
            block.compile(**cfg.compile.asdict())
            graphed_block = block

        graphed_latency = profile_block(
            graphed_block, cfg, include_backwards=include_backwards, attention_bias=None,
        )

        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        status = "success"
        flag_name = "use_compiled" if cfg.compile is not None else "use_cuda_graph"

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        flag_name = "use_cuda_graph"
    finally:
        aggressive_cleanup()

    base_row = {
        "max_sequence_length": max_seq_len,
        "batch_size": batch_size,
        "mode": mode,
        "d_model": d_model,
        "n_heads": n_heads,
        "mlp_hidden_size": mlp_hidden_size,
        "max_memory_gb": max_memory_mb / 1024 if max_memory_mb is not None else None,
        "status": status,
        "with_backwards": include_backwards,
        "precision": precision,
        "use_alibi": use_alibi,
    }

    return [
        {**base_row, flag_name: False, "latency_ms": vanilla_latency},
        {**base_row, flag_name: True, "latency_ms": graphed_latency},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["recurrent", "recurrent_autograd", "sequential"])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--mlp-hidden-size", type=int, default=None, help="defaults to 4 * d_model")
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--backwards", action="store_true", help="Include backward pass in timing")
    parser.add_argument("--alibi", action="store_true", help="Benchmark with causal+ALiBi attention bias")
    args = parser.parse_args()

    torch._dynamo.config.cache_size_limit = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = args.mlp_hidden_size if args.mlp_hidden_size is not None else 4 * args.d_model

    results = []
    combos = list(product(args.seq_lens, args.batch_sizes, args.modes))
    for i, (seq_len, batch_size, mode) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] seq={seq_len}, batch={batch_size}, mode={mode}, backwards={args.backwards}")

        rows = run_benchmark(
            seq_len,
            batch_size,
            mode,
            device,
            args.precision,
            args.backwards,
            d_model=args.d_model,
            n_heads=args.n_heads,
            mlp_hidden_size=args.mlp_hidden_size,
            use_alibi=args.alibi,
        )
        if "recurrent" not in mode:
            best = min(rows, key=lambda r: r["latency_ms"])
            rows = [best]
        results.extend(rows)

        aggressive_cleanup()

        df = pd.DataFrame(results)
        df.to_csv("block_implementation_latencies.csv", index=False)

    print(
        f"\nCompleted. Results saved to block_implementation_latencies.csv "
        f"(d_model={args.d_model}, n_heads={args.n_heads}, mlp_hidden_size={mlp})"
    )


if __name__ == "__main__":
    main()
