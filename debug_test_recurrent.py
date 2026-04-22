#!/usr/bin/env python3
"""Equivalence test: recurrent_autograd (reference) vs recurrent (tiled kernel).

Builds a sequential model, copies weights into both recurrent variants,
then compares logits, loss, and gradients.  Reports max absolute errors
and per-parameter gradient R2 (||Δg||² / ||g_ref||²).

With ``--cuda-capture``, the tiled recurrent blocks are CUDA-graph captured
before the comparison, validating that the graphed path is numerically
equivalent.
"""

from debug_utils import environment_setup

environment_setup()

import argparse
from pprint import pprint

import torch

from olmo.config import BlockType, CudaGraphMode, TrainConfig
from debug_utils import (
    aggressive_cleanup,
    compare_gradients,
    compare_losses,
    compare_outputs,
    create_model,
    get_debug_base_config,
    initialize_recurrent_from_sequential,
)
from olmo.efficient_utils import (
    compute_loss,
    create_test_config,
    cuda_capture_model,
    forward_pass,
    get_next_batch,
)

_EQUIV_OVERRIDES = {
    "model.d_model": 128,
    "model.mlp_hidden_size": 512,
    "model.n_heads": 4,
    "model.n_kv_heads": 4,
    "model.n_layers": 4,
    "model.max_sequence_length": 64,
    "model.vocab_size": 64,
    "model.embedding_size": 64,
    "global_train_batch_size": 32,
}


def run_single_test(
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
    include_backwards: bool = True,
    cuda_capture: bool = False,
):
    torch.compiler.reset()
    aggressive_cleanup()
    assert config.model.block_type == BlockType.sequential
    config.precision = "fp32"

    config_baseline = create_test_config(config, {
        "model.block_type": BlockType.recurrent_autograd,
        "precision": "fp32",
    })
    config_test = create_test_config(config, {
        "model.block_type": BlockType.recurrent,
    })

    model_seq = create_model(config, device)
    model_baseline = create_model(config_baseline, device)
    model_test = create_model(config_test, device)
    initialize_recurrent_from_sequential(model_seq, model_baseline)
    initialize_recurrent_from_sequential(model_seq, model_test)
    del model_seq

    if cuda_capture:
        if verbose:
            print("CUDA-capturing recurrent blocks …")
        config_test.cuda_graph = CudaGraphMode.per_layer
        cuda_capture_model(model_test, config_test)

    if include_backwards:
        model_baseline.zero_grad(set_to_none=False)
        # After CUDA capture, .grad tensors must never be deallocated (the
        # captured graph writes to their fixed addresses).  Always use
        # set_to_none=False from this point on.
        model_test.zero_grad(set_to_none=False)

    if verbose:
        print("Running full model test")
    input_ids = get_next_batch(config, device)

    logits_baseline = forward_pass(model_baseline, input_ids, attention_mask=None)
    logits_test = forward_pass(model_test, input_ids, attention_mask=None)
    logits_stats = compare_outputs(logits_baseline, logits_test, "logits", verbose=verbose)

    loss_baseline = compute_loss(logits_baseline, input_ids)
    loss_test = compute_loss(logits_test, input_ids)
    loss_stats = compare_losses(loss_baseline.item(), loss_test.item(), "loss", verbose=verbose)

    if include_backwards:
        torch.autograd.set_multithreading_enabled(False)
        loss_baseline.backward()
        model_test.zero_grad(set_to_none=False)
        loss_test.backward()
        torch.autograd.set_multithreading_enabled(True)
        grad_stats = compare_gradients(model_baseline, model_test, verbose=verbose)
    else:
        grad_stats = {}

    del loss_baseline, logits_baseline, loss_test, logits_test

    return {
        "max_logits_abs_diff": logits_stats.get("max_diff", float("inf")),
        "max_loss_abs_diff": loss_stats.get("abs_diff", float("inf")),
        "max_grad_abs_diff": grad_stats.get("max_grad_diff", float("inf")),
        "max_grad_r2": grad_stats.get("max_r2", 0.0),
        "num_params_compared": grad_stats.get("num_params_compared", 0),
        "num_params_skipped": grad_stats.get("num_params_skipped", 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-capture", action="store_true", default=False,
        help="CUDA-graph capture the tiled recurrent blocks before comparing.",
    )
    parser.add_argument(
        "--no-backward", action="store_true", default=False,
        help="Skip backward pass and gradient comparison.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print per-parameter gradient R2 and detailed comparisons.",
    )
    args = parser.parse_args()

    torch._dynamo.config.cache_size_limit = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.cuda_capture:
        print("CUDA capture mode enabled")
    if args.no_backward:
        print("Backward pass disabled")
    print()

    base_cfg = get_debug_base_config()
    base_cfg.compile = None

    # Recurrent blocks assert ``not rope``; only exercise rope=False.
    test_configs = [
        # {"rope": False, "alibi": False, "norm_after": False},
        # {"rope": False, "alibi": False, "norm_after": True},
        # {"rope": False, "alibi": True, "norm_after": False},
        {"rope": False, "alibi": True, "norm_after": True},
    ]

    for settings in test_configs:
        config_name = ", ".join(f"{k}={v}" for k, v in settings.items())

        overrides = {
            "model.block_type": BlockType.sequential,
            "model.rope": settings["rope"],
            "model.alibi": settings["alibi"],
            "model.norm_after": settings["norm_after"],
            **_EQUIV_OVERRIDES,
        }
        config = create_test_config(base_cfg, overrides)

        print(f"Running test with config: {config_name}")
        result = run_single_test(
            config, device,
            verbose=args.verbose,
            include_backwards=not args.no_backward,
            cuda_capture=args.cuda_capture,
        )
        result["config_name"] = config_name

        print("=" * 80)
        print("SUMMARY  (max_grad_r2 = max over params of ||Δg||² / ||g_ref||²)")
        print("=" * 80)
        pprint(result)
        print("-" * 100)
        if torch.cuda.is_available():
            print(f"Peak GPU mem allocated: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")
            print(f"Peak GPU mem reserved:  {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")
        print()


if __name__ == "__main__":
    main()
