import gc
from typing import Any, Dict, Optional

import torch

from olmo.config import BlockType, TrainConfig
from olmo.model import LayerNormBase, OLMo


def environment_setup():
    import os
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "51235"
    os.environ.setdefault("CHECKPOINTS_PATH", "/tmp/checkpoints")


def create_model(config: TrainConfig, device: torch.device) -> OLMo:
    config.model.precision = config.autocast_precision
    model = OLMo(config.model)
    model = model.to(device)
    model.set_activation_checkpointing(config.activation_checkpointing)
    model.train()
    return model


def get_debug_base_config() -> TrainConfig:
    """Kempner C4 + 150m model defaults (no extra YAML overlay)."""
    base_config_path = "configs/kempner/base-c4-t5.yaml+configs/kempner/models/150m.yaml"
    cfg = TrainConfig.load(base_config_path)
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    # Base Kempner config uses ``meta`` for FSDP-style init; local scripts materialize on CUDA.
    cfg.model.init_device = "cuda"
    return cfg


def mode_to_block_type(mode: str) -> BlockType:
    """Convert mode string to BlockType. Asserts if mode is not recognized."""
    mapping = {
        "sequential": BlockType.sequential,
        "llama": BlockType.llama,
        "recurrent": BlockType.recurrent,
        "recurrent_autograd": BlockType.recurrent_autograd,
    }
    assert mode in mapping, f"Unknown mode: '{mode}'. Valid modes: {list(mapping.keys())}"
    return mapping[mode]


def initialize_recurrent_from_sequential(model_seq: OLMo, model_rec: OLMo):
    for block_seq, block_rec in zip(model_seq.transformer.blocks, model_rec.transformer.blocks):
        block_rec.init_from_sequential_block(block_seq)
    for module_type in ["wte", "emb_norm", "wpe", "ln_f", "ff_out"]:
        if module_type in model_rec.transformer and module_type in model_seq.transformer:
            src_module = model_seq.transformer[module_type]
            dst_module = model_rec.transformer[module_type]
            if isinstance(src_module, torch.nn.Linear):
                dst_module.weight.data.copy_(src_module.weight.data)
                if src_module.bias is not None:
                    dst_module.bias.data.copy_(src_module.bias.data)
            elif isinstance(src_module, torch.nn.Embedding):
                dst_module.weight.data.copy_(src_module.weight.data)
            else:
                assert isinstance(src_module, LayerNormBase), (
                    f"module type {module_type} is not a linear or embedding; not transferring"
                )


def aggressive_cleanup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Numerical comparison helpers                                                 #
# --------------------------------------------------------------------------- #


def compare_outputs(
    output1: torch.Tensor,
    output2: torch.Tensor,
    name: str = "outputs",
    verbose: bool = True,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    if output1.shape != output2.shape:
        stats["shape_match"] = False
        stats["shape1"] = str(output1.shape)
        stats["shape2"] = str(output2.shape)
        return stats

    stats["shape_match"] = True

    diff = output1 - output2
    abs_diff = torch.abs(diff)

    stats["max_diff"] = abs_diff.max().item()
    stats["mean_diff"] = abs_diff.mean().item()
    stats["max_abs_value1"] = torch.abs(output1).max().item()
    stats["max_abs_value2"] = torch.abs(output2).max().item()

    if verbose:
        print(f"\n{name} comparison:")
        print(f"  Max abs diff: {stats['max_diff']:.6e}")
        print(f"  Mean abs diff: {stats['mean_diff']:.6e}")

    return stats


def compare_losses(loss1: float, loss2: float, name: str = "losses", verbose: bool = True) -> Dict[str, float]:
    diff = abs(loss1 - loss2)
    rel_diff = diff / max(abs(loss1), abs(loss2), 1e-10)

    stats = {
        "abs_diff": diff,
        "rel_diff": rel_diff,
    }

    if verbose:
        print(f"\n{name} comparison:")
        print(f"  Loss 1: {loss1:.6f},  Loss 2: {loss2:.6f}")
        print(f"  Abs diff: {diff:.6e},  Rel diff: {rel_diff:.6e}")

    return stats


def _normalize_param_name(name: str) -> str:
    """Strip wrapper prefixes so graphed-callable params align with plain ones."""
    return name.replace("graphed_callable.block.", "")


def compare_gradients(
    model1: OLMo,
    model2: OLMo,
    param_name: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare gradients between two models (model1 = reference, model2 = test).

    Returns dict with ``max_grad_diff``, ``mean_grad_diff``, ``max_r2``,
    ``num_params_compared``.
    """
    stats: Dict[str, Any] = {}

    params1 = dict(model1.named_parameters())
    params2 = {_normalize_param_name(n): p for n, p in model2.named_parameters()}

    if param_name is not None:
        p1, p2 = params1[param_name], params2[param_name]
        if p1.grad is None or p2.grad is None:
            stats["has_grad"] = False
            return stats
        stats["has_grad"] = True
        diff = p1.grad - p2.grad
        stats["max_grad_diff"] = torch.abs(diff).max().item()
        stats["mean_grad_diff"] = torch.abs(diff).mean().item()
        stats["max_r2"] = ((diff ** 2).sum() / (p1.grad ** 2).sum()).item()
        if verbose:
            print(f"\nGradient comparison for {param_name}:")
            print(f"  Max grad diff: {stats['max_grad_diff']:.6e}")
            print(f"  R2: {stats['max_r2']:.6e}")
        return stats

    max_diffs = []
    mean_diffs = []
    max_r2 = 0.0
    compared = 0
    skipped = 0

    for name in params1:
        if name not in params2:
            skipped += 1
            continue
        grad1, grad2 = params1[name].grad, params2[name].grad
        if grad1 is None or grad2 is None:
            skipped += 1
            continue
        diff = grad1 - grad2
        max_diffs.append(torch.abs(diff).max().item())
        mean_diffs.append(torch.abs(diff).mean().item())
        r2 = (diff ** 2).sum() / (grad1 ** 2).sum()
        if verbose:
            print(f"{name} r2 = {(diff**2).mean().item():.6e} / {(grad1**2).mean().item():.6e} = {r2.item():.6e}")
        r2_val = 1.0 if torch.isnan(r2) else r2.item()
        max_r2 = max(max_r2, r2_val)
        compared += 1

    print(f"Gradient comparison: {compared} params compared, {skipped} skipped")

    if max_diffs:
        stats["max_grad_diff"] = max(max_diffs)
        stats["mean_grad_diff"] = sum(mean_diffs) / len(mean_diffs)
        stats["num_params_compared"] = compared
        stats["num_params_skipped"] = skipped
        stats["max_r2"] = max_r2
        if verbose:
            print(f"  Max grad diff: {stats['max_grad_diff']:.6e}")
            print(f"  Max R2: {stats['max_r2']:.6e}")
    else:
        stats["has_grad"] = False

    return stats
