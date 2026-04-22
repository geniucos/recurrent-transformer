#!/usr/bin/env python3

import math
import statistics
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, Optional

import torch
from torch.cuda import make_graphed_callables

from olmo.config import CudaGraphMode, TrainConfig
from olmo.model import OLMo, OLMoBlock


def create_test_config(base_cfg: TrainConfig, overrides: Dict = None) -> TrainConfig:
    cfg = deepcopy(base_cfg)
    if overrides:
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            obj = cfg
            for key in keys[:-1]:
                obj = getattr(obj, key)
            setattr(obj, keys[-1], value)
    cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_microbatch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = 1
    return cfg


def get_next_batch(cfg: TrainConfig, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(
        0, cfg.model.vocab_size,
        size=(cfg.device_train_batch_size, cfg.model.max_sequence_length),
        device=device, dtype=torch.long,
    )


def get_next_block_batch(cfg: TrainConfig, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(
        (cfg.device_train_batch_size, cfg.model.max_sequence_length, cfg.model.d_model),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )


def alibi_attention_bias(
    model: OLMo, cfg: TrainConfig, seq_len: int
) -> Optional[torch.Tensor]:
    """ALiBi slopes for ``seq_len``, or ``None`` when ALiBi is disabled.

    No causal mask: recurrent blocks enforce causality structurally, and the
    sequential OLMo forward path adds its own causal mask internally.
    """
    if not cfg.model.alibi:
        return None
    return model.get_alibi_attention_bias(seq_len, model.device)


def forward_pass(model: OLMo, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return logits.mean()


def cuda_capture_block(
    block: OLMoBlock,
    cfg: TrainConfig,
    attention_bias: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    class BlockWrapperForGraph(torch.nn.Module):
        def __init__(self, block, attention_bias=None):
            super().__init__()
            self.block = block
            self.attention_bias = attention_bias

        def forward(self, x):
            out, _ = self.block(x, attention_bias=self.attention_bias)
            return out

    device = block.ff_proj.weight.device
    return make_graphed_callables(
        BlockWrapperForGraph(block, attention_bias),
        (get_next_block_batch(cfg, device),),
        allow_unused_input=True,
        num_warmup_iters=20,
    )


class BlockWrapperForModel(OLMoBlock):
    """Wraps a graphed callable so it can sit in ``model.transformer.blocks``."""

    def __init__(self, graphed_callable, device: torch.device):
        torch.nn.Module.__init__(self)
        self.device = device
        self.graphed_callable = graphed_callable

    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False, max_doc_len=None, cu_doc_lens=None):
        assert (layer_past is None) and (use_cache is False) and (max_doc_len is None) and (cu_doc_lens is None), \
            "BlockWrapperForModel does not support layer_past, use_cache, max_doc_len, cu_doc_lens"
        return self.graphed_callable(x), None


def cuda_capture_model(model: OLMo, cfg: TrainConfig) -> None:
    attention_bias = alibi_attention_bias(model, cfg, cfg.model.max_sequence_length)

    if cfg.cuda_graph == CudaGraphMode.per_layer:
        for idx in range(len(model.transformer.blocks)):
            graphed = cuda_capture_block(model.transformer.blocks[idx], cfg, attention_bias)
            model.transformer.blocks[idx] = BlockWrapperForModel(graphed, model.device)
    else:
        assert cfg.cuda_graph == CudaGraphMode.whole, "cuda_capture_model called with invalid CUDA graph mode"

        class BlockWrapperForGraph(torch.nn.Module):
            def __init__(self, blocks, attention_bias=None):
                super().__init__()
                self.blocks = blocks
                self.attention_bias = attention_bias

            def forward(self, x):
                for block in self.blocks:
                    x, _ = block(x, attention_bias=self.attention_bias)
                return x

        graphed = make_graphed_callables(
            BlockWrapperForGraph(model.transformer.blocks, attention_bias),
            (get_next_block_batch(cfg, model.device),),
            allow_unused_input=True,
            num_warmup_iters=20,
        )
        model.transformer.blocks = torch.nn.ModuleList([BlockWrapperForModel(graphed, model.device)])


# --------------------------------------------------------------------------- #
# Timing helpers                                                              #
# --------------------------------------------------------------------------- #

def do_bench_iters(fn, warmup_iters: int = 3, reps: int = 5) -> float:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_ms = statistics.mean(times) * 1000.0
    stderr_ms = statistics.stdev([t * 1000.0 for t in times]) / math.sqrt(len(times))
    print(f"{mean_ms} +- {stderr_ms} ms")
    return statistics.median(times) * 1000.0


def profile_block(
    block: torch.nn.Module,
    cfg: TrainConfig,
    include_backwards: bool = True,
    attention_bias: Optional[torch.Tensor] = None,
) -> float:
    if isinstance(block, BlockWrapperForModel):
        input_example = get_next_block_batch(cfg, block.device)
    else:
        assert isinstance(block, OLMoBlock)
        input_example = get_next_block_batch(cfg, block.ff_proj.weight.device)

    def do_one_pass():
        block.zero_grad(set_to_none=False)
        whether_grad = nullcontext() if include_backwards else torch.no_grad()
        with whether_grad:
            out, _ = block(input_example, attention_bias=attention_bias)
        if include_backwards and out.requires_grad:
            out.sum().backward()

    return do_bench_iters(do_one_pass)
