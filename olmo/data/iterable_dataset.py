import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.data

from ..aliases import PathOrStr
from ..torch_util import barrier, get_fs_local_rank, get_global_rank, get_world_size
from ..util import roundrobin, threaded_generator

__all__ = ["IterableDataset"]

log = logging.getLogger(__name__)

class IterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """
    Multi-pass, memmapped indices. Final logical length = max_examples * num_passes.
    Each pass-sized block is independently shuffled (or reshuffle of a fixed subset).
    """

    def __init__(
        self,
        dataset: Union[Sequence[List[int]], Sequence[torch.Tensor], Sequence[Dict[str, Any]]],
        global_batch_size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        start_index: int = 0,
        max_examples: Optional[int] = None,       # REQUIRED for multi-pass
        num_passes: int = 1,                      # NEW: number of repeats
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        fs_local_rank: Optional[int] = None,
        work_dir: Optional[PathOrStr] = None,     # must be set for memmap
        num_threads: Optional[int] = None,
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.max_examples = max_examples
        self.num_passes = num_passes
        self.sticky_subset = True
        self.subset_seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank if rank is not None else get_global_rank()
        self.fs_local_rank = fs_local_rank if fs_local_rank is not None else get_fs_local_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.num_threads = num_threads
        self.work_dir = Path(work_dir) if work_dir is not None else None

        assert global_batch_size % self.world_size == 0
        self.device_batch_size = global_batch_size // self.world_size

        if self.max_examples is None:
            raise ValueError("Set max_examples when using num_passes>1.")

        # If you don't drop_last, make sure all ranks see the same count.
        if not self.drop_last:
            total = self.max_examples * self.num_passes
            assert total % self.world_size == 0, \
                "When drop_last=False, (max_examples * num_passes) must be divisible by world_size."

        # File path for the long concatenated indices
        self.long_indices_file: Optional[Path] = None
        if self.work_dir is not None:
            self.long_indices_file = self.work_dir / "long_indices.npy"
            self._build_and_save_long_indices()  # build once at init
        else:
            # If you really don't want files, you can adapt to in-memory, but this code path assumes memmap.
            raise ValueError("work_dir must be set for memmap indices.")

    # ------------------------- builders -------------------------

    def _get_fixed_subset(self) -> np.ndarray:
        """Deterministic fixed subset of size max_examples (only used when sticky_subset=True)."""
        N = len(self.dataset)
        assert N < np.iinfo(np.uint32).max
        base = np.arange(N, dtype=np.uint32)
        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64(self.subset_seed))
            rng.shuffle(base)
        if self.max_examples <= N:
            return base[: self.max_examples]
        # If you asked for more than N, tile deterministically
        reps = math.ceil(self.max_examples / N)
        return np.tile(base, reps)[: self.max_examples]

    def _build_one_pass(self, pass_seed: int, fixed_subset: Optional[np.ndarray]) -> np.ndarray:
        """Return exactly max_examples indices for a single pass (independently shuffled)."""
        if fixed_subset is not None:
            pool = fixed_subset.copy()
            if self.shuffle:
                rng = np.random.Generator(np.random.PCG64(pass_seed))
                rng.shuffle(pool)
            return pool

        # else: draw from whole dataset each pass
        N = len(self.dataset)
        assert N < np.iinfo(np.uint32).max
        base = np.arange(N, dtype=np.uint32)
        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64(pass_seed))
            rng.shuffle(base)

        if self.max_examples <= N:
            return base[: self.max_examples]
        reps = math.ceil(self.max_examples / N)
        return np.tile(base, reps)[: self.max_examples]

    def _build_long_indices(self) -> np.ndarray:
        """Concatenate num_passes blocks of length max_examples, each block independently shuffled."""
        fixed_subset = self._get_fixed_subset() if self.sticky_subset else None
        blocks = []
        for p in range(self.num_passes):
            seed_p = self.seed + self.epoch + p
            blocks.append(self._build_one_pass(seed_p, fixed_subset))
        indices = np.concatenate(blocks)  # length = max_examples * num_passes
        # Optional trimming when drop_last=True to keep clean full-batch alignment per rank
        if self.drop_last:
            per_rank_batch = self.device_batch_size
            # After rank stride (world_size), we want per-rank length divisible by per_rank_batch.
            # Ensuring global length divisible by (world_size * per_rank_batch) guarantees that.
            total_mult = self.world_size * per_rank_batch
            trim_len = (len(indices) // total_mult) * total_mult
            if trim_len == 0:
                raise ValueError("Too few examples to form one full batch across all ranks.")
            indices = indices[:trim_len]
        return indices

    def _build_and_save_long_indices(self):
        assert self.work_dir is not None and self.long_indices_file is not None
        if self.fs_local_rank == 0:
            log.info("Saving long concatenated indices...")
            self.long_indices_file.parent.mkdir(parents=True, exist_ok=True)
            arr = self._build_long_indices()
            mmap = np.memmap(self.long_indices_file, dtype=np.uint32, mode="w+", shape=(len(arr),))
            mmap[:] = arr
            mmap.flush()
            del mmap
            log.info("Long indices saved to '%s' (len=%d)", self.long_indices_file, len(arr))
        barrier()

    def get_long_indices(self) -> np.ndarray:
        assert self.long_indices_file is not None
        return np.memmap(self.long_indices_file, mode="r", dtype=np.uint32)  # type: ignore

    def reshuffle(self, epoch: int):
        """Rebuild the long array for a new epoch (same num_passes; fresh per-pass permutations)."""
        self.epoch = epoch
        if self.work_dir is not None:
            self._build_and_save_long_indices()

    # ------------------------- iteration -------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = self.get_long_indices()  # memmap: no big copies
        # print("INDICES=", indices, flush=True)
        # Apply global cap/offset on the long sequence (length ~ max_examples * num_passes, maybe trimmed)
        if self.start_index > 0:
            indices = indices[self.start_index :]

        # Rank stride across the WHOLE long sequence
        indices = indices[self.rank :: self.world_size]

        # If we don't drop_last, lengths are equal across ranks due to the assert in __init__.
        # Now handle worker slicing and (optional) per-worker threading.
        worker_info = torch.utils.data.get_worker_info()
        num_threads = self.num_threads

        if worker_info is not None:
            # With multiprocessing workers, avoid threading inside workers.
            # Slice by worker so each processes whole batches round-robin.
            truncated_size = self.device_batch_size * (len(indices) // self.device_batch_size)
            left_overs = indices[truncated_size + worker_info.id :: worker_info.num_workers]
            indices = (
                indices[:truncated_size]
                .reshape((-1, self.device_batch_size))[worker_info.id :: worker_info.num_workers]  # type: ignore
                .reshape((-1,))
            )
            indices = np.concatenate([indices, left_overs])
            num_threads = None  # disable intra-worker threads
        elif num_threads is None:
            # A small default if single-process data loading
            num_threads = 4

        if num_threads:
            queue_size = math.ceil(self.device_batch_size * 2 / num_threads)
            gens = []
            for i in range(num_threads):
                gen = (self._get_dataset_item(int(idx)) for idx in indices[i::num_threads])
                gens.append(threaded_generator(gen, maxsize=queue_size, thread_name=f"data thread {i}"))
            return (x for x in roundrobin(*gens))
        else:
            return (self._get_dataset_item(int(idx)) for idx in indices)

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            # always attach the source index
            return dict(**item, index=idx)
        elif dataclasses.is_dataclass(item):
            return dict(**dataclasses.asdict(item), index=idx)  # type: ignore
        else:
            # fallback: treat item as token ids / tensor
            return {"input_ids": item, "index": idx}