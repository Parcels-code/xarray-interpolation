import numpy as np
import xarray as xr

from pathlib import Path

import dask  # noqa: F401
import zarr
from zarr.abc.store import Store
from contextlib import contextmanager
import math
import cProfile
import zarr.storage
import memray
from abc import ABC, abstractmethod
from typing import Any
import copy
from typing import Callable

import time
import json
from dataclasses import dataclass

# dask.config.set(scheduler="single-threaded")

# full dataset size is ~24Gb. To simulate particles occupying in-memory chunks (an assumption that will hold for Parcels), we set the coverage proportion to be aligned with our machine RAM
# i.e., if our usable memory is 2Gb, coverage proportion should be less than 2/24 = 0.083

DEFAULT_CHUNK_COVERAGE_PROP = 0.03  # 3% coverage

N_PARTICLES = 10**5
ONE_GB = 1024 * 1024

OUTPUT_FOLDER = Path("output")


def get_current_time() -> str:
    t = time.localtime()
    return time.strftime("%Y%m%d-%H%M-%S", t)


def get_barycentric_coordinates(n, ds, n_active_chunks, chunk_sizes, chunk_counts):
    dims = list(ds.sizes.keys())
    counts_tuple = tuple(chunk_counts[d] for d in dims)
    assert n_active_chunks > 0
    assert n > 0

    # set numpy seed (this is useful since repeated calls to this will simulate particles being in the same chunks/cells, meaning we can effectively use caching and represent real-world scenarios)
    rng = np.random.default_rng(seed=22)

    # Map linear chunk indices → per-dim chunk indices
    active_chunks = np.arange(min(n_active_chunks, int(np.prod(counts_tuple))))
    chunk_indices = np.unravel_index(active_chunks, counts_tuple)
    coords = {}
    for dim, dim_chunk_indices in zip(dims, chunk_indices, strict=True):
        chunk_size = chunk_sizes[dim]
        lo = dim_chunk_indices * chunk_size
        hi = np.minimum((dim_chunk_indices + 1) * chunk_size, ds.sizes[dim])
        coord = rng.uniform(size=lo.size) * (hi - lo) + lo
        if coord.size >= n:
            coords[dim] = coord[:n]
        else:
            coords[dim] = np.concat((coord, rng.uniform(size=n - coord.size)))

    return coords


def floor_it_all(positions):
    return {k: np.floor(v).astype(int) for k, v in positions.items()}


def wrap_in_da(positions):
    return {k: xr.DataArray(arr, dims=("points")) for k, arr in positions.items()}


def create_cache_store(source_store: Store, max_size: int):
    from zarr.experimental.cache_store import CacheStore

    cache_store = zarr.storage.MemoryStore()
    return CacheStore(store=source_store, cache_store=cache_store, max_size=max_size)


class Data:
    def __init__(
        self,
        open_zarr_kwargs: dict[str, Any],
        n_particles: int,
        chunk_coverage: float,
    ):  # % of chunks that are covered
        assert "store" in open_zarr_kwargs
        assert isinstance(open_zarr_kwargs["store"], (str, Path, Store))
        self.open_zarr_kwargs = open_zarr_kwargs
        self.n_particles = n_particles
        assert 0 < chunk_coverage <= 1.0
        self.chunk_coverage = chunk_coverage
        self.postprocess_ds: Callable[[xr.Dataset], xr.Dataset] | None = None

    def __copy__(self):
        ret = type(self)(
            self.open_zarr_kwargs.copy(),
            self.n_particles,
            self.chunk_coverage,
        )
        ret.postprocess_ds = copy.copy(self.postprocess_ds)
        return ret

    def then(self, *, postprocess_ds: Callable[[xr.Dataset], xr.Dataset]):
        if self.postprocess_ds is not None:
            raise NotImplementedError(
                "self.postprocess_ds is already set. Chaining of post-processing is not yet implemented"
            )
        ret = copy.copy(self)
        ret.postprocess_ds = postprocess_ds
        return ret

    def get_ds(self):
        ds = xr.open_zarr(**self.open_zarr_kwargs)
        if self.postprocess_ds is not None:
            return ds.pipe(self.postprocess_ds)
        return ds

    def get_particle_positions(self):
        ds = self.ds
        chunks_coverage = self.chunk_coverage
        _z_store = zarr.open(self.open_zarr_kwargs["store"], mode="r")
        assert isinstance(_z_store, zarr.Group)
        _chunk_meta = _z_store["V_A_grid"]
        assert isinstance(_chunk_meta, zarr.Array)
        chunk_size_per_dim = dict(zip(ds.sizes.keys(), _chunk_meta.chunks))
        chunks_per_dim_count = {
            d: math.ceil(ds.sizes[d] / chunk_size_per_dim[d]) for d in ds.sizes
        }
        total_chunks = _chunk_meta.nchunks

        chunks_covered = int(chunks_coverage * total_chunks)

        positions = wrap_in_da(
            floor_it_all(
                get_barycentric_coordinates(
                    self.n_particles,
                    ds,
                    chunks_covered,
                    chunk_size_per_dim,
                    chunks_per_dim_count,
                )
            )
        )
        return positions

    @contextmanager
    def setup(self):
        self.ds = self.get_ds()
        self.positions = self.get_particle_positions()
        yield self.ds, self.positions
        self.ds = None
        self.positions = None

    def __repr__(self):
        return (
            f"Data(open_zarr_kwargs={repr(self.open_zarr_kwargs)}, "
            f"n_particles={self.n_particles}, "
            f"chunk_coverage={self.chunk_coverage}, "
            f"postprocess_ds={self.postprocess_ds})"
        )


DEFAULT_DATA = Data(
    {"store": "datasets/ds_2d_left_agrid.zarr", "consolidated": False},
    n_particles=N_PARTICLES,
    chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
)  # ~24Gb uncompressed

DEFAULT_DATA_SMALL = Data(
    {"store": "datasets/ds_2d_left_agrid_small.zarr", "consolidated": False},
    n_particles=N_PARTICLES,
    chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
)  # ~1Gb uncompressed


class Task(ABC):
    name: str

    @abstractmethod
    def run(self, ds: xr.Dataset, positions: xr.Dataset): ...


class SingleInterpolation(Task):
    name = "single-interpolation"

    def run(self, ds: xr.Dataset, positions: xr.Dataset):
        ds.isel(positions).compute()


class LoadThenSingleInterpolation(Task):
    name = "load-then-single-interpolation"

    def run(self, ds: xr.Dataset, positions: xr.Dataset):
        ds = ds.load()
        ds.isel(positions)


class TripleInterpolation(Task):
    name = "triple-interpolation"

    def run(self, ds: xr.Dataset, positions: xr.Dataset):
        ds.isel(positions).compute()
        ds.isel(positions).compute()
        ds.isel(positions).compute()


Profiler = Callable[
    [Path, Data, Task], Path
]  # Functions that take a folder and save a profiling report


def profile_execution_time(folder: Path, data: Data, task: Task) -> Path:
    assert folder.is_dir()
    assert folder.exists()
    report = folder / f"cprofile_{task.name}_{get_current_time()}.prof"

    with data.setup() as (ds, positions):
        prof = cProfile.Profile()
        prof.enable()
        task.run(ds, positions)
        prof.disable()
        prof.dump_stats(report)
    return report


def profile_memory(folder: Path, data: Data, task: Task) -> Path:
    assert folder.is_dir()
    assert folder.exists()
    report = folder / f"memray_{task.name}_{get_current_time()}.bin"

    with data.setup() as (ds, positions):
        with memray.Tracker(report):
            task.run(ds, positions)
    return report


@dataclass
class Workspace:
    folder: Path
    test_cases: list[tuple[Profiler, Task, Data]]

    def run_test_cases(self):
        if self.folder.exists():
            msg = (
                f"Cannot run test cases. Output folder '{self.folder}' already exists."
            )
            raise RuntimeError(msg)
        self.folder.mkdir()
        summary = {"test_cases": []}  # type: ignore[var-annotated]
        for profiler, task, data in self.test_cases:
            report = profiler(self.folder, data, task)
            summary["test_cases"].append(
                {
                    "data": repr(data),
                    "task": task.name,
                    "profiler": profiler.__name__ + "()",
                    "profile_path": str(report.relative_to(self.folder)),
                }
            )
        with open(self.folder / "summary.json", "w") as f:
            json.dump(summary, f)


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Workspace(
    #     folder=OUTPUT_FOLDER / "single-interpolation",
    #     test_cases=[
    #         (profile_execution_time, SingleInterpolation(), DEFAULT_DATA),
    #         (profile_memory, SingleInterpolation(), DEFAULT_DATA),
    #     ],
    # ).run_test_cases()

    # default_data_with_cache = Data(
    #     {
    #         "store": create_cache_store(
    #             zarr.storage.LocalStore("datasets/ds_2d_left_agrid.zarr"),
    #             2 * ONE_GB,
    #         ),
    #         "consolidated": False,
    #     },
    #     n_particles=N_PARTICLES,
    #     chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
    # )
    # Workspace(
    #     folder=OUTPUT_FOLDER / "compare-zarr-cache-single-call",
    #     test_cases=[
    #         (profile_execution_time, SingleInterpolation(), DEFAULT_DATA),
    #         (profile_execution_time, SingleInterpolation(), default_data_with_cache),
    #     ],
    # ).run_test_cases()

    # Workspace(
    #     folder=OUTPUT_FOLDER / "compare-zarr-cache-triple-call",
    #     test_cases=[
    #         (profile_execution_time, TripleInterpolation(), DEFAULT_DATA),
    #         (profile_execution_time, TripleInterpolation(), default_data_with_cache),
    #     ],
    # ).run_test_cases()

    def load_dataset(ds: xr.Dataset) -> xr.Dataset:
        return ds.load()

    Workspace(
        folder=OUTPUT_FOLDER / "compare-for-xarray-folks",
        test_cases=[
            # 1 - interp on already loaded data. Only profile interp
            (
                profile_execution_time,
                SingleInterpolation(),
                DEFAULT_DATA_SMALL.then(postprocess_ds=load_dataset),
            ),
            # 2 - interp on already loaded data. Profile load and interp
            (profile_execution_time, LoadThenSingleInterpolation(), DEFAULT_DATA_SMALL),
            # 3 - interp using dask (i.e., no pre-fetching)
            (profile_execution_time, SingleInterpolation(), DEFAULT_DATA_SMALL),
            # 4 - triple interp using dask (i.e., no pre-fetching)
            (profile_execution_time, TripleInterpolation(), DEFAULT_DATA_SMALL),
            # 5 - triple interp using dask with LRU cache Zarr Store
            (
                profile_execution_time,
                TripleInterpolation(),
                Data(
                    {
                        "store": create_cache_store(
                            zarr.storage.LocalStore(
                                "datasets/ds_2d_left_agrid_small.zarr"
                            ),
                            2 * ONE_GB,
                        ),
                        "consolidated": False,
                    },
                    n_particles=N_PARTICLES,
                    chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
                ),
            ),
        ],
    ).run_test_cases()
