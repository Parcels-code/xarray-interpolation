import numpy as np
import xarray as xr

from pathlib import Path

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
from typing import Callable, Any  # noqa: F811

from datetime import datetime
import json
from viztracer import VizTracer
from dataclasses import dataclass


# full dataset size is ~24Gb. To simulate particles occupying in-memory chunks (an assumption that will hold for Parcels), we set the coverage proportion to be aligned with our machine RAM
# i.e., if our usable memory is 2Gb, coverage proportion should be less than 2/24 = 0.083

DEFAULT_CHUNK_COVERAGE_PROP = 0.03  # 3% coverage

N_PARTICLES = 10**5
ONE_GB = 1024 * 1024 * 1024

OUTPUT_FOLDER = Path("output")


def get_current_time() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M-%S-%f")[:-3]


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
        use_zarr_array: bool = False,
    ):  # % of chunks that are covered
        assert "store" in open_zarr_kwargs
        assert isinstance(open_zarr_kwargs["store"], (str, Path, Store))
        self.open_zarr_kwargs = open_zarr_kwargs
        self.n_particles = n_particles
        assert 0 < chunk_coverage <= 1.0
        self.chunk_coverage = chunk_coverage
        self.postprocess_ds: Callable[[xr.Dataset], xr.Dataset] | None = None
        self._within_ctx: Any = None
        self.use_zarr_array = use_zarr_array

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

    def within_ctx(self, ctx):
        self._within_ctx = ctx

    def get_ds(self):
        ds = xr.open_zarr(**self.open_zarr_kwargs)
        if self.postprocess_ds is not None:
            return ds.pipe(self.postprocess_ds)
        return ds

    def get_zarr_array(self):
        _z_store = zarr.open(self.open_zarr_kwargs["store"], mode="r")
        assert isinstance(_z_store, zarr.Group)
        return _z_store["V_A_grid"]

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

        positions = floor_it_all(
            get_barycentric_coordinates(
                self.n_particles,
                ds,
                chunks_covered,
                chunk_size_per_dim,
                chunks_per_dim_count,
            )
        )
        return [v for _, v in positions.items()]

    @contextmanager
    def setup(self):
        self.ds = self.get_ds()
        self.zarr_arr = self.get_zarr_array()
        self.positions = self.get_particle_positions()
        with self._within_ctx:
            if self.use_zarr_array:
                yield self.zarr_arr, self.positions
            else:
                yield np.array(self.zarr_arr), self.positions


        self.ds = None
        self.zarr_arr = None
        self.positions = None

    def __repr__(self):
        return (
            f"Data(open_zarr_kwargs={repr(self.open_zarr_kwargs)}, "
            f"n_particles={self.n_particles}, "
            f"chunk_coverage={self.chunk_coverage}, "
            f"postprocess_ds={self.postprocess_ds}, "
            f"within_ctx={self._within_ctx}, "
            f"use_zarr_array={self.use_zarr_array})"
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
    def run(self, arr, positions): ...


class SingleInterpolation(Task):
    name = "single-interpolation"

    def run(self, arr, positions):
        arr[*positions]


class LoadThenSingleInterpolation(Task):
    name = "load-then-single-interpolation"

    def run(self, arr, positions):
        arr = np.array(arr)
        arr[*positions]


class TripleInterpolation(Task):
    name = "triple-interpolation"

    def run(self, arr, positions):
        arr[*positions]
        arr[*positions]
        arr[*positions]


Profiler = Callable[
    [Path, Data, Task, str | None], Path
]  # Functions that take a folder and save a profiling report


def profile_execution_time(
    folder: Path, data: Data, task: Task, file_stem: str | None = None
) -> Path:
    assert folder.is_dir()
    assert folder.exists()
    stem = (
        file_stem
        if file_stem is not None
        else f"cprofile_{task.name}_{get_current_time()}"
    )
    report = folder / f"{stem}.prof"

    with data.setup() as (ds, positions):
        prof = cProfile.Profile()
        prof.enable()
        task.run(ds, positions)
        prof.disable()
        prof.dump_stats(report)
    return report


@contextmanager
def use_zarrs_backend():
    # uses the context manager provided by the Zarr-rs python package (Rust based)
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        yield


@contextmanager
def use_single_threaded_dask():
    import dask

    with dask.config.set(scheduler="single-threaded"):
        yield


def profile_memory(
    folder: Path, data: Data, task: Task, file_stem: str | None = None
) -> Path:
    assert folder.is_dir()
    assert folder.exists()
    stem = (
        file_stem
        if file_stem is not None
        else f"memray_{task.name}_{get_current_time()}"
    )
    report = folder / f"{stem}.bin"

    with data.setup() as (ds, positions):
        with memray.Tracker(report):
            task.run(ds, positions)
    return report


def run_viztracer(
    folder: Path, data: Data, task: Task, file_stem: str | None = None
) -> Path:
    assert folder.is_dir()
    assert folder.exists()
    stem = (
        file_stem
        if file_stem is not None
        else f"viztracer_{task.name}_{get_current_time()}"
    )
    report = folder / f"{stem}.json"

    with data.setup() as (ds, positions):
        with open(report, "w") as f:
            with VizTracer(output_file=f):
                task.run(ds, positions)
    return report


@dataclass
class TestCase:
    profiler: Profiler
    task: Task
    data: Data
    file_stem: str | None = None


@dataclass
class Workspace:
    folder: Path
    test_cases: list[TestCase]

    def run_test_cases(self):
        if self.folder.exists():
            msg = (
                f"Cannot run test cases. Output folder '{self.folder}' already exists."
            )
            raise RuntimeError(msg)
        self.folder.mkdir()
        summary = {"test_cases": []}  # type: ignore[var-annotated]
        for test_case in self.test_cases:
            report = test_case.profiler(
                self.folder, test_case.data, test_case.task, test_case.file_stem
            )
            summary["test_cases"].append(
                {
                    "data": repr(test_case.data),
                    "task": test_case.task.name,
                    "profiler": test_case.profiler.__name__ + "()",
                    "profile_path": str(report.relative_to(self.folder)),
                }
            )
        with open(self.folder / "summary.json", "w") as f:
            json.dump(summary, f)


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    numpy_data, zarr_data = [
        Data(  # ~1Gb uncompressed
            {"store": "datasets/ds_2d_left_agrid_small.zarr", "consolidated": False},
            n_particles=N_PARTICLES,
            chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
            use_zarr_array=use_zarr_array,
        )
        for use_zarr_array in [False, True]
    ]
    zarr_data_with_cache = Data(  # ~1Gb uncompressed
        {
            "store": create_cache_store(
                zarr.storage.LocalStore("datasets/ds_2d_left_agrid_small.zarr"),
                2 * ONE_GB,
            ),
            "consolidated": False,
        },
        n_particles=N_PARTICLES,
        chunk_coverage=DEFAULT_CHUNK_COVERAGE_PROP,
        use_zarr_array=True,
    )

    Workspace(
        folder=OUTPUT_FOLDER / "compare-from-raw-zarr",
        test_cases=[
            # 1 - Interpolation on already loaded numpy data
            TestCase(
                profile_execution_time,
                SingleInterpolation(),
                numpy_data,
                file_stem="case1",
            ),
            # 2 - Interpolation on zarr data using zarr array
            TestCase(
                profile_execution_time,
                SingleInterpolation(),
                zarr_data,
                file_stem="case2",
            ),
            # 3 - Load from zarr array then interpolate
            TestCase(
                profile_execution_time,
                LoadThenSingleInterpolation(),
                zarr_data,
                file_stem="case3",
            ),
            # 4 - triple interpolation on already loaded numpy data
            TestCase(
                profile_execution_time,
                TripleInterpolation(),
                numpy_data,
                file_stem="case4",
            ),
            # 5 - triple interpolation on zarr data
            TestCase(
                profile_execution_time,
                TripleInterpolation(),
                zarr_data,
                file_stem="case5",
            ),
            # 6 - triple interpolation on zarr data with 2 Gb cache
            TestCase(
                profile_execution_time,
                TripleInterpolation(),
                zarr_data_with_cache,
                file_stem="case6",
            ),
        ],
    ).run_test_cases()
