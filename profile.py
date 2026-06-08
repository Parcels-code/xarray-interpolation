import numpy as np
import xarray as xr

from pathlib import Path

import zarr
from contextlib import contextmanager
import math
import cProfile
import memray
from abc import ABC, abstractmethod
from typing import Any


CHUNK_COVERAGE_PROP = 0.2
N_PARTICLES = 10**5


def get_barycentric_coordinates(n, ds, n_active_chunks, chunk_sizes, chunk_counts):
    dims = list(ds.sizes.keys())
    counts_tuple = tuple(chunk_counts[d] for d in dims)
    assert n_active_chunks > 0
    assert n > 0

    # Map linear chunk indices → per-dim chunk indices
    active_chunks = np.arange(min(n_active_chunks, int(np.prod(counts_tuple))))
    chunk_indices = np.unravel_index(active_chunks, counts_tuple)
    coords = {}
    for dim, dim_chunk_indices in zip(dims, chunk_indices, strict=True):
        chunk_size = chunk_sizes[dim]
        lo = dim_chunk_indices * chunk_size
        hi = np.minimum((dim_chunk_indices + 1) * chunk_size, ds.sizes[dim])
        coord = np.random.uniform(size=lo.size) * (hi - lo) + lo
        if coord.size >= n:
            coords[dim] = coord[:n]
        else:
            coords[dim] = np.concat((coord, np.random.uniform(size=n - coord.size)))

    return coords


def floor_it_all(positions):
    return {k: np.floor(v).astype(int) for k, v in positions.items()}


def wrap_in_da(positions):
    return {k: xr.DataArray(arr, dims=("points")) for k, arr in positions.items()}


class Data:
    def __init__(
        self,
        open_zarr_kwargs: dict[str, Any],
        n_particles: int,
        chunk_coverage: float,
    ):  # % of chunks that are covered
        assert "store" in open_zarr_kwargs
        assert isinstance(open_zarr_kwargs["store"], (str, Path))
        self.open_zarr_kwargs = open_zarr_kwargs
        self.n_particles = n_particles
        assert 0 < chunk_coverage <= 1.0
        self.chunk_coverage = chunk_coverage

    def get_ds(self):
        return xr.open_zarr(**self.open_zarr_kwargs)

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
            f"chunk_coverage={self.chunk_coverage})"
        )


default_data = Data(
    {"store": "datasets/ds_2d_left_agrid.zarr", "consolidated": False},
    n_particles=N_PARTICLES,
    chunk_coverage=CHUNK_COVERAGE_PROP,
)


class Task(ABC):
    name: str

    @abstractmethod
    def run(self, ds: xr.Dataset, positions: xr.Dataset): ...


class SingleInterpolation(Task):
    name = "Single interpolation"

    def run(self, ds: xr.Dataset, positions: xr.Dataset):
        ds.isel(positions).compute()


class TripleInterpolation(Task):
    name = "Triple interpolation"

    def run(self, ds: xr.Dataset, positions: xr.Dataset):
        ds.isel(positions).compute()
        ds.isel(positions).compute()
        ds.isel(positions).compute()


def execution_profile_task(data: Data, task: Task):
    with data.setup() as (ds, positions):
        prof = cProfile.Profile()
        prof.enable()
        task.run(ds, positions)
        prof.disable()
        prof.dump_stats("cprofile.prof")
    return


def memory_profile_task(data: Data, task: Task):
    with data.setup() as (ds, positions):
        with memray.Tracker("memray.bin"):
            task.run(ds, positions)
    return


if __name__ == "__main__":
    data = default_data
    task = SingleInterpolation()
    execution_profile_task(data, task)
    memory_profile_task(data, task)
