import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import xarray as xr

    import dask.array as da

    from functools import reduce
    import operator
    from pathlib import Path

    from zarr.storage import LocalStore
    import marimo as mo
    import zarr
    import math


@app.cell(hide_code=True)
def data_generation():
    X = 1000
    Y = 1000
    Z = 50
    T = 30

    TIME = xr.date_range("2000", "2001", T)

    def prod(seq):
        return reduce(operator.mul, seq, 1)

    def _rotated_curvilinear_grid():
        XG = np.arange(X)
        YG = np.arange(Y)
        LON, LAT = np.meshgrid(XG, YG)

        angle = -np.pi / 24
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        # rotate the LON and LAT grids
        LON, LAT = np.einsum("ji, mni -> jmn", rotation, np.dstack([LON, LAT]))

        return xr.Dataset(
            {
                "data_g": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "data_c": (
                    ["time", "ZC", "YC", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_C_grid": (
                    ["time", "ZG", "YC", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_C_grid": (
                    ["time", "ZG", "YG", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
            },
            coords={
                "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
                "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
                "XC": (["XC"], XG + 0.5, {"axis": "X"}),
                "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
                "ZG": (
                    ["ZG"],
                    np.arange(Z),
                    {"axis": "Z", "c_grid_axis_shift": -0.5},
                ),
                "ZC": (
                    ["ZC"],
                    np.arange(Z) + 0.5,
                    {"axis": "Z"},
                ),
                "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
                "time": (["time"], TIME, {"axis": "T"}),
                "lon": (
                    ["YG", "XG"],
                    LON,
                    {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
                ),
                "lat": (
                    ["YG", "XG"],
                    LAT,
                    {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
                ),
            },
        )

    def random_dask_array(shape, scaling=1):
        return da.random.uniform(scaling, size=prod(shape)).reshape(shape)

    def _cartesion_to_polar(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

    def _polar_to_cartesian(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def _unrolled_cone_curvilinear_grid():
        # Not a great unrolled cone, but this is good enough for testing
        # you can use matplotlib pcolormesh to plot
        XG = np.arange(X)
        YG = np.arange(Y) * 0.25

        pivot = -10, 0
        LON, LAT = np.meshgrid(XG, YG)

        new_lon_lat = []

        min_lon = np.min(XG)
        for lon, lat in zip(LON.flatten(), LAT.flatten(), strict=True):
            r, _ = _cartesion_to_polar(lon - pivot[0], lat - pivot[1])
            _, theta = _cartesion_to_polar(min_lon - pivot[0], lat - pivot[1])
            theta *= 1.2
            r *= 1.2
            lon, lat = _polar_to_cartesian(r, theta)
            new_lon_lat.append((lon + pivot[0], lat + pivot[1]))

        new_lon, new_lat = zip(*new_lon_lat, strict=True)
        LON, LAT = (
            np.array(new_lon).reshape(LON.shape),
            np.array(new_lat).reshape(LAT.shape),
        )

        return xr.Dataset(
            {
                "data_g": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "data_c": (
                    ["time", "ZC", "YC", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_C_grid": (
                    ["time", "ZG", "YC", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_C_grid": (
                    ["time", "ZG", "YG", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
            },
            coords={
                "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
                "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
                "XC": (["XC"], XG + 0.5, {"axis": "X"}),
                "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
                "ZG": (
                    ["ZG"],
                    np.arange(Z),
                    {"axis": "Z", "c_grid_axis_shift": -0.5},
                ),
                "ZC": (
                    ["ZC"],
                    np.arange(Z) + 0.5,
                    {"axis": "Z"},
                ),
                "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
                "time": (["time"], TIME, {"axis": "T"}),
                "lon": (
                    ["YG", "XG"],
                    LON,
                    {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
                ),
                "lat": (
                    ["YG", "XG"],
                    LAT,
                    {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
                ),
            },
        )

    datasets = {
        "2d_left_rotated": _rotated_curvilinear_grid(),
        "ds_2d_left": xr.Dataset(  # MITgcm indexing style
            {
                "data_g": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "data_c": (
                    ["time", "ZC", "YC", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_A_grid": (
                    ["time", "ZG", "YG", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "U_C_grid": (
                    ["time", "ZG", "YC", "XG"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
                "V_C_grid": (
                    ["time", "ZG", "YG", "XC"],
                    random_dask_array(scaling=5, shape=(T, Z, Y, X)),
                ),
            },
            coords={
                "XG": (
                    ["XG"],
                    2 * np.pi / X * np.arange(0, X),
                    {"axis": "X", "c_grid_axis_shift": -0.5},
                ),
                "XC": (["XC"], 2 * np.pi / X * (np.arange(0, X) + 0.5), {"axis": "X"}),
                "YG": (
                    ["YG"],
                    2 * np.pi / (Y) * np.arange(0, Y),
                    {"axis": "Y", "c_grid_axis_shift": -0.5},
                ),
                "YC": (
                    ["YC"],
                    2 * np.pi / (Y) * (np.arange(0, Y) + 0.5),
                    {"axis": "Y"},
                ),
                "ZG": (
                    ["ZG"],
                    np.arange(Z),
                    {"axis": "Z", "c_grid_axis_shift": -0.5},
                ),
                "ZC": (
                    ["ZC"],
                    np.arange(Z) + 0.5,
                    {"axis": "Z"},
                ),
                "lon": (["XG"], 2 * np.pi / X * np.arange(0, X)),
                "lat": (["YG"], 2 * np.pi / (Y) * np.arange(0, Y)),
                "depth": (["ZG"], np.arange(Z)),
                "time": (["time"], TIME, {"axis": "T"}),
            },
        ),
        "2d_left_unrolled_cone": _unrolled_cone_curvilinear_grid(),
    }

    def save(ds: xr.Dataset, path: str, chunks: dict) -> None:
        """Save dataset to zarr with specified chunking."""
        store = LocalStore(path)
        ds.chunk(chunks).to_zarr(store, mode="w", encoding=None, consolidated=False)
        size_mb = sum(ds[v].nbytes for v in ds.data_vars) / 1e6
        print(f"  {path}  dims={dict(ds.sizes)}  ~{size_mb:.0f} MB uncompressed")

    dataset_path = "datasets/ds_2d_left_agrid.zarr"
    print("Generating ds_2d_left...")
    if Path(dataset_path).exists():
        print(f"Dataset {dataset_path} already exists")
    else:
        save(
            datasets["ds_2d_left"][["U_A_grid", "V_A_grid"]],
            dataset_path,
            {"time": 15, "XG": 40, "YG": 40, "ZG": 8},
        )

    print("\nDone.")
    mo.md(
        """
        # Generate test data
        Uses an example dataset in Parcels to generate zarr test data.
        """
    )
    return


@app.cell
def _():
    ds = xr.open_zarr("datasets/ds_2d_left_agrid.zarr", consolidated=False)
    return (ds,)


@app.cell(hide_code=True)
def _(ds):
    _z_store = zarr.open("datasets/ds_2d_left_agrid.zarr", mode="r")
    assert isinstance(_z_store, zarr.Group)
    _chunk_meta = _z_store["V_A_grid"]
    assert isinstance(_chunk_meta, zarr.Array)
    chunk_size_per_dim = dict(zip(ds.sizes.keys(), _chunk_meta.chunks))
    chunks_per_dim_count = {
        d: math.ceil(ds.sizes[d] / chunk_size_per_dim[d]) for d in ds.sizes
    }
    total_chunks = _chunk_meta.nchunks

    chunks_covered = mo.ui.slider(
        0, total_chunks, step=1, label="Chunks covered: ", value=total_chunks
    )

    n = mo.ui.slider(start=2, stop=7, step=1, label="Number of particles (10^n): ")
    return (
        chunk_size_per_dim,
        chunks_covered,
        chunks_per_dim_count,
        n,
        total_chunks,
    )


@app.cell(hide_code=True)
def _(chunks_covered, n, total_chunks):
    n_particles = 10**n.value
    mo.vstack(
        [
            n,
            mo.md(f"$10^{n.value}={n_particles}$ particles"),
            chunks_covered,
            mo.md(f"**{chunks_covered.value}** / {total_chunks} chunks covered"),
        ]
    )
    return (n_particles,)


@app.cell
def _():
    zarr.open("datasets/ds_2d_left_agrid.zarr", mode="r")["V_A_grid"].chunks
    return


@app.cell
def _(
    chunk_size_per_dim,
    chunks_covered,
    chunks_per_dim_count,
    ds,
    n_particles,
):
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

    positions = wrap_in_da(
        floor_it_all(
            get_barycentric_coordinates(
                n_particles,
                ds,
                chunks_covered.value,
                chunk_size_per_dim,
                chunks_per_dim_count,
            )
        )
    )
    return (positions,)


@app.cell
def _(positions):
    positions
    return


@app.cell
def _(ds, positions):
    queried = ds.isel(positions)
    return (queried,)


@app.cell
def _(queried):
    data = queried["V_A_grid"].data
    data
    # data.visualize(filename="transpose.svg")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
