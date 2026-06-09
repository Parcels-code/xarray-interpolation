# Profiling analysis

We roughly want to have the following:

- loop start:
  - Physical particle positions are looked up in the grid to calculate the relevant index positions
    - Once known, can be "isel"ed using Xarray:
    - When this is done in Xarray, it goes to the respective underlying Zarr/Netcdf chunks and retrieve these values (using Dask)
    - These chunks which were retrieved from are stored in memory using an LRU cache, skipping potentially expensive trips to disk in future
  - isel results are combined to perform the interpolation
  - particle positions are updated

Here we construct 4 test cases to explore how isel behaviour works in Xarray with point cloud data where particles are seeding sparsely (i.e., not full coverage of the Zarr dataset). I think there is a lot more to explore here - this is just a starting point.

Here we only measure execution time using CProfile. We haven't run memory profiles yet, though that can be easily done. Also note that CProfile (N.B. as far as I know) isn't able to fully introspect Dask workings, so the resulting profile is somewhat limited.

See [`xarray-profile.py`](./xarray-profile.py) for the exact code to reproduce these results.

## Data and parameters

- Array dataset: (small) Zarr dataset (~1.31Mb per chunk)

```
>>> _z_store["V_A_grid"].info_complete()
Type               : Array
Zarr format        : 3
Data type          : Float64(endianness='little')
Fill value         : nan
Shape              : (30, 50, 200, 200)
Chunk shape        : (15, 8, 40, 40)
Order              : C
Read-only          : True
Store type         : LocalStore
Filters            : ()
Serializer         : BytesCodec(endian=<Endian.little: 'little'>)
Compressors        : (ZstdCodec(level=0, checksum=False),)
No. bytes          : 480000000 (457.8M)
No. bytes stored   : 455087483 (434.0M)
Storage ratio      : 1.1
```

- Params
  - no. particles = 10^5
  - Chunk coverage = 0.03 (i.e., 3% chunk coverage - particles reside in 3% of the chunks of the dataset)

## Cases

| Case number | Description                                            | Execution time (s) |
| :---------- | :----------------------------------------------------- | :----------------- |
| 1.          | interp on already loaded data. Only profile interp     | 0.003              |
| 2.          | interp on already loaded data. Profile load and interp | 0.730              |
| 3.          | interp using dask (i.e., no pre-fetching)              | 2.413              |
| 4.          | triple interp using dask (i.e., no pre-fetching)       | 7.260              |
| 5.          | triple interp using dask with LRU cache Zarr Store     | 7.735              |

See [Appendix A](#appendix-a-full-profiling-results-from-cases) for the full results.

Evidently working with numpy data (1) is the fastest. Comparing (2) and (3), you can see
that loading _the full data_ and then fetching the values is significantly faster than
using Dask to fetch these values. Looking at (3) and (4), we can see that the interpolation
with Dask scales linearly (alluding to no caching of values or computation graph), and
comparing (4) and (5) shows that the experimental Zarr CacheStore has no performance improvement.

All of this indicates that the major slowdown is Dask itself - which is also evident when opening the profile outputs in `snakeviz`.
Looking at Viztracer output, it's clear that a lot of time is spent waiting for the thread lock (since we're working with a
Dask threadpool executor). Switching to a single threaded executor via the config actually makes our code in (3), (4), (5) run _faster_
bringing the execution time for (3) a bit below that of (2).

# Appendix A: Full profiling results from cases

The full results are saved in [./saved_outputs](./saved_outputs) (note that the file names have been changed to match the cases above).

You can use `viztracer saved_outputs` to explore the outputs, or use the following script to print output which are pasted below:

```python
import pstats
from pstats import SortKey

for i in [1,2,3,4,5]:
    path = f'saved_outputs/compare-for-xarray-folks/case{i}.prof'
    p = pstats.Stats(path)
    print(f"Summary for {path}")
    print("===================")

    p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
```

## Summaries

### Case1

```
Summary for saved_outputs/compare-for-xarray-folks/case1.prof
===================
Tue Jun  9 13:45:49 2026    saved_outputs/compare-for-xarray-folks/case1.prof

         3967 function calls (3879 primitive calls) in 0.003 seconds

   Ordered by: cumulative time
   List reduced from 226 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.003    0.003 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:187(run)
        1    0.000    0.000    0.002    0.002 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:2748(isel)
        1    0.000    0.000    0.002    0.002 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:2903(_isel_fancy)
        9    0.000    0.000    0.002    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/variable.py:1108(isel)
        9    0.000    0.000    0.002    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/variable.py:815(__getitem__)
        9    0.000    0.000    0.002    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)
        9    0.000    0.000    0.002    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:466(__getitem__)
        2    0.000    0.000    0.001    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1703(_vindex_get)
        2    0.001    0.000    0.001    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/nputils.py:168(__getitem__)
        4    0.000    0.000    0.001    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:2027(_oindex_get)


<pstats.Stats object at 0x1057fdd30>

```

### Case2

```
Summary for saved_outputs/compare-for-xarray-folks/case2.prof
===================
Tue Jun  9 13:45:50 2026    saved_outputs/compare-for-xarray-folks/case2.prof

         1656074 function calls (1587140 primitive calls) in 0.730 seconds

   Ordered by: cumulative time
   List reduced from 847 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  723/722    0.001    0.000    4.816    0.007 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
     2080    0.019    0.000    3.177    0.002 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
  723/722    0.002    0.000    2.870    0.004 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
     2080    0.004    0.000    2.457    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
   1241/3    0.003    0.000    1.254    0.418 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/threading.py:337(wait)
    15919    0.006    0.000    0.768    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/events.py:92(_run)
      2/1    0.001    0.001    0.692    0.692 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:194(run)
      2/1    0.000    0.000    0.689    0.689 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      2/1    0.000    0.000    0.689    0.689 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      2/1    0.000    0.000    0.689    0.689 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)


<pstats.Stats object at 0x1057f1f90>
```

### Case3

```
Summary for saved_outputs/compare-for-xarray-folks/case3.prof
===================
Tue Jun  9 13:45:52 2026    saved_outputs/compare-for-xarray-folks/case3.prof

         8733671 function calls (8555789 primitive calls) in 2.413 seconds

   Ordered by: cumulative time
   List reduced from 1013 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
19350/19349    0.005    0.000    1.232    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
      2/1    0.005    0.003    0.982    0.982 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:187(run)
      2/1    0.000    0.000    0.976    0.976 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      2/1    0.000    0.000    0.976    0.976 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      2/1    0.002    0.001    0.976    0.976 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      2/1    0.000    0.000    0.975    0.975 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      2/1    0.004    0.002    0.974    0.974 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      2/1    0.030    0.015    0.970    0.970 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
19350/19349    0.025    0.000    0.865    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
       65    0.000    0.000    0.799    0.012 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)


<pstats.Stats object at 0x1057f20d0>
```

### Case4

```
Summary for saved_outputs/compare-for-xarray-folks/case4.prof
===================
Tue Jun  9 13:45:59 2026    saved_outputs/compare-for-xarray-folks/case4.prof

         26153577 function calls (25620421 primitive calls) in 7.260 seconds

   Ordered by: cumulative time
   List reduced from 1007 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.011    0.005    5.829    5.829 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:202(run)
      4/3    0.000    0.000    4.235    1.412 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      4/3    0.000    0.000    4.234    1.411 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      4/3    0.002    0.000    4.234    1.411 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      4/3    0.000    0.000    4.232    1.411 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      4/3    0.014    0.004    4.191    1.397 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      4/3    0.086    0.021    4.176    1.392 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
58048/58047    0.015    0.000    3.532    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
58048/58047    0.076    0.000    3.032    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
      195    0.000    0.000    2.368    0.012 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)


<pstats.Stats object at 0x104f6bce0>
```

### Case5

```
Summary for saved_outputs/compare-for-xarray-folks/case5.prof
===================
Tue Jun  9 13:46:07 2026    saved_outputs/compare-for-xarray-folks/case5.prof

         26113928 function calls (25581215 primitive calls) in 7.735 seconds

   Ordered by: cumulative time
   List reduced from 1027 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.013    0.006    6.275    6.275 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:202(run)
      4/3    0.000    0.000    4.285    1.428 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      4/3    0.000    0.000    4.285    1.428 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      4/3    0.003    0.001    4.284    1.428 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      4/3    0.000    0.000    4.281    1.427 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
58048/58047    0.015    0.000    4.254    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
      4/3    0.010    0.002    4.238    1.413 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      4/3    0.081    0.020    4.228    1.409 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
58048/58047    0.076    0.000    2.857    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
      324    0.006    0.000    2.812    0.009 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)


<pstats.Stats object at 0x1057fa8b0>
```

### Comparison
