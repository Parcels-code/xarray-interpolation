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
| 1.          | interp on already loaded data. Only profile interp     | 0.004              |
| 2.          | interp on already loaded data. Profile load and interp | 0.742              |
| 3.          | interp using dask (i.e., no pre-fetching)              | 2.448              |
| 4.          | triple interp using dask (i.e., no pre-fetching)       | 7.932              |
| 5.          | triple interp using dask with LRU cache Zarr Store     | 7.302              |

See [Appendix A](#appendix-a-full-profiling-results-from-cases) for the full results.

Evidently working with numpy data (1) is the fastest. Comparing (2) and (3), you can see
that loading _the full data_ and then fetching the values is significantly faster than
using Dask to fetch these values. Looking at (3) and (4), we can see that the interpolation
with Dask scales linearly (alluding to no caching of values or computation graph), and
comparing (4) and (5) shows that the experimental Zarr CacheStore has a minor performance improvement.

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

## Output

Summary for saved_outputs/compare-for-xarray-folks/case1.prof
===================
Wed Jun 10 12:09:38 2026    saved_outputs/compare-for-xarray-folks/case1.prof

         3967 function calls (3879 primitive calls) in 0.004 seconds

   Ordered by: cumulative time
   List reduced from 226 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    0.004    0.004 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:190(run)
        1    0.000    0.000    0.004    0.004 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:2748(isel)
        1    0.000    0.000    0.004    0.004 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:2903(_isel_fancy)
        9    0.000    0.000    0.003    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/variable.py:1108(isel)
        9    0.000    0.000    0.003    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/variable.py:815(__getitem__)
        9    0.000    0.000    0.003    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)
        9    0.000    0.000    0.003    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:466(__getitem__)
        2    0.000    0.000    0.001    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1703(_vindex_get)
        2    0.001    0.001    0.001    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/nputils.py:168(__getitem__)
        4    0.000    0.000    0.001    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:2027(_oindex_get)


Summary for saved_outputs/compare-for-xarray-folks/case2.prof
===================
Wed Jun 10 12:09:39 2026    saved_outputs/compare-for-xarray-folks/case2.prof

         1656694 function calls (1588418 primitive calls) in 0.742 seconds

   Ordered by: cumulative time
   List reduced from 847 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  723/722    0.001    0.000    5.512    0.008 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
  723/722    0.002    0.000    3.592    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
     2099    0.016    0.000    3.099    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
     2099    0.004    0.000    2.650    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
   1240/3    0.003    0.000    1.266    0.422 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/threading.py:337(wait)
      2/1    0.001    0.000    0.703    0.703 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:197(run)
      2/1    0.000    0.000    0.700    0.700 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      2/1    0.001    0.000    0.700    0.700 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      2/1    0.000    0.000    0.699    0.699 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      2/1    0.000    0.000    0.699    0.699 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)


Summary for saved_outputs/compare-for-xarray-folks/case3.prof
===================
Wed Jun 10 12:09:42 2026    saved_outputs/compare-for-xarray-folks/case3.prof

         8761329 function calls (8583489 primitive calls) in 2.448 seconds

   Ordered by: cumulative time
   List reduced from 1013 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
19350/19349    0.005    0.000    1.098    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
      2/1    0.006    0.003    1.041    1.041 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:190(run)
      2/1    0.000    0.000    1.035    1.035 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      2/1    0.000    0.000    1.035    1.035 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      2/1    0.001    0.000    1.035    1.035 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      2/1    0.000    0.000    1.034    1.034 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      2/1    0.004    0.002    1.034    1.034 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      2/1    0.030    0.015    1.030    1.030 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
19350/19349    0.028    0.000    0.855    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
       65    0.000    0.000    0.791    0.012 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)


Summary for saved_outputs/compare-for-xarray-folks/case4.prof
===================
Wed Jun 10 12:09:50 2026    saved_outputs/compare-for-xarray-folks/case4.prof

         26183413 function calls (25648945 primitive calls) in 7.932 seconds

   Ordered by: cumulative time
   List reduced from 1007 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.021    0.010    6.422    6.422 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:205(run)
      4/3    0.000    0.000    4.392    1.464 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      4/3    0.000    0.000    4.391    1.464 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      4/3    0.002    0.000    4.390    1.463 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      4/3    0.000    0.000    4.389    1.463 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      4/3    0.016    0.004    4.345    1.448 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      4/3    0.087    0.022    4.328    1.443 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
58048/58047    0.016    0.000    3.791    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
58048/58047    0.073    0.000    2.952    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
        3    0.000    0.000    2.808    0.936 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:2748(isel)


Summary for saved_outputs/compare-for-xarray-folks/case5.prof
===================
Wed Jun 10 12:09:57 2026    saved_outputs/compare-for-xarray-folks/case5.prof

         26183819 function calls (25650830 primitive calls) in 7.302 seconds

   Ordered by: cumulative time
   List reduced from 1030 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.009    0.004    5.867    5.867 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:205(run)
      4/3    0.000    0.000    4.249    1.416 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:772(compute)
      4/3    0.000    0.000    4.249    1.416 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/dataset.py:531(load)
      4/3    0.002    0.000    4.249    1.416 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/namedarray/daskmanager.py:80(compute)
      4/3    0.000    0.000    4.247    1.416 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/base.py:601(compute)
      4/3    0.007    0.002    4.206    1.402 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/threaded.py:62(get)
      4/3    0.082    0.020    4.198    1.399 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:382(get_async)
58048/58047    0.016    0.000    3.968    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/dask/local.py:140(queue_get)
58048/58047    0.079    0.000    3.158    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/queue.py:177(get)
      195    0.000    0.000    2.427    0.012 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/xarray/core/indexing.py:1179(apply_indexer)



