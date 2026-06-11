# Profiling analysis
## Problem overview

See [../compare-for-xarray-folks/README.md](../compare-for-xarray-folks/README.md).

## Case overview

Here we construct 6 test cases to explore what happens if we completely bypass Xarray (and hence the Zarr scheduler) and just do raw interpolation with Numpy and Zarr.

Here we only measure execution time using CProfile. We haven't run memory profiles yet, though that can be easily done.

See [`xarray-profile.py`](./xarray-profile.py) (specifically at commit 451dc96e39387a3917cc7ee9864b04484016367f) for the exact code to reproduce these results.

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



| Case number | Description                                       | Cumulative time for `.run()` (s) |
| :---------- | :------------------------------------------------ | :------------------ |
| 1.          | Interpolation on already loaded numpy data        | 0.001               |
| 2.          | Interpolation on zarr data using zarr array       | 0.005               |
| 3.          | Load from zarr array then interpolate             | 0.260               |
| 4.          | triple interpolation on already loaded numpy data | 0.001               |
| 5.          | triple interpolation on zarr data                 | 0.022               |
| 6.          | triple interpolation on zarr data with 2 Gb cache | 0.019               |

See [Appendix A](#appendix-a-full-profiling-results-from-cases) for the full results.

Note that case (1) is faster than the equivalent case in [../compare-for-xarray-folks/README.md](../compare-for-xarray-folks/README.md)
- this is because we've cut out the Xarray overhead (this applies to all the test cases of course). Comparing (1) and (2) we see that
indexing on already loaded data is faster than on non-loaded data (as exected). Comparing (2) and (3) however we can see that interpolating
with Zarr (2) is much quicker than loading the data and then indexing (3) indicating that Zarr is only loading chunks with particles in them.
Comparing case (5) and (6) indices a slight performance improvement from using a cachestore.

This performance is much better than the Dask approach profiled before, and is performance that we would be very happy with in Parcels. Working
from raw Zarr data arrays, however, does mean that users aren't able to take advantage of lazy computations provided by dask - meaning that their data
on disk will need to be in a format Parcels understands.


# Appendix A: Full profiling results from cases

You can use `snakeviz saved_outputs` to explore the outputs, or use the following script to print output which are pasted below:

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


```

Summary for output/compare-from-raw-zarr/case1.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case1.prof

         2 function calls in 0.001 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    0.001    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:201(run)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


Summary for output/compare-from-raw-zarr/case2.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case2.prof

         9351 function calls (9168 primitive calls) in 0.009 seconds

   Ordered by: cumulative time
   List reduced from 350 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       34    0.000    0.000    0.021    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
       34    0.000    0.000    0.015    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
      184    0.000    0.000    0.009    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/events.py:92(_run)
      184    0.000    0.000    0.009    0.000 {method 'run' of '_contextvars.Context' objects}
       90    0.000    0.000    0.006    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/common.py:106(run)
       34    0.000    0.000    0.005    0.000 {method 'control' of 'select.kqueue' objects}
      2/1    0.000    0.000    0.005    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:201(run)
      2/1    0.000    0.000    0.005    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:2429(__getitem__)
      2/1    0.000    0.000    0.005    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/indexing.py:1326(__getitem__)
      2/1    0.000    0.000    0.005    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:3354(get_coordinate_selection)


Summary for output/compare-from-raw-zarr/case3.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case3.prof

         304867 function calls (300026 primitive calls) in 0.276 seconds

   Ordered by: cumulative time
   List reduced from 321 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      787    0.008    0.000    0.794    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
     6551    0.002    0.000    0.433    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/events.py:92(_run)
     6551    0.003    0.000    0.417    0.000 {method 'run' of '_contextvars.Context' objects}
      787    0.002    0.000    0.399    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
     3490    0.007    0.000    0.284    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/common.py:106(run)
      787    0.011    0.000    0.274    0.000 {method 'control' of 'select.kqueue' objects}
      2/1    0.000    0.000    0.260    0.260 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:208(run)
    353/1    0.053    0.000    0.260    0.260 {built-in method numpy.array}
      2/1    0.000    0.000    0.255    0.255 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:2410(__array__)
      2/1    0.000    0.000    0.207    0.207 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:2429(__getitem__)


Summary for output/compare-from-raw-zarr/case4.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case4.prof

         2 function calls in 0.001 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    0.001    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:216(run)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


Summary for output/compare-from-raw-zarr/case5.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case5.prof

         28004 function calls (27482 primitive calls) in 0.026 seconds

   Ordered by: cumulative time
   List reduced from 345 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       98    0.001    0.000    0.067    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
       98    0.000    0.000    0.053    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
       98    0.001    0.000    0.023    0.000 {method 'control' of 'select.kqueue' objects}
      2/1    0.000    0.000    0.022    0.022 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:216(run)
      4/3    0.000    0.000    0.022    0.007 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:2429(__getitem__)
      4/3    0.000    0.000    0.022    0.007 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/indexing.py:1326(__getitem__)
      4/3    0.000    0.000    0.022    0.007 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:3354(get_coordinate_selection)
      4/3    0.000    0.000    0.017    0.006 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/sync.py:123(sync)
      4/3    0.000    0.000    0.017    0.006 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/concurrent/futures/_base.py:257(wait)
      554    0.000    0.000    0.016    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/events.py:92(_run)


Summary for output/compare-from-raw-zarr/case6.prof
===================
Thu Jun 11 14:35:08 2026    output/compare-from-raw-zarr/case6.prof

         24551 function calls (24209 primitive calls) in 0.022 seconds

   Ordered by: cumulative time
   List reduced from 367 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       79    0.001    0.000    0.049    0.001 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/base_events.py:1977(_run_once)
       79    0.000    0.000    0.034    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/selectors.py:540(select)
       79    0.001    0.000    0.020    0.000 {method 'control' of 'select.kqueue' objects}
      2/1    0.000    0.000    0.019    0.019 /Users/Hodgs004/coding/repos/xarray-interpolation/xarray-profile.py:216(run)
      4/3    0.000    0.000    0.019    0.006 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:2429(__getitem__)
      4/3    0.000    0.000    0.019    0.006 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/indexing.py:1326(__getitem__)
      4/3    0.000    0.000    0.018    0.006 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/array.py:3354(get_coordinate_selection)
      481    0.000    0.000    0.015    0.000 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/asyncio/events.py:92(_run)
      481    0.000    0.000    0.015    0.000 {method 'run' of '_contextvars.Context' objects}
      4/3    0.000    0.000    0.014    0.005 /Users/Hodgs004/coding/repos/xarray-interpolation/.pixi/envs/default/lib/python3.14/site-packages/zarr/core/sync.py:123(sync)


```
