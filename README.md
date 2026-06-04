# point interpolation in Xarray

**Abstract**: This repo explores the performance of random point cloud interpolation within chunked Xarray datasets for use in Lagrangian particle tracking.

### Background

Xarray is well suited for array computations within Eulerian oceanography, where operations are done on whole arrays. The support for [Dask arrays](https://docs.dask.org/en/latest/array.html) and [Cubed arrays](https://cubed-dev.github.io/cubed/) allow for the definition and lazy execution of array computations. In the case of Dask, a directed acyclic computation graph (DAG) is created, splitting the computation into various steps which can be distributed based on a "scheduler" during execution. This technique allows for splitting of computation, and horizontal scaling enabling for the processing of larger-than-memory datasets.

In Lagrangian oceanography, where particles are moving around in a flow field, we are interested in the data surrounding individual points moving in the gridded data. In the case of structured grid data using simple bi-linear interpolation, this is the $2^n$ field values that surround the point - where $n$ is the dimensionality of the flow field. e.g.,

- $2^2=4$ data values for 2D data (2 spatial dimensions, 0 temporal dimensions)
- $2^4=16$ data values for 4D data (3 spatial dimensions, 1 temporal dimensions)

During interpolation these values are combined together based on the position of the particle in the grid, by calculating the barycentric coordinates of the particle within the model grid.

In our simulations, we would like to run with hundreds of thousands of particles.

- In a sufficiently spun up simulation, the particles for the large part fully explore the grid and can be anywhere.
- Some assumptions can be made about particle positions. For example, between timesteps a particle occupies the same or a neighbouring grid cell. This assumption allows for grid searching performance improvements.
- This computation is parallelisable on a particle level (assuming no particle-particle interation), however is not parallelisable in time. The location of the particle, and the field locations it samples, is directly dependent on the previous computations.

All of this to illusrtate that the data access patterns within Lagrangian oceanography is fundamentally different to that of Eulerian oceanography, and different to a lot of the image-data processing techniques that are explored in the Xarray and Pangeo communities.

By exploring the performance of Xarray when it comes to the interpolation of point-cloud data within data-cubes, we can hopefully measure how feasible it is to use Xarray in it's current state for integration in Lagrangian simulation frameworks. In the case of poor perforamnce, this profiling will hopefully show (a) what changes can be made to Xarray to enable this usecase, or (b) how Xarray users interested in this use case can use Xarray's current abstractions to achieve acceptable performance for this problem.

### ...
