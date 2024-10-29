# Picasso

Picasso is a performance portable library for particle-in-cell (PIC) simulations
which are used in applications including plasmas physics and fluid/solid
mechanics. Picasso provides a range of interpolation schemes between
particles and grids, both in order of the scheme and the type
(FLIP, APIC, PolyPIC), embedded free-surface tracking, extensive batched 
linear algebra capabilties, as well as utilities for managing field data
and simplifying parallel execution.

## Dependencies

Picasso's main dependency is [Cabana](https://github.com/ECP-copa/Cabana).
The instructions for building Cabana can be found in the [Cabana wiki](https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions)
Cabana requires the Kokkos library and must be built with the optional Cabana::Grid
subpackage enabled (which in turn requires MPI).

Picasso Dependencies
| Dependency |Required | CMake Variable | Details |
| ---------- | ------- | -------------  | ------  |
|[CMake](https://cmake.org/download/)      | Yes     | Build system |
|[Kokkos](https://github.com/kokkos/kokkos)    | Yes      | Portable on-node parallelism |
|[Cabana](https://github.com/ECP-copa/Cabana) | Yes | Performance-portable particle and grid library |
|MPI | Yes | Message Passing Interface |
|[JSON](https://github.com/nlohmann/json) | Yes | JSON input files |
|ArborX | No | Picasso_REQUIRE_ARBORX | Performance-portable geometric search (required for level-set) | N/A |
|GTest | No | Picasso_REQUIRE_TESTING |Unit test Framework | N/A |

In addition to these required dependencies, note that Cabana must be built with
`Cabana_ENABLE_GRID=ON` (enabled by default) and will need further options as noted below for some additional capabilities.

There are additional optional dependencies for Picasso that depend on the 
use case desired or hardware type on the system.

Picasso dependencies imported from Cabana. Note that the `REQUIRE` variables enforce that those options are
available for Cabana at configuration, but are not necessary.
| Dependency |Required | CMake Variable | Details |
| ----------| -------- | -------------- | -------  |
| HDF5 | No | Cabana_REQUIRE_HDF5 | Particle I/O |
| Silo | No | Cabana_REQUIRE_SILO | Particle I/O |
| CUDA | No | Cabana_REQUIRE_CUDA | Programming model for NVIDIA GPUs |
| HIP  | No | Cabana_REQUIRE_HIP  | Programming model for AMD GPUs |
| SYCL | No | Cabana_REQUIRE_SYCL | Programming model for Intel GPUs |

## Building Picasso

To build Picasso, clone the repository via
`git clone https://github.com/picassodev/picasso.git`. Ensure that
you have an install of Cabana with the grid build option
enabled ([Cabana Build Details](https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions)

A [Cabana Docker container](https://github.com/ECP-copa/Cabana/pkgs/container/cabana)
is maintained to facilitate development with most optional dependencies.

From the the source directory, run the following script to create
a build directory for Picasso, configure the Picasso build in that
directory, and build Picasso in that directory

```
# Change directory as needed
export CABANA_DIR='pwd'/Cabana/build/install

cd picasso
cmake \
  -B build
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="$CABANA_DIR" \
  -D CMAKE_INSTALL_PREFIX=install \
  -D Picasso_ENABLE_TESTING=ON
cmake --build build
cmake --install build
```

## Testing Picasso

To test your Picasso install, from the build directory of a Picasso
build run with testing enabled, run the `ctest` command and ensure
that all tests list as passing

## Contributing

We encourage you to contribute to Picasso! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.

## Citing

If you use Picasso in your work, please cite the appropriate [release](https://doi.org/10.5281/zenodo.8309476).

## License

Picasso is distributed under an [open source 3-clause BSD license](LICENSE).
