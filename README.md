# Picasso

Picasso is a performance portable library for particle-in-cell simulations
which are utilized in application including plasmas physics and fluids and 
solid mechanics. Picasso provides a range of interpolation schemes between
particles and grids, both in order of the scheme and the type of scheme 
(FLIP, APIC, PolyPIC), embedded free-surface tracking, extensive batched 
linear algebra capabilties, as well as utilities for managing field data
and simplifying on-node parallel execution.


## Dependencies

Picasso's main dependency for building is [Cabana](https://github.com/ECP-copa/Cabana).
The instructions for building Cabana can be found in the [Cabana wiki](https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions)
In addition to the core Cabana dependencies of Kokkos and CMake, Cabana must 
be built with the Cabana::grid library option turned on.

Picasso Required Dependencies
| Dependency |Required | Details |
| ---------- |-------- | ------  |
|[CMake](https://cmake.org/download/)      | Yes     | Build system |
|[Kokkos](https://github.com/kokkos/kokkos)    | Yes      | Portable on-node parallelism |
|[Cabana](https://github.com/ECP-copa/Cabana) | Yes | Performance-portable particle and grid library |
|MPI | Yes | Message Passing Interface |
| [JSON]() | Yes | JSON variable input |

In addition to these requried dependencies, the build of Cabana is required 
to have certain options enabled, Specifically `Cabana_ENABLE_GRID=ON`.

There are additional optional dependencies for Picasso that depend on the 
use case desired from Picasso or hardware type for the system building 
Picasso.

Picasso Optional Dependencies imported from Cabana
| Dependency |Required | CMake Variable | Details | Required in or inherited from upstream dependency |
| ----------| -------- | -------------- | -------  | ----- |
| ArborX | No | Picasso_REQUIRE_ARBORX | Performance-portable geometric search (required for level-set) | N/A |
| HDF5 | No | Cabana_REQUIRE_HDF5 (defined in Cabana cmake)| Particle I/O | inherited from Cabana build with HDF5 |
| Silo | No | Picasso_REQUIRE_SILO | Particle I/O | Cabana required to be built with SILO and Picasso built with SILO |
| GTest | NO | Picasso_REQUIRE_TESTING |Unit test Framework | N/A |
| CUDA | NO | | Programming model for NVIDIA GPUs | Inherited from Cabana and Kokkos | Yes |
| HIP | NO | | Programming model for AMD GPUs | Inherited from Cabana and Kokkos | Yes |
| SYCL | NO | | Programming model for Intel GPUs | Inherited from Cabana and Kokkos | Yes |

For Picasso-related questions you can open a GitHub issue to interact with the
developers.

## Building Picasso

To build picasso, clone the repository via
`git clone https://github.com/picassodev/picasso.git`. Ensure that
you have an install of Cabana with the MPI and grid build options
enabled ([Cabana Build Details](https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions)
Alternatively to using a cmake build of Cabana, you can also use
a [Cabana Container build](https://github.com/ECP-copa/Cabana/pkgs/container/cabana)

From the the source directory, run the following script to create
a build directory for picasso, configure the picasso build in that
directory, and build Picasso in that directory

```
export CABANA_DIR='pwd'/Cabana/build/install
export PICASSO_DIR='pwd'/Cabana/build/install

cd picasso
mkdir build
cd build
cmake \
  -D CMAKE_BUILD_TYPE="RELEASE" \
  -D CMAKE_PREFIX_PATH="$CABANA_DIR \
  -D CMAKE_INSTALL_PREFIX="$PICASSO_DIR \
  -D Picasso_ENABLE_TESTING=ON \
  ..
make install
```

## Testing Picasso install

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
