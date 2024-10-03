# Picasso

Picasso is a performance portable library for particle-in-cell simulations
which are utilized in application including plasmas physics and fluids and 
solid mechanics. Picasso provides a range of interpolation schemes between
particles and grids, both in order of the scheme and the type of scheme 
(FLIP, APIC, PolyPIC), embedded free-surface tracking, extensive batched 
linear algebra capabilties, as well as utilities for managing field data
and simplifying on-node parallel execution.


## Building Picasso

### Dependencies

Picasso's main dependency for building is [Cabana](https://github.com/ECP-copa/Cabana).
The instructions for building Cabana can be found in the [Cabana wiki](https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions)
In addition to the core Cabana dependencies of Kokkos and CMake, Cabana must 
be built with the Cabana::grid library option turned on.

Picasso Required Dependencies
| Dependency |Required | Details |
| ---------- -------- | ------  |
|[CMake](https://cmake.org/download/)      | Yes     | Build system |
|[Kokkos](https://github.com/kokkos/kokkos)    | Yes      | Portable on-node parallelism |
|[Cabana](https://github.com/ECP-copa/Cabana) | Yes | Performance-portable particle and grid library |
| ---------- -------- | ------  |

In addition to these requried dependencies, the build of Cabana is required 
to have certain options enabled. (Cabana::Grid, MPI?)

There are additional optional dependencies for Picasso that depend on the 
use case desired from Picasso or hardware type for the system building 
Picasso.

Picasso Optional Dependencies imported from Cabana
| Dependency |Required | CMake Variable | Details | Required in or inherited from upstream dependency |
| ---------- -------- | -------------- | -------  | ----- |
|MPI | No | |Message Passing Interface | inherited from Cabana MPI build |
| ArborX | No | Picasso_ENABLE_ARBORX |(Experimental) performance-portable geometric search (required for level-set) | N/A (is Cabana arborx entirely seperate from Picasso?)
| HDF5 | No | (defined in Cabana?) | Particle I/O | inherited from Cabana build with HDF5 |
| Silo | No | Picasso_ENABLE_SILO | Particle I/O | Cabana required to be built with SILO and Picasso built with SILO |
| JSON | NO | (?) | JSON variable input |
| GTest | NO | Picasso_ENABLE_TESTING |Unit test Framework | 
| CUDA | NO | | Programming model for NVIDIA GPUs | Inherited from Cabana and Kokkos |
| HIP | NO | | Programming model for AMD GPUs | Inherited from Cabana and Kokkos |
| SYCL | NO | | Programming model for Intel GPUs | Inherited from Cabana and Kokkos |

For Picasso-related questions you can open a GitHub issue to interact with the
developers.

## Contributing

We encourage you to contribute to Picasso! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.

## Citing

If you use Picasso in your work, please cite the [JOSS article](CITATION.bib).
Also consider citing the appropriate [release](https://doi.org/10.5281/zenodo.2558368).

## License

Picasso is distributed under an [open source 3-clause BSD license](LICENSE).
