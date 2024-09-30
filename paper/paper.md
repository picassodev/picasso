---
title: 'Picasso: Performance Portable Particle-in-Cell'
tags:
  - C++
  - Kokkos
  - Cabana
  - particle-in-cell
  - material point method
authors:
  - name: Stuart Slattery^[corresponding author]
    orcid: 0000-0003-0103-888X
    affiliation: 1
  - name: Samuel Temple Reeve
    orcid: 0000-0002-4250-9476
    affiliation: 1
  - name: Austin Isner
    affiliation: 1
  - name: Kwitae Chong
    affiliation: 1
  - name: Lance Bullerwell
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory, Oak Ridge, TN, USA
    index: 1
date: 12 April 2024
bibliography: paper.bib
---

# Summary

Particle-in-cell (PIC) methods are used is disparate fields, from plasma
physics and solid mechanics to computer graphics.
`Picasso` is a performance portable library for PIC simulations, developed
through the Exascale Computing Project (ECP) [@ecp:2020]. `Picasso` builds
on the `Cabana` library [@cabana:2022] for particle and structured grid
methods, which in turn extends the `Kokkos` library for on-node parallelism
across hardware architectures [@kokkos:2022] and uses `MPI` for scalable distributed
simulation. `Picasso` provides interpolation schemes for particle-to-grid
and grid-to-particle updates, embedded free surface tracking, and robust
linear algebra support for particle and grid operations. This separation of
concerns results in a parallelism layer, a simulation motif (and distributed
communication) layer, and finally a PIC algorithm and utility layer, all enabling
performant user-level physics simulations.

# Statement of need

Computational predictions are increasingly necessary for answering scientific
and engineering questions; PIC methods are a robust option for simulations of highly
dynamic systems, including plasmas, fluids, and solid materials undergoing severe
deformation. As these needs continue to expand, striking the proper balance of
computational performance, portability across hardware architectures, and programmer
productivity requires substantial effort [@ppp:2021].
Existing frameworks for PIC simulations are increasingly written with performance
portability in mind, but focused much more on the plasma physics domain [@vpic, @ippl, @pumipic].
In addition, there are no known current libraries to support state-of-the-art PIC algorithms
and related complex particle operations [@apic, @polypic].

# Library capabilities

`Picasso` introduces various classes and supporting types for enabling a high-level specification of field data and PIC algorithms, with the intent to hide implementation details such as memory spaces and allocation, halo communication patterns, and device versus host access. The design principles of performance transparency and PIC domain abstraction thus allows the algorithm designer/user implementor to focus only on the field data definitions and their interdependencies within various sequential "operations", allowing for a natural identification. `Picasso` defines two necessary types for facilitating data management and communication: the `FieldManager` and `GridOperator`'s, discussed in the following sections.

At the core of `Picasso`'s domain model is the concept of a `FieldLayout`, which contains a location and a field tag. A location can be one of any structured mesh entity (`Node`, `Cell`, `Face<D>`, `Edge<D>`), where `D` is a specific dimension `I`,`J`, or `K`, or a `Particle`. A tag is any type with a string-like `label()` static member function. The complete specification of a `FieldLayout` uniquely defines a field as known to the `FieldManager`, while also allowing the same field tag (e.g. `Temperature`) to be associated with multiple locations (e.g. both `Node` and `Particle`).
### Field Manager
The `Picasso::FieldManager` is a single collection wrapper type for storing unique handles to field layouts in an `unordered_map`. The `FieldManager` itself is separately initialized by passing it to each `GridOperator::setup()` function.

### Grid Operators
The `Picasso::GridOperator` encapsulates all of the field management work associated with parallel grid operations. Both particle-centric loops are supported (for operations like P2G and G2P) as well as entity centric loops for more traditional discrete grid operations.
A `GridOperator` is defined over a given mesh type and a user defines the fields which must be gathered on the grid to apply the operator and the fields which will be scattered on the grid after the operation is complete. All distributed parallel work is managed by these dependencies - the user needs to only write the body of the operator kernel to be applied at each data point (e.g. each particle or mesh entity).
The order of the dependency templates (if there are any dependencies) is defined above in the documentation for `GridOperatorDependencies`.

### Particle Initialization
`Picasso` provides two modes for initializing particles on a background grid: `InitUniform` and `InitRandom`. In addition to requiring parameters (particles per dimension for `InitUniform` or particles per cell for `InitRandom`), a user-defined predicate functor must also be provided that returns true or false based on certain criteria of the candidate particle, such as position. The design for a custom user-hook during the creation phase allows for flexible and arbitrary particle initialization.

### Batched Linear Algebra
Matrix-vector and matrix-matrix operations are nearly ubiquitous in particle-to-grid and grid-to-particle mappings, function space projections, and other common operations wherein support for writing concise vectorial expressions is benefitial for both code readability, as well as exposing algorithmic parallelism.  The `Picasso` library also implements kernel-level dense linear algebra operations in a corresponding `LinearAlgebra` namespace using a combination of expression templates for lazy evaluation and data structures to hold intermediates for eager evaluations when necessary. The general concept in the `Picasso` implementation for lazy vs. eager evaluations alleviates the consideration of additional performance factors such as overhead incurred from excessive copies or total operation counts in otherwise unoptimized code. Using built-in operator overloading with support for expression templates, users can build complex nested tensorial expressions that are a mixture of both eager and lazy evaluations. For example, if `A` and `B` are NxN and `x` is length N:
```cpp
  auto C = 0.5 * (A + ~A) * B + (x * ~x);
```
where the returned `C` is an NxN matrix, `~` is the transpose operator, and the `*` is an outer-product operation in the last term.

`Picasso` provides an interface for various supported linear algebra types defined in `FieldTypes`: `Scalar`, `Vector`, `Matrix`, as well as specialized support for higher-rank `Tensor3`, `Tensor4`, and `Quaternion` types. Field tags need only derive from these types in order to make use of `Picasso` linear algebra features. Although all basic operations on vectors and matrices are implemented, several specialized operations are also available, including matrix determinant, inverse, exponential, LU, and SVD decompositions, higher-order tensor contractions, and quaternion-matrix conjugation.

## Interpolation


## Level Sets and Boundaries
The `Picasso::LevelSet` and `Picasso::ParticleLevelSet` are fast parallel implementations of grid-based signed distance fields (SDF), which are relevant to level-set based approaches in interface reconstruction and boundary tracking. In particular, the `ParticleLevelSet` class relies on functionality from [ArborX::BVH](https://github.com/arborx/ArborX) (bounding volume hiererarchy) for nearest point queries to compute a level set signed distance estimate, $\phi_i^{0}$, on the mesh using analytic spherical particle level sets $\phi_p$ as proxy. A parallel Hopf-Lax redistancing algorithm is also implemented that properly reinitializes the level set estimate $\phi$ into a final SDF ($|\nabla\phi_i|=1$). Additionally, the `Picasso::MarchingCubes` namespace provides methods and data structures for triangulating the resulting SDF. The computed mesh facets may be serialized to disk in STL format or used for further processing, for instance in convex hull initialization or embedded free surface tracking algorithms.

## Examples and performance testing


## Future work

Picasso also provides a clear place to include further algorithmic development in the PIC
field, e.g. @powerpic, @ipic.

# Acknowledgments

This work was supported by the Exascale Computing Project (17-SC-20-SC), a
collaborative effort of the U.S. DOE Office of Science and the NNSA.

This manuscript has been authored by UT-Battelle, LLC under Contract No.
DE-AC05-00OR22725 with the U.S. Department of Energy (DOE). The publisher, by
accepting the article for publication, acknowledges that the United States
Government retains a non-exclusive, paid-up, irrevocable, world-wide license to
publish or reproduce the published form of this manuscript, or allow others to
do so, for United States Government purposes. The DOE will provide public
access to these results of federally sponsored research in accordance with the
DOE Public Access Plan.

This research used resources of the Oak Ridge Leadership Computing Facility
(OLCF), supported by DOE under contract DE-AC05-00OR22725.

# References
