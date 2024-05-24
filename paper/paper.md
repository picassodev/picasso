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
physics to solid mechanics and computer graphics. 
`Picasso` is a performance portable library for PIC simulations, developed
throughout the Exascale Computing Project (ECP) [@ecp:2020]. `Picasso` builds
on the `Cabana` library [@cabana:2022] for particle and structured grid
methods, which in turn extends the `Kokkos` library for on-node parallelism
across hardware architectures [@kokkos:2022] and `MPI` for scalable multi-node
communication. `Picasso` provides interpolation schemes for particle-to-grid
and grid-to-particle updates, embedded free surface tracking, and robust
linear algebra support for particle and grid operations.

# Statement of need


# Library capabilities

## Interpolation


## Free surfaces


## Examples and performance testing


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
