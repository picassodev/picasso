/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_CONSERVATION_HPP
#define PICASSO_CONSERVATION_HPP

#include <Picasso.hpp>

#include <mpi.h>

namespace Picasso
{
// Output routine for integral conservation sums of particle quantities
// (momentum-only)
template <class ExecutionSpace, class ParticleList, class ParticleVelocity>
void particleConservation( MPI_Comm comm, ExecutionSpace,
                           const ParticleList particles, ParticleVelocity,
                           double& global_mass, double& global_ke,
                           double& global_pe, const int pe_dir,
                           const double zero_ground,
                           const Kokkos::Array<double, 3> gravity )
{
    std::size_t num_p = particles.size();
    auto aosoa = particles.aosoa();

    double mass_integral = 0;
    double ke_integral = 0;
    double pe_integral = 0;
    Kokkos::parallel_reduce(
        "Picasso::Particle Conservation",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_p ),
        KOKKOS_LAMBDA( const int pn, double& mass_cont, double& ke_cont,
                       double& pe_cont ) {
            typename ParticleList::particle_type p( aosoa.getTuple( pn ) );
            auto m_p = Picasso::get( p, Picasso::Field::Mass() );
            auto v_p = Picasso::get( p, ParticleVelocity() );
            auto x_p = Picasso::get( p, Picasso::Field::LogicalPosition<3>() );
            auto u = Picasso::getField( v_p, 0 );
            auto v = Picasso::getField( v_p, 1 );
            auto w = Picasso::getField( v_p, 2 );
            mass_cont += m_p;
            double mag_v = sqrt( u * u + v * v + w * w );
            ke_cont += m_p * mag_v * mag_v / 2;
            pe_cont += m_p * Kokkos::abs( gravity[pe_dir] ) *
                       ( x_p( pe_dir ) - zero_ground );
        },
        mass_integral, ke_integral, pe_integral );

    MPI_Allreduce( &mass_integral, &global_mass, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &ke_integral, &global_ke, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &pe_integral, &global_pe, 1, MPI_DOUBLE, MPI_SUM, comm );
}

// Output for integral conservation sums for grid quantities (momentum-only)
template <class ExecutionSpace, class Mesh>
void gridConservation( MPI_Comm comm, ExecutionSpace exec_space,
                       const std::shared_ptr<Mesh>& mesh,
                       Picasso::FieldManager<Mesh>& fm, double& global_mass,
                       double& global_ke, double& global_pe, const int pe_dir,
                       const double zero_ground,
                       const Kokkos::Array<double, 3> gravity )
{
    auto own_index_space = mesh->localGrid()->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    auto m_i =
        fm.view( Picasso::FieldLocation::Node(), Picasso::Field::Mass() );
    auto v_i = fm.view( Picasso::FieldLocation::Node(),
                        Picasso::Field::Velocity<3>() );
    auto x_i = fm.view( Picasso::FieldLocation::Node(),
                        Picasso::Field::PhysicalPosition<3>() );
    double mass_integral = 0;
    double ke_integral = 0;
    double pe_integral = 0;

    // FIXME: Collapse the parallel reduce calls once that capability is
    // available
    Cabana::Grid::grid_parallel_reduce(
        "Picasso:MassIntegral", exec_space, own_index_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       double& mass_cont ) { mass_cont += m_i( i, j, k, 0 ); },
        mass_integral );

    Cabana::Grid::grid_parallel_reduce(
        "Picasso:KineticEnergyIntegral", exec_space, own_index_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       double& ke_cont ) {
            double mag_v = sqrt( v_i( i, j, k, 0 ) * v_i( i, j, k, 0 ) +
                                 v_i( i, j, k, 1 ) * v_i( i, j, k, 1 ) +
                                 v_i( i, j, k, 2 ) * v_i( i, j, k, 2 ) );
            ke_cont += m_i( i, j, k, 0 ) * mag_v * mag_v / 2;
        },
        ke_integral );

    Cabana::Grid::grid_parallel_reduce(
        "Picasso:PotentialEnergyIntegral", exec_space, own_index_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       double& pe_cont ) {
            pe_cont += m_i( i, j, k, 0 ) * Kokkos::abs( gravity[pe_dir] ) *
                       ( x_i( i, j, k, pe_dir ) - zero_ground );
        },
        pe_integral );

    MPI_Allreduce( &mass_integral, &global_mass, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &ke_integral, &global_ke, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &pe_integral, &global_pe, 1, MPI_DOUBLE, MPI_SUM, comm );
}

} // end namespace Picasso

#endif // end PICASSO_CONSERVATION_HPP
