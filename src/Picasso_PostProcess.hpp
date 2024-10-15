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

#ifndef PICASSO_POSTPROCESS_HPP
#define PICASSO_VPOSTPROCESSHPP

#include <Picasso.hpp>

#include <mpi.h>


namespace Picasso
{
namespace PostProcess
{

// Output routine for integral conservation sums of particle quantities
// (momentum-only)
template <class ExecutionSpace, class ParticleList, class ParticleVelocity>
void particleConservation(
    MPI_Comm comm, ExecutionSpace, ParticleVelocity, const int n,
    const int write_frequency, const ParticleList particles )
{
    if ( write_frequency > 0 && n % write_frequency == 0 )
    {
        std::size_t num_p = particles.size();
        auto aosoa = particles.aosoa();

        double mass_integral = 0;
        double ke_integral = 0;
        Kokkos::parallel_reduce(
            "Picasso::Particle Conservation",
            Kokkos::RangePolicy<ExecutionSpace>( 0, num_p ),
            KOKKOS_LAMBDA( const int pn, double& mass_cont, double& ke_cont ) {
                typename ParticleList::particle_type p( aosoa.getTuple( pn ) );
                auto m_p = Picasso::get( p, Picasso::Field::Mass() );
                auto v_p = Picasso::get( p, ParticleVelocity() );
                auto u = Picasso::getField( v_p, 0 );
                auto v = Picasso::getField( v_p, 1 );
                auto w = Picasso::getField( v_p, 2 );
                mass_cont += m_p;
                double mag_v = sqrt( u * u + v * v + w * w );
                ke_cont += m_p * mag_v * mag_v / 2;
            },
            mass_integral, ke_integral );

        double global_mass = 0.0;
        double global_ke = 0.0;
        MPI_Allreduce( &mass_integral, &global_mass, 1, MPI_DOUBLE, MPI_SUM, comm );
        MPI_Allreduce( &ke_integral, &global_ke, 1, MPI_DOUBLE, MPI_SUM, comm );

        int rank;
        MPI_Comm_rank( comm, &rank );
        if ( rank == 0 )
            std::cout << "Current Particle Integrals for Step: " << n
                      << " Mass:" << global_mass
                      << " Kinetic Energy:" << global_ke << std::endl;
    }
}

// Output for integral conservation sums for grid quantities (momentum-only)
template <class ExecutionSpace, class Mesh, class ParticleVelocity>
void gridConservation( MPI_Comm comm, ExecutionSpace exec_space,
                       const std::shared_ptr<Mesh>& mesh,
                       Picasso::FieldManager<Mesh>& fm, const int n,
                       const int write_frequency, ParticleVelocity )
{
    if ( write_frequency > 0 && n % write_frequency == 0 )
    {
        auto own_index_space = mesh->localGrid()->indexSpace(
            Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
        auto m_i = fm.view( Picasso::FieldLocation::Node(),
                            Picasso::Field::Mass() );
        auto v_i = fm.view( Picasso::FieldLocation::Node(),
                            Picasso::Field::Velocity() );
        double mass_integral = 0;
        double ke_integral = 0;

        // FIX ME: Collapse the parallel reduce calls once that capability is
        // available
        Cabana::Grid::grid_parallel_reduce(
            "Picasso:MassIntegral", exec_space, own_index_space,
            KOKKOS_LAMBDA( const int i, const int j, const int k,
                           double& mass_cont ) {
                mass_cont += m_i( i, j, k, 0 );
            },
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

        double global_mass = 0.0;
        double global_ke = 0.0;
        MPI_Allreduce( &mass_integral, &global_mass, 1, MPI_DOUBLE, MPI_SUM, comm );
        MPI_Allreduce( &ke_integral, &global_ke, 1, MPI_DOUBLE, MPI_SUM, comm );

        int rank;
        MPI_Comm_rank( comm, &rank );
        if ( rank == 0 )
            std::cout << "Current Grid Integrals for Step: " << n
                      << " Mass:" << global_mass
                      << " Kinetic Energy:" << global_ke << std::endl;
    }
}

// Global conservation output for momentum problem
template <class ExecutionSpace, class Mesh, class ParticleVelocity,
          class ParticleList>
void globalConservation( MPI_Comm comm, ExecutionSpace exec_space,
                         const std::shared_ptr<Mesh>& mesh,
                         Picasso::FieldManager<Mesh>& fm, const int n,
                         const int write_frequency,
                         const ParticleList particles, ParticleVelocity p_v )
{
    particleConservation( comm, exec_space, p_v, n, write_frequency,
                          particles );

    gridConservation( comm, exec_space, mesh, fm, n, write_frequency, p_v );
}

} // end namespace PostProcess
} // end namespace Picasso

#endif // end PICASSO_POSTPROCESS_HPP
