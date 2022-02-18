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

#ifndef PICASSO_PARTICLEINIT_HPP
#define PICASSO_PARTICLEINIT_HPP

#include <Picasso_Types.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Initialization type tags.
struct InitUniform
{
};
struct InitRandom
{
};

//---------------------------------------------------------------------------//
// Filter out empty particles that weren't created.
template <class CreationView, class ParticleAoSoA, class ExecutionSpace>
void filterEmpties( const ExecutionSpace& exec_space,
                    const int local_num_create,
                    const CreationView& particle_created,
                    ParticleAoSoA& particles )
{
    using memory_space = typename CreationView::memory_space;

    // Determine the empty particle positions in the compaction zone.
    int num_particles = particles.size();
    Kokkos::View<int*, memory_space> empties(
        Kokkos::ViewAllocateWithoutInitializing( "empties" ),
        std::min( num_particles - local_num_create, local_num_create ) );
    Kokkos::parallel_scan(
        "Picasso::ParticleInit::FindEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, local_num_create ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( !particle_created( i ) )
            {
                if ( final_pass )
                {
                    empties( count ) = i;
                }
                ++count;
            }
        } );

    // Compact the list so the it only has real particles.
    Kokkos::parallel_scan(
        "Picasso::ParticleInit::RemoveEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, local_num_create,
                                             num_particles ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( particle_created( i ) )
            {
                if ( final_pass )
                {
                    particles.setTuple( empties( count ),
                                        particles.getTuple( i ) );
                }
                ++count;
            }
        } );
    particles.resize( local_num_create );
    particles.shrinkToFit();
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a random number of particles in each cell given an
  initialization functor.

  \tparam ParticleListType The type of particle list to initialize.

  \tparam InitFunctor Initialization functor type. See the documentation below
  for the create_functor parameter on the signature of this functor.

  \param Initialization type tag.

  \param particles_per_cell The number of particles to sample each cell with.

  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double px[3],
                          typename ParticleAoSoA::tuple_type& particle );

  \param particle_list The list of particles to populate. This will be filled
  with particles and resized to a size equal to the number of particles
  created.
*/
template <class ParticleListType, class InitFunctor, class ExecutionSpace>
void initializeParticles( InitRandom, const ExecutionSpace& exec_space,
                          const int particles_per_cell,
                          const InitFunctor& create_functor,
                          ParticleListType& particle_list )
{
    // Particle type.
    using particle_type = typename ParticleListType::particle_type;

    // Get the local grid.
    const auto& local_grid = *( particle_list.mesh().localGrid() );

    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Create a random number generator.
    uint64_t seed =
        global_grid.blockId() + ( 19383747 % ( global_grid.blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    rnd_type pool;
    pool.init( seed, owned_cells.size() );

    // Get the particles.
    auto& particles = particle_list.aosoa();

    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int num_particles = particles_per_cell * owned_cells.size();
    particles.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, ExecutionSpace>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    Kokkos::parallel_reduce(
        "Picasso::ParticleInit::Random",
        Cajita::createExecutionPolicy( owned_cells, exec_space ),
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       int& create_count ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // Particle coordinate.
            double px[3];

            // Particle volume.
            // FIXME: this is incorrect for an adaptive mesh. We will need an
            // overload which gets the nodes and computes the volume. We will
            // still place particles uniformly in the logical space but we
            // will then need to map them back to the reference space later.
            double pv = local_mesh.measure( Cajita::Cell(), low_node ) /
                        particles_per_cell;

            // Particle.
            particle_type particle;

            // Create particles.
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Local particle id.
                int pid = cell_id * particles_per_cell + p;

                // Select a random point in the cell for the particle
                // location. These coordinates are logical.
                for ( int d = 0; d < 3; ++d )
                {
                    px[d] = Kokkos::rand<decltype( rand ), double>::draw(
                        rand, low_coords[d], high_coords[d] );
                }

                // Create a new particle with the given logical coordinates.
                particle_created( pid ) =
                    create_functor( px, pv, pid, particle );

                // If we created a new particle insert it into the list.
                if ( particle_created( pid ) )
                {
                    particles.setTuple( pid, particle.tuple() );
                    ++create_count;
                }
            }
        },
        local_num_create );

    // Filter empties.
    filterEmpties( exec_space, local_num_create, particle_created, particles );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a uniform number of particles in each cell given an
  initialization functor.

  \tparam ParticleListType The type of particle list to initialize.

  \tparam InitFunctor Initialization functor type. See the documentation below
  for the create_functor parameter on the signature of this functor.

  \param Initialization type tag.

  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.

  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double px[3],
                          typename ParticleAoSoA::tuple_type& particle );

  \param particle_list The list of particles to populate. This will be filled
  with particles and resized to a size equal to the number of particles
  created.
*/
template <class ParticleListType, class InitFunctor, class ExecutionSpace>
void initializeParticles( InitUniform, const ExecutionSpace& exec_space,
                          const int particles_per_cell_dim,
                          const InitFunctor& create_functor,
                          ParticleListType& particle_list )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    // Particle type.
    using particle_type = typename ParticleListType::particle_type;

    // Get the local grid.
    const auto& local_grid = *( particle_list.mesh().localGrid() );

    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Get the particles.
    auto& particles = particle_list.aosoa();

    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;
    int num_particles = particles_per_cell * owned_cells.size();
    particles.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    Kokkos::parallel_reduce(
        "Picasso::ParticleInit::Uniform",
        Cajita::createExecutionPolicy( owned_cells, exec_space ),
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       int& create_count ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

            // Compute the particle spacing in each dimension.
            double spacing[3] = { ( high_coords[Dim::I] - low_coords[Dim::I] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::J] - low_coords[Dim::J] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::K] - low_coords[Dim::K] ) /
                                      particles_per_cell_dim };

            // Particle coordinate.
            double px[3];

            // Particle volume.
            // FIXME: this is incorrect for an adaptive mesh. We will need an
            // overload which gets the nodes and computes the volume. We will
            // still place particles uniformly in the logical space but we
            // will then need to map them back to the reference space
            // later.
            double pv = local_mesh.measure( Cajita::Cell(), low_node ) /
                        particles_per_cell;

            // Particle.
            particle_type particle;

            // Create particles.
            for ( int ip = 0; ip < particles_per_cell_dim; ++ip )
                for ( int jp = 0; jp < particles_per_cell_dim; ++jp )
                    for ( int kp = 0; kp < particles_per_cell_dim; ++kp )
                    {
                        // Local particle id.
                        int pid = cell_id * particles_per_cell + ip +
                                  particles_per_cell_dim *
                                      ( jp + particles_per_cell_dim * kp );

                        // Set the particle position in logical coordinates.
                        px[Dim::I] = 0.5 * spacing[Dim::I] +
                                     ip * spacing[Dim::I] + low_coords[Dim::I];
                        px[Dim::J] = 0.5 * spacing[Dim::J] +
                                     jp * spacing[Dim::J] + low_coords[Dim::J];
                        px[Dim::K] = 0.5 * spacing[Dim::K] +
                                     kp * spacing[Dim::K] + low_coords[Dim::K];

                        // Create a new particle with the given logical
                        // coordinates.
                        particle_created( pid ) =
                            create_functor( px, pv, pid, particle );

                        // If we created a new particle insert it into the list.
                        if ( particle_created( pid ) )
                        {
                            particles.setTuple( pid, particle.tuple() );
                            ++create_count;
                        }
                    }
        },
        local_num_create );

    // Filter empties.
    filterEmpties( exec_space, local_num_create, particle_created, particles );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_PARTICLEINIT_HPP
