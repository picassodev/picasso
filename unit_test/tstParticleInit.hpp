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

#include <Picasso_FieldTypes.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleInit.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
int totalParticlesPerCell( InitUniform, int ppc ) { return ppc * ppc * ppc; }

int totalParticlesPerCell( InitRandom, int ppc ) { return ppc; }

//---------------------------------------------------------------------------//
// Field tags.
struct Foo : public Field::Vector<double, 3>
{
    static std::string label() { return "foo"; }
};

struct Bar : public Field::Scalar<double>
{
    static std::string label() { return "bar"; }
};

//---------------------------------------------------------------------------//
template <class InitType>
void InitTest( InitType init_type, const int ppc, const int boundary = 1,
               const int multiplier = 1 )
{
    // Global bounding box.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 14, 17, 19 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "particle_init_test.json" );
    Kokkos::Array<double, 6> global_box = {
        global_low_corner[0],  global_low_corner[1],  global_low_corner[2],
        global_high_corner[0], global_high_corner[1], global_high_corner[2] };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        inputs, global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type = ParticleList<UniformMesh<TEST_MEMSPACE>, Foo, Bar>;
    list_type particles( "test_particles", mesh );
    using particle_type = typename list_type::particle_type;

    // Particle initialization functor.
    const Kokkos::Array<double, 6> box = {
        global_low_corner[Dim::I] + cell_size * boundary,
        global_high_corner[Dim::I] - cell_size * boundary,
        global_low_corner[Dim::J] + cell_size * boundary,
        global_high_corner[Dim::J] - cell_size * boundary,
        global_low_corner[Dim::K] + cell_size * boundary,
        global_high_corner[Dim::K] - cell_size * boundary };
    auto particle_init_func =
        KOKKOS_LAMBDA( const double x[3], const double v, particle_type& p )
    {
        // Put particles in a box that is one cell smaller than the global
        // mesh. This will give us a layer of empty cells.
        if ( x[Dim::I] > box[0] && x[Dim::I] < box[1] && x[Dim::J] > box[2] &&
             x[Dim::J] < box[3] && x[Dim::K] > box[4] && x[Dim::K] < box[5] )
        {
            for ( int d = 0; d < 3; ++d )
                get( p, Foo(), d ) = x[d];

            get( p, Bar() ) = v;

            return true;
        }
        else
        {
            return false;
        }
    };

    // Initialize particles (potentially multiple times).
    for ( int m = 0; m < multiplier; ++m )
        initializeParticles( init_type, TEST_EXECSPACE(), ppc,
                             particle_init_func, particles );

    // Check that we made particles.
    int num_p = particles.size();
    EXPECT_TRUE( num_p > 0 );

    // Compute the global number of particles.
    const auto& global_grid = mesh->localGrid()->globalGrid();
    int global_num_particle = num_p;
    MPI_Allreduce( MPI_IN_PLACE, &global_num_particle, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    int expect_num_particle =
        multiplier * totalParticlesPerCell( init_type, ppc ) *
        ( global_grid.globalNumEntity( Cajita::Cell(), Dim::I ) -
          2 * boundary ) *
        ( global_grid.globalNumEntity( Cajita::Cell(), Dim::J ) -
          2 * boundary ) *
        ( global_grid.globalNumEntity( Cajita::Cell(), Dim::K ) -
          2 * boundary );
    EXPECT_EQ( global_num_particle, expect_num_particle );

    // Particle volume.
    double volume = mesh->cellSize() * mesh->cellSize() * mesh->cellSize() /
                    totalParticlesPerCell( init_type, ppc );

    // Check that all particles are in the box and got initialized correctly.
    auto host_particles = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particles.aosoa() );
    auto px = Cabana::slice<0>( host_particles );
    auto pv = Cabana::slice<1>( host_particles );
    for ( int p = 0; p < num_p; ++p )
    {
        EXPECT_TRUE( px( p, Dim::I ) > box[0] );
        EXPECT_TRUE( px( p, Dim::I ) < box[1] );
        EXPECT_TRUE( px( p, Dim::J ) > box[2] );
        EXPECT_TRUE( px( p, Dim::J ) < box[3] );
        EXPECT_TRUE( px( p, Dim::K ) > box[4] );
        EXPECT_TRUE( px( p, Dim::K ) < box[5] );

        EXPECT_DOUBLE_EQ( pv( p ), volume );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, random_init_test )
{
    // Boundary layers (last input) chosen to test when creating more particles
    // than empty and vice versa.
    InitTest( InitRandom(), 17 );
    InitTest( InitRandom(), 9, 3 );
}

TEST( TEST_CATEGORY, uniform_init_test )
{
    InitTest( InitUniform(), 3 );
    InitTest( InitUniform(), 2, 2 );
}

TEST( TEST_CATEGORY, multiple_uniform_init_test )
{
    InitTest( InitUniform(), 3, 1, 3 );
    InitTest( InitUniform(), 2, 3, 3 );
}

TEST( TEST_CATEGORY, multiple_random_init_test )
{
    InitTest( InitRandom(), 11, 1, 4 );
    InitTest( InitRandom(), 4, 4, 3 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
