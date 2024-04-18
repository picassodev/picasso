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

#include <Picasso_AdaptiveMesh.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_Types.hpp>

#include <Picasso_ParticleInit.hpp>

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void constructionTest()
{
    // Global parameters.
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int minimum_halo_size = 1;

    // Get inputs for mesh.
    auto inputs_1 = Picasso::parse( "adaptive_mesh_test_1.json" );

    // Make mesh 1.
    AdaptiveMesh<TEST_MEMSPACE> mesh_1( inputs_1, global_box, minimum_halo_size,
                                        MPI_COMM_WORLD );

    // Check grid 1.
    const auto& global_grid_1 = mesh_1.localGrid()->globalGrid();
    const auto& global_mesh_1 = global_grid_1.globalMesh();

    EXPECT_EQ( global_mesh_1.lowCorner( 0 ), 0.0 );
    EXPECT_EQ( global_mesh_1.lowCorner( 1 ), -1.0 );
    EXPECT_EQ( global_mesh_1.lowCorner( 2 ), 0.0 );

    EXPECT_EQ( global_mesh_1.highCorner( 0 ), num_cell );
    EXPECT_EQ( global_mesh_1.highCorner( 1 ), num_cell + 1 );
    EXPECT_EQ( global_mesh_1.highCorner( 2 ), num_cell );

    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 0 ),
               num_cell );
    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 1 ),
               num_cell + 2 );
    EXPECT_EQ( global_grid_1.globalNumEntity( Cabana::Grid::Cell(), 2 ),
               num_cell );

    EXPECT_TRUE( global_grid_1.isPeriodic( 0 ) );
    EXPECT_FALSE( global_grid_1.isPeriodic( 1 ) );
    EXPECT_TRUE( global_grid_1.isPeriodic( 2 ) );

    EXPECT_EQ( mesh_1.localGrid()->haloCellWidth(), 2 );

    // Check grid 1 nodes.
    const auto& nodes_1 = mesh_1.nodes();
    auto host_coords_1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), nodes_1->view() );
    auto local_space_1 = nodes_1->layout()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    auto local_mesh_1 = Cabana::Grid::createLocalMesh<TEST_EXECSPACE>(
        *( nodes_1->layout()->localGrid() ) );
    for ( int i = local_space_1.min( 0 ); i < local_space_1.max( 0 ); ++i )
        for ( int j = local_space_1.min( 1 ); j < local_space_1.max( 1 ); ++j )
            for ( int k = local_space_1.min( 2 ); k < local_space_1.max( 2 );
                  ++k )
            {
                EXPECT_EQ( host_coords_1( i, j, k, 0 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 0 ) +
                               i * cell_size );
                EXPECT_EQ( host_coords_1( i, j, k, 1 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 1 ) +
                               j * cell_size );
                EXPECT_EQ( host_coords_1( i, j, k, 2 ),
                           local_mesh_1.lowCorner( Cabana::Grid::Ghost(), 2 ) +
                               k * cell_size );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test ) { constructionTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
