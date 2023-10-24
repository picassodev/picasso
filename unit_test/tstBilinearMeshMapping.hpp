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

#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_BilinearMeshMapping.hpp>
#include <Picasso_CurvilinearMesh.hpp>
#include <Picasso_Types.hpp>

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void mappingTest3d()
{
    // Global parameters.
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int halo_width = 1;
    Kokkos::Array<bool, 3> periodic = { true, false, true };

    // Partition only in the x direction.
    std::array<int, 3> ranks_per_dim = { 1, 1, 1 };
    MPI_Comm_size( MPI_COMM_WORLD, &ranks_per_dim[0] );

    // Uniform bilinear mesh generator.
    UniformBilinearMeshGenerator<3> generator = { cell_size, global_box,
                                                  periodic };

    // Create the mesh and field manager.
    auto manager = createBilinearMesh( TEST_MEMSPACE{}, generator, halo_width,
                                       MPI_COMM_WORLD, ranks_per_dim );

    // Check the mapping.
    auto mapping = manager->mesh()->mapping();
    using mapping_type = decltype( mapping );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( num_cell, mapping._global_num_cell[d] );
        EXPECT_EQ( periodic[d], mapping._periodic[d] );
    }

    // Check grid.
    const auto& global_grid = manager->mesh()->localGrid()->globalGrid();
    const auto& global_mesh = global_grid.globalMesh();

    EXPECT_EQ( global_mesh.lowCorner( 0 ), 0.0 );
    EXPECT_EQ( global_mesh.lowCorner( 1 ), 0.0 );
    EXPECT_EQ( global_mesh.lowCorner( 2 ), 0.0 );

    EXPECT_EQ( global_mesh.highCorner( 0 ), num_cell );
    EXPECT_EQ( global_mesh.highCorner( 1 ), num_cell );
    EXPECT_EQ( global_mesh.highCorner( 2 ), num_cell );

    EXPECT_EQ( global_grid.globalNumEntity( Cabana::Grid::Cell(), 0 ),
               num_cell );
    EXPECT_EQ( global_grid.globalNumEntity( Cabana::Grid::Cell(), 1 ),
               num_cell );
    EXPECT_EQ( global_grid.globalNumEntity( Cabana::Grid::Cell(), 2 ),
               num_cell );

    EXPECT_TRUE( global_grid.isPeriodic( 0 ) );
    EXPECT_FALSE( global_grid.isPeriodic( 1 ) );
    EXPECT_TRUE( global_grid.isPeriodic( 2 ) );

    auto local_grid = manager->mesh()->localGrid();
    EXPECT_EQ( local_grid->haloCellWidth(), 1 );

    // Give a bad guess that is still in the local domain to test the ability
    // to search.
    auto global_cells = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Global() );
    Kokkos::Array<double, 3> guess = {
        static_cast<double>( global_cells.min( Dim::I ) ),
        static_cast<double>( global_cells.min( Dim::J ) ),
        static_cast<double>( global_cells.min( Dim::K ) ) };

    // Check mapping.
    auto ghosted_cells = local_grid->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_MEMSPACE>( *local_grid );
    Kokkos::View<double*** [3], TEST_MEMSPACE> cell_forward_map(
        "cell_forward_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ), ghosted_cells.extent( Dim::K ) );
    Kokkos::View<double*** [3], TEST_MEMSPACE> cell_reverse_map(
        "cell_reverse_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ), ghosted_cells.extent( Dim::K ) );
    Kokkos::View<bool***, TEST_MEMSPACE> cell_map_success(
        "cell_reverse_map_success", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ), ghosted_cells.extent( Dim::K ) );
    Kokkos::View<double*** [3], TEST_MEMSPACE> default_cell_reverse_map(
        "default_cell_reverse_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ), ghosted_cells.extent( Dim::K ) );
    Kokkos::View<bool***, TEST_MEMSPACE> default_cell_map_success(
        "default_cell_reverse_map_success", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ), ghosted_cells.extent( Dim::K ) );
    Cabana::Grid::grid_parallel_for(
        "check_mapping", TEST_EXECSPACE{}, ghosted_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Map to physical frame.
            LinearAlgebra::Vector<double, 3> cell_coords = 0.0;
            int ijk[3] = { i, j, k };
            local_mesh.coordinates( Cabana::Grid::Cell{}, ijk,
                                    cell_coords.data() );

            LinearAlgebra::VectorView<double, 3> phys_coords(
                &cell_forward_map( i, j, k, 0 ), cell_forward_map.stride( 3 ) );
            CurvilinearMeshMapping<mapping_type>::mapToPhysicalFrame(
                mapping, cell_coords, phys_coords );

            // Map to reference frame. Use the bad guess to test search.
            LinearAlgebra::VectorView<double, 3> ref_coords(
                &cell_reverse_map( i, j, k, 0 ), cell_reverse_map.stride( 3 ) );
            ref_coords = { guess[Dim::I], guess[Dim::J], guess[Dim::K] };
            cell_map_success( i, j, k ) =
                CurvilinearMeshMapping<mapping_type>::mapToReferenceFrame(
                    mapping, phys_coords, ref_coords );

            // Default map to reference frame. Use a good guess in the cell.
            LinearAlgebra::VectorView<double, 3> default_ref_coords(
                &default_cell_reverse_map( i, j, k, 0 ),
                default_cell_reverse_map.stride( 3 ) );
            default_ref_coords = { cell_coords( Dim::I ) - 0.1,
                                   cell_coords( Dim::J ) - 0.1,
                                   cell_coords( Dim::K ) - 0.1 };
            default_cell_map_success( i, j, k ) = DefaultCurvilinearMeshMapping<
                mapping_type>::mapToReferenceFrame( mapping, phys_coords,
                                                    default_ref_coords );
        } );

    auto forward_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_forward_map );
    auto reverse_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_reverse_map );
    auto map_success_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_map_success );
    auto default_reverse_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, default_cell_reverse_map );
    auto default_map_success_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, default_cell_map_success );

    auto global_offset_i =
        local_grid->globalGrid().globalOffset( Dim::I ) - halo_width;
    auto global_offset_j =
        local_grid->globalGrid().globalOffset( Dim::J ) - halo_width;
    auto global_offset_k =
        local_grid->globalGrid().globalOffset( Dim::K ) - halo_width;

    for ( int i = ghosted_cells.min( Dim::I ); i < ghosted_cells.max( Dim::I );
          ++i )
        for ( int j = ghosted_cells.min( Dim::J );
              j < ghosted_cells.max( Dim::J ); ++j )
            for ( int k = ghosted_cells.min( Dim::K );
                  k < ghosted_cells.max( Dim::K ); ++k )
            {
                EXPECT_FLOAT_EQ( cell_size * ( global_offset_i + i + 0.5 ) +
                                     global_box[0],
                                 forward_map_h( i, j, k, Dim::I ) );
                EXPECT_FLOAT_EQ( cell_size * ( global_offset_j + j + 0.5 ) +
                                     global_box[1],
                                 forward_map_h( i, j, k, Dim::J ) );
                EXPECT_FLOAT_EQ( cell_size * ( global_offset_k + k + 0.5 ) +
                                     global_box[2],
                                 forward_map_h( i, j, k, Dim::K ) );

                EXPECT_TRUE( map_success_h( i, j, k ) );

                EXPECT_FLOAT_EQ( global_offset_i + i + 0.5,
                                 reverse_map_h( i, j, k, Dim::I ) );
                EXPECT_FLOAT_EQ( global_offset_j + j + 0.5,
                                 reverse_map_h( i, j, k, Dim::J ) );
                EXPECT_FLOAT_EQ( global_offset_k + k + 0.5,
                                 reverse_map_h( i, j, k, Dim::K ) );

                EXPECT_TRUE( default_map_success_h( i, j, k ) );

                EXPECT_FLOAT_EQ( global_offset_i + i + 0.5,
                                 default_reverse_map_h( i, j, k, Dim::I ) );
                EXPECT_FLOAT_EQ( global_offset_j + j + 0.5,
                                 default_reverse_map_h( i, j, k, Dim::J ) );
                EXPECT_FLOAT_EQ( global_offset_k + k + 0.5,
                                 default_reverse_map_h( i, j, k, Dim::K ) );
            }
}

//---------------------------------------------------------------------------//
void mappingTest2d()
{
    // Global parameters.
    Kokkos::Array<double, 4> global_box = { -10.0, -10.0, 10.0, 10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int halo_width = 1;
    Kokkos::Array<bool, 2> periodic = { true, false };

    // Partition only in the x direction.
    std::array<int, 2> ranks_per_dim = { 1, 1 };
    MPI_Comm_size( MPI_COMM_WORLD, &ranks_per_dim[0] );

    // Uniform bilinear mesh generator.
    UniformBilinearMeshGenerator<2> generator = { cell_size, global_box,
                                                  periodic };

    // Create the mesh and field manager.
    auto manager = createBilinearMesh( TEST_MEMSPACE{}, generator, halo_width,
                                       MPI_COMM_WORLD, ranks_per_dim );

    // Check the mapping.
    auto mapping = manager->mesh()->mapping();
    using mapping_type = decltype( mapping );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( num_cell, mapping._global_num_cell[d] );
        EXPECT_EQ( periodic[d], mapping._periodic[d] );
    }

    // Check grid.
    const auto& global_grid = manager->mesh()->localGrid()->globalGrid();
    const auto& global_mesh = global_grid.globalMesh();

    EXPECT_EQ( global_mesh.lowCorner( 0 ), 0.0 );
    EXPECT_EQ( global_mesh.lowCorner( 1 ), 0.0 );

    EXPECT_EQ( global_mesh.highCorner( 0 ), num_cell );
    EXPECT_EQ( global_mesh.highCorner( 1 ), num_cell );

    EXPECT_EQ( global_grid.globalNumEntity( Cabana::Grid::Cell(), 0 ),
               num_cell );
    EXPECT_EQ( global_grid.globalNumEntity( Cabana::Grid::Cell(), 1 ),
               num_cell );

    EXPECT_TRUE( global_grid.isPeriodic( 0 ) );
    EXPECT_FALSE( global_grid.isPeriodic( 1 ) );

    auto local_grid = manager->mesh()->localGrid();
    EXPECT_EQ( local_grid->haloCellWidth(), 1 );

    // Give a bad guess that is still in the local domain to test the ability
    // to search.
    auto global_cells = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Global() );
    Kokkos::Array<double, 2> guess = {
        static_cast<double>( global_cells.min( Dim::I ) ),
        static_cast<double>( global_cells.min( Dim::J ) ) };

    // Check mapping.
    auto ghosted_cells = local_grid->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_MEMSPACE>( *local_grid );
    Kokkos::View<double** [2], TEST_MEMSPACE> cell_forward_map(
        "cell_forward_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ) );
    Kokkos::View<double** [2], TEST_MEMSPACE> cell_reverse_map(
        "cell_reverse_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ) );
    Kokkos::View<bool**, TEST_MEMSPACE> cell_map_success(
        "cell_reverse_map_success", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ) );
    Kokkos::View<double** [2], TEST_MEMSPACE> default_cell_reverse_map(
        "default_cell_reverse_map", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ) );
    Kokkos::View<bool**, TEST_MEMSPACE> default_cell_map_success(
        "default_cell_reverse_map_success", ghosted_cells.extent( Dim::I ),
        ghosted_cells.extent( Dim::J ) );
    Cabana::Grid::grid_parallel_for(
        "check_mapping", TEST_EXECSPACE{}, ghosted_cells,
        KOKKOS_LAMBDA( const int i, const int j ) {
            // Map to physical frame.
            LinearAlgebra::Vector<double, 2> cell_coords = 0.0;
            int ijk[2] = { i, j };
            local_mesh.coordinates( Cabana::Grid::Cell{}, ijk,
                                    cell_coords.data() );
            LinearAlgebra::VectorView<double, 2> phys_coords(
                &cell_forward_map( i, j, 0 ), cell_forward_map.stride( 2 ) );
            CurvilinearMeshMapping<mapping_type>::mapToPhysicalFrame(
                mapping, cell_coords, phys_coords );

            // Map to reference frame. Use a bad guess to test the search.
            LinearAlgebra::VectorView<double, 2> ref_coords(
                &cell_reverse_map( i, j, 0 ), cell_reverse_map.stride( 2 ) );
            ref_coords = { guess[Dim::I], guess[Dim::J] };
            cell_map_success( i, j ) =
                CurvilinearMeshMapping<mapping_type>::mapToReferenceFrame(
                    mapping, phys_coords, ref_coords );

            // Default map to reference frame. Use a good guess that is in the
            // cell.
            LinearAlgebra::VectorView<double, 2> default_ref_coords(
                &default_cell_reverse_map( i, j, 0 ),
                default_cell_reverse_map.stride( 2 ) );
            default_ref_coords = { cell_coords( Dim::I ) - 0.1,
                                   cell_coords( Dim::J ) - 0.1 };
            default_cell_map_success( i, j ) = DefaultCurvilinearMeshMapping<
                mapping_type>::mapToReferenceFrame( mapping, phys_coords,
                                                    default_ref_coords );
        } );

    auto forward_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_forward_map );
    auto reverse_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_reverse_map );
    auto map_success_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, cell_map_success );
    auto default_reverse_map_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, default_cell_reverse_map );
    auto default_map_success_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, default_cell_map_success );

    auto global_offset_i =
        local_grid->globalGrid().globalOffset( Dim::I ) - halo_width;
    auto global_offset_j =
        local_grid->globalGrid().globalOffset( Dim::J ) - halo_width;

    for ( int i = ghosted_cells.min( Dim::I ); i < ghosted_cells.max( Dim::I );
          ++i )
        for ( int j = ghosted_cells.min( Dim::J );
              j < ghosted_cells.max( Dim::J ); ++j )
        {
            EXPECT_FLOAT_EQ( cell_size * ( global_offset_i + i + 0.5 ) +
                                 global_box[0],
                             forward_map_h( i, j, Dim::I ) );
            EXPECT_FLOAT_EQ( cell_size * ( global_offset_j + j + 0.5 ) +
                                 global_box[1],
                             forward_map_h( i, j, Dim::J ) );

            EXPECT_TRUE( map_success_h( i, j ) );

            EXPECT_FLOAT_EQ( global_offset_i + i + 0.5,
                             reverse_map_h( i, j, Dim::I ) );
            EXPECT_FLOAT_EQ( global_offset_j + j + 0.5,
                             reverse_map_h( i, j, Dim::J ) );

            EXPECT_TRUE( default_map_success_h( i, j ) );

            EXPECT_FLOAT_EQ( global_offset_i + i + 0.5,
                             default_reverse_map_h( i, j, Dim::I ) );
            EXPECT_FLOAT_EQ( global_offset_j + j + 0.5,
                             default_reverse_map_h( i, j, Dim::J ) );
        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, mapping_test_3d ) { mappingTest3d(); }

TEST( TEST_CATEGORY, mapping_test_2d ) { mappingTest2d(); }

//---------------------------------------------------------------------------//

} // end namespace Test
