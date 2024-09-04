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

#include <Picasso_FieldManager.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_Types.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
template <class View, class IndexSpace>
void checkGather( const View& view, const IndexSpace& index_space,
                  const double value )
{
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );
    for ( int i = 0; i < index_space.max( Dim::I ); ++i )
        for ( int j = 0; j < index_space.max( Dim::J ); ++j )
            for ( int k = 0; k < index_space.max( Dim::K ); ++k )
                EXPECT_EQ( host_view( i, j, k, 0 ), value );
}

//---------------------------------------------------------------------------//
void uniformTest()
{
    // Get inputs for mesh.
    auto inputs = Picasso::parse( "field_manager_test.json" );
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        inputs, global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a field Manager
    FieldManager<UniformMesh<TEST_MEMSPACE>> fm( mesh );

    // Add some fields. Add the same field so we can insure the variants are
    // loaded into different maps.
    fm.add( FieldLocation::Node(), Field::Color() );
    fm.add( FieldLocation::Cell(), Field::Color() );
    fm.add( FieldLocation::Face<Dim::I>(), Field::Color() );
    fm.add( FieldLocation::Face<Dim::J>(), Field::Color() );
    fm.add( FieldLocation::Face<Dim::K>(), Field::Color() );
    fm.add( FieldLocation::Edge<Dim::I>(), Field::Color() );
    fm.add( FieldLocation::Edge<Dim::J>(), Field::Color() );
    fm.add( FieldLocation::Edge<Dim::K>(), Field::Color() );

    // Put data in the fields.
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Node(), Field::Color() ), 1,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Cell(), Field::Color() ), 2,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Face<Dim::I>(), Field::Color() ), 3,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Face<Dim::J>(), Field::Color() ), 4,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Face<Dim::K>(), Field::Color() ), 5,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Edge<Dim::I>(), Field::Color() ), 6,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Edge<Dim::J>(), Field::Color() ), 7,
        Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign(
        *fm.array( FieldLocation::Edge<Dim::K>(), Field::Color() ), 8,
        Cabana::Grid::Own() );

    // Gather into the ghost zones.
    fm.gather( FieldLocation::Node(), Field::Color() );
    fm.gather( FieldLocation::Cell(), Field::Color() );
    fm.gather( FieldLocation::Face<Dim::I>(), Field::Color() );
    fm.gather( FieldLocation::Face<Dim::J>(), Field::Color() );
    fm.gather( FieldLocation::Face<Dim::K>(), Field::Color() );
    fm.gather( FieldLocation::Edge<Dim::I>(), Field::Color() );
    fm.gather( FieldLocation::Edge<Dim::J>(), Field::Color() );
    fm.gather( FieldLocation::Edge<Dim::K>(), Field::Color() );

    // Check the gather.
    checkGather( fm.view( FieldLocation::Node(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Node(),
                                                Cabana::Grid::Local() ),
                 1 );
    checkGather( fm.view( FieldLocation::Cell(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Cell(),
                                                Cabana::Grid::Local() ),
                 2 );
    checkGather( fm.view( FieldLocation::Face<Dim::I>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Face<Dim::I>(),
                                                Cabana::Grid::Local() ),
                 3 );
    checkGather( fm.view( FieldLocation::Face<Dim::J>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Face<Dim::J>(),
                                                Cabana::Grid::Local() ),
                 4 );
    checkGather( fm.view( FieldLocation::Face<Dim::K>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Face<Dim::K>(),
                                                Cabana::Grid::Local() ),
                 5 );
    checkGather( fm.view( FieldLocation::Edge<Dim::I>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Edge<Dim::I>(),
                                                Cabana::Grid::Local() ),
                 6 );
    checkGather( fm.view( FieldLocation::Edge<Dim::J>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Edge<Dim::J>(),
                                                Cabana::Grid::Local() ),
                 7 );
    checkGather( fm.view( FieldLocation::Edge<Dim::K>(), Field::Color() ),
                 mesh->localGrid()->indexSpace( Cabana::Grid::Ghost(),
                                                Cabana::Grid::Edge<Dim::K>(),
                                                Cabana::Grid::Local() ),
                 8 );

    // Scatter back to owned.
    fm.scatter( FieldLocation::Node(), Field::Color() );
    fm.scatter( FieldLocation::Cell(), Field::Color() );
    fm.scatter( FieldLocation::Face<Dim::I>(), Field::Color() );
    fm.scatter( FieldLocation::Face<Dim::J>(), Field::Color() );
    fm.scatter( FieldLocation::Face<Dim::K>(), Field::Color() );
    fm.scatter( FieldLocation::Edge<Dim::I>(), Field::Color() );
    fm.scatter( FieldLocation::Edge<Dim::J>(), Field::Color() );
    fm.scatter( FieldLocation::Edge<Dim::K>(), Field::Color() );
}

//---------------------------------------------------------------------------//
void adaptiveTest()
{
    // Get inputs for mesh.
    auto inputs = Picasso::parse( "field_manager_test.json" );
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    int minimum_halo_size = 1;
    double cell_size = 0.5;

    // Make mesh.
    auto mesh = std::make_shared<AdaptiveMesh<TEST_MEMSPACE>>(
        inputs, global_box, minimum_halo_size, MPI_COMM_WORLD,
        TEST_EXECSPACE() );

    // Make a field Manager
    FieldManager<AdaptiveMesh<TEST_MEMSPACE>> fm( mesh );

    // Check that the adaptive mesh coordinates got loaded.
    auto nodes =
        fm.array( FieldLocation::Node(), Field::PhysicalPosition<3>() );
    auto host_coords = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                            nodes->view() );
    auto local_mesh = Cabana::Grid::createLocalMesh<TEST_EXECSPACE>(
        *( nodes->layout()->localGrid() ) );
    auto local_space = nodes->layout()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    for ( int i = local_space.min( 0 ); i < local_space.max( 0 ); ++i )
        for ( int j = local_space.min( 1 ); j < local_space.max( 1 ); ++j )
            for ( int k = local_space.min( 2 ); k < local_space.max( 2 ); ++k )
            {
                EXPECT_EQ( host_coords( i, j, k, 0 ),
                           local_mesh.lowCorner( Cabana::Grid::Ghost(), 0 ) +
                               i * cell_size );
                EXPECT_EQ( host_coords( i, j, k, 1 ),
                           local_mesh.lowCorner( Cabana::Grid::Ghost(), 1 ) +
                               j * cell_size );
                EXPECT_EQ( host_coords( i, j, k, 2 ),
                           local_mesh.lowCorner( Cabana::Grid::Ghost(), 2 ) +
                               k * cell_size );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, uniform_test ) { uniformTest(); }

TEST( TEST_CATEGORY, adaptive_test ) { adaptiveTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
