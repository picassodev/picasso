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
#include <Picasso_GridOperator.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleInit.hpp>
#include <Picasso_ParticleInterpolation.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformCartesianMeshMapping.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct FooP : Field::Vector<double, 2>
{
    static std::string label() { return "foo_p"; }
};

struct FooIn : Field::Vector<double, 2>
{
    static std::string label() { return "foo_in"; }
};

struct FooOut : Field::Vector<double, 2>
{
    static std::string label() { return "foo_out"; }
};

struct BarP : Field::Scalar<double>
{
    static std::string label() { return "bar_p"; }
};

struct BarIn : Field::Scalar<double>
{
    static std::string label() { return "bar_in"; }
};

struct BarOut : Field::Scalar<double>
{
    static std::string label() { return "bar_out"; }
};

struct Baz : Field::Matrix<double, 2, 2>
{
    static std::string label() { return "baz"; }
};

//---------------------------------------------------------------------------//
// Grid operation.
struct GridFunc
{
    struct Tag
    {
    };

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void operator()(
        Tag, const LocalMeshType&, const GatherDependencies& gather_deps,
        const ScatterDependencies& scatter_deps,
        const LocalDependencies& local_deps, const int i, const int j ) const
    {
        // Get input dependencies.
        auto foo_in = gather_deps.get( FieldLocation::Cell(), FooIn() );
        auto bar_in = gather_deps.get( FieldLocation::Cell(), BarIn() );

        // Get output dependencies.
        auto foo_out = scatter_deps.get( FieldLocation::Cell(), FooOut() );
        auto bar_out = scatter_deps.get( FieldLocation::Cell(), BarOut() );

        // Get scatter view accessors.
        auto foo_out_access = foo_out.access();
        auto bar_out_access = bar_out.access();

        // Get local dependencies.
        auto baz = local_deps.get( FieldLocation::Cell(), Baz() );

        // Set up the local dependency to be the identity.
        baz( i, j ) = { { 1.0, 0.0 }, { 0.0, 1.0 } };

        // Assign foo_out - build a point-wise expression to do this. Test out
        // both separate index and array-based indices.
        const int index[2] = { i, j };
        auto baz_t_foo_in_t_2 =
            baz( index ) * ( foo_in( index ) + foo_in( i, j ) );
        for ( int d = 0; d < 2; ++d )
            foo_out_access( i, j, d ) += baz_t_foo_in_t_2( d ) + i + j;

        // Assign the bar_out - use mixture of field dimension indices
        // or direct ijk access.
        bar_out_access( i, j, 0 ) += bar_in( i, j, 0 ) + bar_in( i, j ) + i + j;
    }
};

//---------------------------------------------------------------------------//
void gatherScatterTest()
{
    // Global bounding box.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 43, 22 };
    std::array<double, 2> global_low_corner = { 1.2, 2.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    Kokkos::Array<bool, 2> periodic = { false, false };
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { comm_size, 1 };

    // Get inputs for mesh.
    Kokkos::Array<double, 4> global_box = {
        global_low_corner[0], global_low_corner[1], global_high_corner[0],
        global_high_corner[1] };
    int minimum_halo_size = 0;

    // Make mesh and field manager.
    auto fm = createUniformCartesianMesh(
        TEST_MEMSPACE{}, cell_size, global_box, periodic, minimum_halo_size,
        MPI_COMM_WORLD, ranks_per_dim );

    // Make an operator.
    using gather_deps =
        GatherDependencies<FieldLayout<FieldLocation::Cell, FooIn>,
                           FieldLayout<FieldLocation::Cell, BarIn>>;
    using scatter_deps =
        ScatterDependencies<FieldLayout<FieldLocation::Cell, FooOut>,
                            FieldLayout<FieldLocation::Cell, BarOut>>;
    using local_deps = LocalDependencies<FieldLayout<FieldLocation::Cell, Baz>>;
    auto grid_op = createGridOperator( fm->mesh(), gather_deps(),
                                       scatter_deps(), local_deps() );

    // Setup the field manager.
    grid_op->setup( *fm );

    // Initialize gather fields.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooIn() ), 2.0 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), BarIn() ), 3.0 );

    // Initialiize scatter fields to wrong data to make sure they get reset to
    // zero and then overwritten with the data we assign in the operator.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooOut() ), -1.1 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), BarOut() ), -2.2 );

    // Apply the grid operator. Use a tag.
    GridFunc grid_func;
    grid_op->apply( "grid_op", FieldLocation::Cell(), TEST_EXECSPACE(), *fm,
                    GridFunc::Tag(), grid_func );

    // Check the grid results.
    auto foo_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), FooOut() ) );
    auto bar_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), BarOut() ) );
    Kokkos::deep_copy( foo_out_host,
                       fm->view( FieldLocation::Cell(), FooOut() ) );
    Kokkos::deep_copy( bar_out_host,
                       fm->view( FieldLocation::Cell(), BarOut() ) );
    Cajita::grid_parallel_for(
        "check_grid_out", Kokkos::DefaultHostExecutionSpace(),
        *( fm->mesh()->localGrid() ), Cajita::Own(), Cajita::Cell(),
        KOKKOS_LAMBDA( const int i, const int j ) {
            for ( int d = 0; d < 2; ++d )
                EXPECT_EQ( foo_out_host( i, j, d ), 4.0 + i + j );
            EXPECT_EQ( bar_out_host( i, j, 0 ), 6.0 + i + j );
        } );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, gather_scatter_test ) { gatherScatterTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
