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

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct FooP : Field::Vector<double, 3>
{
    static std::string label() { return "foo_p"; }
};

struct FooIn : Field::Vector<double, 3>
{
    static std::string label() { return "foo_in"; }
};

struct FooOut : Field::Vector<double, 3>
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

struct Baz : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "baz"; }
};

//---------------------------------------------------------------------------//
// Particle operation.
struct ParticleFunc
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto foo_in = gather_deps.get( FieldLocation::Cell(), FooIn() );
        auto bar_in = gather_deps.get( FieldLocation::Cell(), BarIn() );

        // Get output dependencies.
        auto foo_out = scatter_deps.get( FieldLocation::Cell(), FooOut() );
        auto bar_out = scatter_deps.get( FieldLocation::Cell(), BarOut() );

        // Get particle data.
        auto foop = get( particle, FooP() );
        auto& barp = get( particle, BarP() );

        // Zero-order cell interpolant.
        auto spline = createSpline(
            FieldLocation::Cell(), InterpolationOrder<0>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue() );

        // Interpolate to the particles.
        G2P::value( spline, foo_in, foop );
        G2P::value( spline, bar_in, barp );

        // Interpolate back to the grid.
        P2G::value( spline, foop, foo_out );
        P2G::value( spline, barp, bar_out );
    }
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
    KOKKOS_INLINE_FUNCTION void
    operator()( Tag, const LocalMeshType&,
                const GatherDependencies& gather_deps,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies& local_deps, const int i, const int j,
                const int k ) const
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
        baz( i, j,
             k ) = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };

        // Assign foo_out - build a point-wise expression to do this. Test out
        // both separate index and array-based indices.
        const int index[3] = { i, j, k };
        auto baz_t_foo_in_t_2 =
            baz( index ) * ( foo_in( index ) + foo_in( i, j, k ) );
        for ( int d = 0; d < 3; ++d )
            foo_out_access( i, j, k, d ) += baz_t_foo_in_t_2( d ) + i + j + k;

        // Assign the bar_out - use mixture of field dimension indices
        // or direct ijk access.
        bar_out_access( i, j, k, 0 ) +=
            bar_in( i, j, k, 0 ) + bar_in( i, j, k ) + i + j + k;
    }
};

//---------------------------------------------------------------------------//
void gatherScatterTest()
{
    // Global bounding box.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 43, 32, 39 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Get inputs for mesh.
    InputParser parser( "particle_init_test.json", "json" );
    Kokkos::Array<double, 6> global_box = {
        global_low_corner[0],  global_low_corner[1],  global_low_corner[2],
        global_high_corner[0], global_high_corner[1], global_high_corner[2] };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh =
        createUniformMesh( TEST_MEMSPACE(), parser.propertyTree(), global_box,
                           minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type = ParticleList<UniformMesh<TEST_MEMSPACE>,
                                   Field::LogicalPosition<3>, FooP, BarP>;
    list_type particles( "test_particles", mesh );
    using particle_type = typename list_type::particle_type;

    // Particle initialization functor. Make particles everywhere.
    auto particle_init_func = KOKKOS_LAMBDA( const double x[3], const double,
                                             const int, particle_type& p )
    {
        for ( int d = 0; d < 3; ++d )
            get( p, Field::LogicalPosition<3>(), d ) = x[d];
        return true;
    };

    // Initialize particles.
    int ppc = 10;
    initializeParticles( InitRandom(), TEST_EXECSPACE(), ppc,
                         particle_init_func, particles );

    // Make an operator.
    using gather_deps =
        GatherDependencies<FieldLayout<FieldLocation::Cell, FooIn>,
                           FieldLayout<FieldLocation::Cell, BarIn>>;
    using scatter_deps =
        ScatterDependencies<FieldLayout<FieldLocation::Cell, FooOut>,
                            FieldLayout<FieldLocation::Cell, BarOut>>;
    using local_deps = LocalDependencies<FieldLayout<FieldLocation::Cell, Baz>>;
    auto grid_op =
        createGridOperator( mesh, gather_deps(), scatter_deps(), local_deps() );

    // Make a field manager.
    auto fm = createFieldManager( mesh );

    // Setup the field manager.
    grid_op->setup( *fm );

    // Initialize gather fields.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooIn() ), 2.0 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), BarIn() ), 3.0 );

    // Initialiize scatter fields to wrong data to make sure they get reset to
    // zero and then overwritten with the data we assign in the operator.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooOut() ), -1.1 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), BarOut() ), -2.2 );

    // Apply the particle operator.
    ParticleFunc particle_func;
    grid_op->apply( "particle_op", FieldLocation::Particle(), TEST_EXECSPACE(),
                    *fm, particles, particle_func );

    // Check the particle results.
    auto host_aosoa = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           particles.aosoa() );
    auto foo_p_host = Cabana::slice<1>( host_aosoa );
    auto bar_p_host = Cabana::slice<2>( host_aosoa );
    for ( std::size_t p = 0; p < particles.size(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_EQ( foo_p_host( p, d ), 2.0 );
        EXPECT_EQ( bar_p_host( p ), 3.0 );
    }

    // Check the grid results.
    auto foo_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), FooOut() ) );
    auto bar_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), BarOut() ) );
    Cajita::grid_parallel_for(
        "check_grid_out", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cajita::Own(), Cajita::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int d = 0; d < 3; ++d )
                EXPECT_EQ( foo_out_host( i, j, k, d ), ppc * 2.0 );
            EXPECT_EQ( bar_out_host( i, j, k, 0 ), ppc * 3.0 );
        } );

    // Apply the grid operator. Use a tag.
    GridFunc grid_func;
    grid_op->apply( "grid_op", FieldLocation::Cell(), TEST_EXECSPACE(), *fm,
                    GridFunc::Tag(), grid_func );

    // Check the grid results.
    Kokkos::deep_copy( foo_out_host,
                       fm->view( FieldLocation::Cell(), FooOut() ) );
    Kokkos::deep_copy( bar_out_host,
                       fm->view( FieldLocation::Cell(), BarOut() ) );
    Cajita::grid_parallel_for(
        "check_grid_out", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cajita::Own(), Cajita::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int d = 0; d < 3; ++d )
                EXPECT_EQ( foo_out_host( i, j, k, d ), 4.0 + i + j + k );
            EXPECT_EQ( bar_out_host( i, j, k, 0 ), 6.0 + i + j + k );
        } );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, gather_scatter_test ) { gatherScatterTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
