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

#include <Cabana_Grid.hpp>

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

struct FezIn : Field::Vector<double, 3>
{
    static std::string label() { return "fez_in"; }
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

struct MatI : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "mat_I"; }
};

struct MatJ : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "mat_J"; }
};

struct MatK : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "mat_K"; }
};

struct Boo : Field::Tensor3<double, 3, 3, 3>
{
    static std::string label() { return "boo"; }
};

struct Cam : Field::Tensor4<double, 3, 3, 3, 3>
{
    static std::string label() { return "cam"; }
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
        auto& barp = Picasso::get( particle, BarP() );

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
// Grid operation using a Tensor3
struct GridTensor3Func
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
        // Get input dependencies
        auto foo_in = gather_deps.get( FieldLocation::Cell(), FooIn() );
        auto fez_in = gather_deps.get( FieldLocation::Cell(), FezIn() );

        // Get output dependencies
        auto foo_out = scatter_deps.get( FieldLocation::Cell(), FooOut() );
        auto foo_out_access = foo_out.access();

        // Get local dependencies
        auto boo = local_deps.get( FieldLocation::Cell(), Boo() );
        auto mat_i_out = local_deps.get( FieldLocation::Cell(), MatI() );
        auto mat_j_out = local_deps.get( FieldLocation::Cell(), MatJ() );
        auto mat_k_out = local_deps.get( FieldLocation::Cell(), MatK() );

        foo_in( i, j, k, 0 ) = 1.0;
        foo_in( i, j, k, 1 ) = 2.0;
        foo_in( i, j, k, 2 ) = 3.0;
        fez_in( i, j, k, 0 ) = 3.0;
        fez_in( i, j, k, 1 ) = 4.0;
        fez_in( i, j, k, 2 ) = 3.0;

        // Set up the local dependency to be the Levi-Civita tensor.
        Picasso::LinearAlgebra::Tensor3<double, 3, 3, 3> levi_civita;
        Picasso::LinearAlgebra::permutation( levi_civita );
        boo( i, j, k ) = levi_civita;

        const int index[3] = { i, j, k };
        auto boo_t_foo_in = LinearAlgebra::contract(
            boo( index ), foo_in( index ), SpaceDim<2>{} );

        auto boo_t_foo_in_t_fez_in = boo_t_foo_in * fez_in( index );
        for ( int d = 0; d < 3; ++d )
        {
            foo_out_access( i, j, k, d ) += boo_t_foo_in_t_fez_in( d );
        }

        // Now test contraction along the other dimensions of a Tensor3
        Picasso::LinearAlgebra::Tensor3<double, 3, 3, 3> tensor = {
            { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } },
            { { 9, 10, 11 }, { 12, 13, 14 }, { 15, 16, 17 } },
            { { 18, 19, 20 }, { 21, 22, 23 }, { 24, 25, 26 } } };

        mat_i_out( index ) =
            LinearAlgebra::contract( tensor, foo_in( index ), SpaceDim<0>() );
        mat_j_out( index ) =
            LinearAlgebra::contract( tensor, foo_in( index ), SpaceDim<1>() );
        mat_k_out( index ) =
            LinearAlgebra::contract( tensor, foo_in( index ), SpaceDim<2>() );
    }
};

// Grid operation using a Tensor4
struct GridTensor4Func
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
        // // Get input dependencies
        auto foo_in = gather_deps.get( FieldLocation::Cell(), FooIn() );
        auto fez_in = gather_deps.get( FieldLocation::Cell(), FezIn() );

        // Get output dependencies
        auto foo_out = scatter_deps.get( FieldLocation::Cell(), FooOut() );

        // Get local dependencies
        auto cam = local_deps.get( FieldLocation::Cell(), Cam() );
        auto baz_out = local_deps.get( FieldLocation::Cell(), Baz() );

        // Set up the local dependency to be the linear elasticity tensor
        // (3x3x3x3)
        double mu = 1;
        double lam = 0.5;
        cam( i, j, k ) = {
            { { { lam + 2 * mu, 0.0, 0.0 },
                { 0.0, lam, 0.0 },
                { 0.0, 0.0, lam } },
              { { 0.0, mu, 0.0 }, { mu, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
              { { 0.0, 0.0, mu }, { 0.0, 0.0, 0.0 }, { mu, 0.0, 0.0 } } },
            { { { 0.0, mu, 0.0 }, { mu, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
              { { lam, 0.0, 0.0 },
                { 0.0, lam + 2 * mu, 0.0 },
                { 0.0, 0.0, lam } },
              { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, mu }, { 0.0, mu, 0.0 } } },
            { { { 0.0, 0.0, mu }, { 0.0, 0.0, 0.0 }, { mu, 0.0, 0.0 } },
              { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, mu }, { 0.0, mu, 0.0 } },
              { { lam, 0.0, 0.0 },
                { 0.0, lam, 0.0 },
                { 0.0, 0.0, lam + 2 * mu } } } };

        Picasso::Mat3<double> strain = {
            { 0.5, 1.0, 0 }, { 1.0, 0, 0 }, { 0, 0, 0 } };

        const int index[3] = { i, j, k };

        // This operation represents a stress tensor evaluation from the
        // "generalized" Hook's law, which is just a double contraction of a
        // fourth-order stiffness tensor with a strain tensor. baz_out is the
        // stress tensor in this example
        baz_out( index ) = LinearAlgebra::contract( cam( index ), strain );
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
    auto inputs = Picasso::parse( "particle_init_test.json" );
    Kokkos::Array<double, 6> global_box = {
        global_low_corner[0],  global_low_corner[1],  global_low_corner[2],
        global_high_corner[0], global_high_corner[1], global_high_corner[2] };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = createUniformMesh( TEST_MEMSPACE(), inputs, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    Cabana::ParticleTraits<Field::LogicalPosition<3>, FooP, BarP> fields;
    auto particles = Cabana::Grid::createParticleList<TEST_MEMSPACE>(
        "test_particles", fields );
    using list_type = decltype( particles );

    using particle_type = typename list_type::particle_type;

    // Particle initialization functor. Make particles everywhere.
    auto particle_init_func = KOKKOS_LAMBDA( const int, const double x[3],
                                             const double, particle_type& p )
    {
        for ( int d = 0; d < 3; ++d )
            Picasso::get( p, Field::LogicalPosition<3>(), d ) = x[d];
        return true;
    };

    // Initialize particles.
    int ppc = 10;
    Cabana::Grid::createParticles( Cabana::InitRandom(), TEST_EXECSPACE(),
                                   particle_init_func, particles, ppc,
                                   *( mesh->localGrid() ) );

    // Make an operator.
    using gather_deps =
        GatherDependencies<FieldLayout<FieldLocation::Cell, FooIn>,
                           FieldLayout<FieldLocation::Cell, FezIn>,
                           FieldLayout<FieldLocation::Cell, BarIn>>;
    using scatter_deps =
        ScatterDependencies<FieldLayout<FieldLocation::Cell, FooOut>,
                            FieldLayout<FieldLocation::Cell, BarOut>>;

    using local_deps =
        LocalDependencies<FieldLayout<FieldLocation::Cell, Baz>,
                          FieldLayout<FieldLocation::Cell, Boo>,
                          FieldLayout<FieldLocation::Cell, Cam>,
                          FieldLayout<FieldLocation::Cell, MatI>,
                          FieldLayout<FieldLocation::Cell, MatJ>,
                          FieldLayout<FieldLocation::Cell, MatK>>;
    auto grid_op =
        createGridOperator( mesh, gather_deps(), scatter_deps(), local_deps() );

    // Make a field manager.
    auto fm = createFieldManager( mesh );

    // Setup the field manager.
    grid_op->setup( *fm );

    // Initialize gather fields.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooIn() ), 2.0 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), BarIn() ), 3.0 );

    // Initialize scatter fields to wrong data to make sure they get reset to
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
    Cabana::Grid::grid_parallel_for(
        "check_grid_out", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cabana::Grid::Own(), Cabana::Grid::Cell(),
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
    Cabana::Grid::grid_parallel_for(
        "check_grid_out", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cabana::Grid::Own(), Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int d = 0; d < 3; ++d )
                EXPECT_EQ( foo_out_host( i, j, k, d ), 4.0 + i + j + k );
            EXPECT_EQ( bar_out_host( i, j, k, 0 ), 6.0 + i + j + k );
        } );

    // Re-initialize gather fields
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooIn() ), 0.0 );
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FezIn() ), 0.0 );

    // Re-initialize scatter field to wrong value.
    Kokkos::deep_copy( fm->view( FieldLocation::Cell(), FooOut() ), -1.1 );

    // Apply the tensor3 grid operator. Use a tag.
    GridTensor3Func grid_tensor3_func;
    grid_op->apply( "grid_tensor3_op", FieldLocation::Cell(), TEST_EXECSPACE(),
                    *fm, GridTensor3Func::Tag(), grid_tensor3_func );

    // Check the grid results.
    Kokkos::deep_copy( foo_out_host,
                       fm->view( FieldLocation::Cell(), FooOut() ) );

    auto mi_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), MatI() ) );
    auto mj_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), MatJ() ) );
    auto mk_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), MatK() ) );

    // Expect the correct cross-product for the given vector fields
    Cabana::Grid::grid_parallel_for(
        "check_tensor3_cross_product", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cabana::Grid::Own(), Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            EXPECT_EQ( foo_out_host( i, j, k, 0 ), -6.0 );
            EXPECT_EQ( foo_out_host( i, j, k, 1 ), 6.0 );
            EXPECT_EQ( foo_out_host( i, j, k, 2 ), -2.0 );
        } );

    // Expect the correct matrices from the various Tensor3 contractions
    Cabana::Grid::grid_parallel_for(
        "check_tensor3_vector_contract", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cabana::Grid::Own(), Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            EXPECT_EQ( mi_out_host( i, j, k, 0 ), 72 );
            EXPECT_EQ( mi_out_host( i, j, k, 1 ), 78 );
            EXPECT_EQ( mi_out_host( i, j, k, 2 ), 84 );
            EXPECT_EQ( mi_out_host( i, j, k, 3 ), 90 );
            EXPECT_EQ( mi_out_host( i, j, k, 4 ), 96 );
            EXPECT_EQ( mi_out_host( i, j, k, 5 ), 102 );
            EXPECT_EQ( mi_out_host( i, j, k, 6 ), 108 );
            EXPECT_EQ( mi_out_host( i, j, k, 7 ), 114 );
            EXPECT_EQ( mi_out_host( i, j, k, 8 ), 120 );

            EXPECT_EQ( mj_out_host( i, j, k, 0 ), 24 );
            EXPECT_EQ( mj_out_host( i, j, k, 1 ), 30 );
            EXPECT_EQ( mj_out_host( i, j, k, 2 ), 36 );
            EXPECT_EQ( mj_out_host( i, j, k, 3 ), 78 );
            EXPECT_EQ( mj_out_host( i, j, k, 4 ), 84 );
            EXPECT_EQ( mj_out_host( i, j, k, 5 ), 90 );
            EXPECT_EQ( mj_out_host( i, j, k, 6 ), 132 );
            EXPECT_EQ( mj_out_host( i, j, k, 7 ), 138 );
            EXPECT_EQ( mj_out_host( i, j, k, 8 ), 144 );

            EXPECT_EQ( mk_out_host( i, j, k, 0 ), 8 );
            EXPECT_EQ( mk_out_host( i, j, k, 1 ), 26 );
            EXPECT_EQ( mk_out_host( i, j, k, 2 ), 44 );
            EXPECT_EQ( mk_out_host( i, j, k, 3 ), 62 );
            EXPECT_EQ( mk_out_host( i, j, k, 4 ), 80 );
            EXPECT_EQ( mk_out_host( i, j, k, 5 ), 98 );
            EXPECT_EQ( mk_out_host( i, j, k, 6 ), 116 );
            EXPECT_EQ( mk_out_host( i, j, k, 7 ), 134 );
            EXPECT_EQ( mk_out_host( i, j, k, 8 ), 152 );
        } );

    GridTensor4Func grid_tensor4_func;
    grid_op->apply( "grid_tensor4_op", FieldLocation::Cell(), TEST_EXECSPACE(),
                    *fm, GridTensor4Func::Tag(), grid_tensor4_func );

    auto baz_out_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fm->view( FieldLocation::Cell(), Baz() ) );

    // Expect the correct matrices from the various tensor contractions
    Cabana::Grid::grid_parallel_for(
        "check_tensor4_matrix_contract", Kokkos::DefaultHostExecutionSpace(),
        *( mesh->localGrid() ), Cabana::Grid::Own(), Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            EXPECT_EQ( baz_out_host( i, j, k, 0 ), 1.25 );
            EXPECT_EQ( baz_out_host( i, j, k, 1 ), 2 );
            EXPECT_EQ( baz_out_host( i, j, k, 2 ), 0 );
            EXPECT_EQ( baz_out_host( i, j, k, 3 ), 2 );
            EXPECT_EQ( baz_out_host( i, j, k, 4 ), 0.25 );
            EXPECT_EQ( baz_out_host( i, j, k, 5 ), 0 );
            EXPECT_EQ( baz_out_host( i, j, k, 6 ), 0 );
            EXPECT_EQ( baz_out_host( i, j, k, 7 ), 0 );
            EXPECT_EQ( baz_out_host( i, j, k, 8 ), 0.25 );
        } );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, gather_scatter_test ) { gatherScatterTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
