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
#include <Picasso_ParticleList.hpp>

#include <Cabana_Core.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Fields.
struct Foo : Field::Scalar<double>
{
    static std::string label() { return "foo"; }
};

struct Bar : Field::Tensor<double, 3, 3>
{
    static std::string label() { return "bar"; }
};

//---------------------------------------------------------------------------//
void particleTest()
{
    // Get inputs for mesh.
    InputParser parser( "uniform_mesh_test_1.json", "json" );
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(), global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type =
        ParticleList<UniformMesh<TEST_MEMSPACE>, Field::LogicalPosition, Foo,
                     Field::Color, Bar>;

    list_type particles( "test_particles", mesh );

    // Resize the aosoa.
    auto& aosoa = particles.aosoa();
    std::size_t num_p = 10;
    aosoa.resize( num_p );
    EXPECT_EQ( particles.aosoa().size(), 10 );

    // Populate fields.
    auto px = particles.slice( Field::LogicalPosition() );
    auto pm = particles.slice( Foo() );
    auto pc = particles.slice( Field::Color() );
    auto pf = particles.slice( Bar() );

    Cabana::deep_copy( px, 1.23 );
    Cabana::deep_copy( pm, 3.3 );
    Cabana::deep_copy( pc, 5 );
    Cabana::deep_copy( pf, -1.2 );

    // Check the slices.
    EXPECT_EQ( px.label(), "logical_position" );
    EXPECT_EQ( pm.label(), "foo" );
    EXPECT_EQ( pc.label(), "color" );
    EXPECT_EQ( pf.label(), "bar" );

    // Check deep copy.
    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto px_h = Cabana::slice<0>( aosoa_host );
    auto pm_h = Cabana::slice<1>( aosoa_host );
    auto pc_h = Cabana::slice<2>( aosoa_host );
    auto pf_h = Cabana::slice<3>( aosoa_host );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 );

        EXPECT_EQ( pc_h( p ), 5 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 );
    }

    // Locally modify.
    Kokkos::parallel_for(
        "modify", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_p ),
        KOKKOS_LAMBDA( const int p ) {
            typename list_type::particle_type particle( aosoa.getTuple( p ) );

            for ( int d = 0; d < 3; ++d )
                get( particle, Field::LogicalPosition(), d ) += p + d;

            get( particle, Foo() ) += p;

            get( particle, Field::Color() ) += p;

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    get( particle, Bar(), i, j ) += p + i + j;

            aosoa.setTuple( p, particle.tuple() );
        } );

    // Check the modification.
    Cabana::deep_copy( aosoa_host, aosoa );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 + p + d );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 + p );

        EXPECT_EQ( pc_h( p ), 5 + p );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 + p + i + j );
    }
}

//---------------------------------------------------------------------------//
void particleViewTest()
{
    // Get inputs for mesh.
    InputParser parser( "uniform_mesh_test_1.json", "json" );
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(), global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type =
        ParticleList<UniformMesh<TEST_MEMSPACE>, Field::LogicalPosition, Foo,
                     Field::Color, Bar>;

    list_type particles( "test_particles", mesh );

    // Resize the aosoa.
    auto& aosoa = particles.aosoa();
    std::size_t num_p = 10;
    aosoa.resize( num_p );
    EXPECT_EQ( particles.aosoa().size(), 10 );

    // Populate fields.
    auto px = particles.slice( Field::LogicalPosition() );
    auto pm = particles.slice( Foo() );
    auto pc = particles.slice( Field::Color() );
    auto pf = particles.slice( Bar() );

    Cabana::deep_copy( px, 1.23 );
    Cabana::deep_copy( pm, 3.3 );
    Cabana::deep_copy( pc, 5 );
    Cabana::deep_copy( pf, -1.2 );

    // Check the slices.
    EXPECT_EQ( px.label(), "logical_position" );
    EXPECT_EQ( pm.label(), "foo" );
    EXPECT_EQ( pc.label(), "color" );
    EXPECT_EQ( pf.label(), "bar" );

    // Check deep copy.
    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto px_h = Cabana::slice<0>( aosoa_host );
    auto pm_h = Cabana::slice<1>( aosoa_host );
    auto pc_h = Cabana::slice<2>( aosoa_host );
    auto pf_h = Cabana::slice<3>( aosoa_host );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 );

        EXPECT_EQ( pc_h( p ), 5 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 );
    }

    // Locally modify.
    Kokkos::parallel_for(
        "modify", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_p ),
        KOKKOS_LAMBDA( const int p ) {
            auto s = Cabana::Impl::Index<
                list_type::particle_view_type::vector_length>::s( p );
            auto a = Cabana::Impl::Index<
                list_type::particle_view_type::vector_length>::a( p );
            typename list_type::particle_view_type particle( aosoa.access( s ),
                                                             a );

            for ( int d = 0; d < 3; ++d )
                get( particle, Field::LogicalPosition(), d ) += p + d;

            get( particle, Foo() ) += p;

            get( particle, Field::Color() ) += p;

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    get( particle, Bar(), i, j ) += p + i + j;
        } );

    // Check the modification.
    Cabana::deep_copy( aosoa_host, aosoa );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 + p + d );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 + p );

        EXPECT_EQ( pc_h( p ), 5 + p );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 + p + i + j );
    }
}

//---------------------------------------------------------------------------//
void linearAlgebraTest()
{
    // Get inputs for mesh.
    InputParser parser( "uniform_mesh_test_1.json", "json" );
    Kokkos::Array<double, 6> global_box = { -10.0, -10.0, -10.0,
                                            10.0,  10.0,  10.0 };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(), global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type =
        ParticleList<UniformMesh<TEST_MEMSPACE>, Field::LogicalPosition, Foo,
                     Field::Color, Bar>;

    list_type particles( "test_particles", mesh );

    // Resize the aosoa.
    auto& aosoa = particles.aosoa();
    std::size_t num_p = 10;
    aosoa.resize( num_p );
    EXPECT_EQ( particles.aosoa().size(), 10 );

    // Populate fields.
    auto px = particles.slice( Field::LogicalPosition() );
    auto pm = particles.slice( Foo() );
    auto pc = particles.slice( Field::Color() );
    auto pf = particles.slice( Bar() );

    Cabana::deep_copy( px, 1.23 );
    Cabana::deep_copy( pm, 3.3 );
    Cabana::deep_copy( pc, 5 );
    Cabana::deep_copy( pf, -1.2 );

    // Check the slices.
    EXPECT_EQ( px.label(), "logical_position" );
    EXPECT_EQ( pm.label(), "foo" );
    EXPECT_EQ( pc.label(), "color" );
    EXPECT_EQ( pf.label(), "bar" );

    // Check deep copy.
    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto px_h = Cabana::slice<0>( aosoa_host );
    auto pm_h = Cabana::slice<1>( aosoa_host );
    auto pc_h = Cabana::slice<2>( aosoa_host );
    auto pf_h = Cabana::slice<3>( aosoa_host );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 );

        EXPECT_EQ( pc_h( p ), 5 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 );
    }

    // Locally modify.
    Kokkos::parallel_for(
        "modify", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_p ),
        KOKKOS_LAMBDA( const int p ) {
            auto s = Cabana::Impl::Index<
                list_type::particle_view_type::vector_length>::s( p );
            auto a = Cabana::Impl::Index<
                list_type::particle_view_type::vector_length>::a( p );
            typename list_type::particle_view_type particle( aosoa.access( s ),
                                                             a );

            auto px_v = get( particle, Field::LogicalPosition() );
            for ( int d = 0; d < 3; ++d )
                px_v( d ) += p + d;

            get( particle, Foo() ) += p;

            get( particle, Field::Color() ) += p;

            auto pf_m = get( particle, Bar() );
            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    pf_m( i, j ) += p + i + j;
        } );

    // Check the modification.
    Cabana::deep_copy( aosoa_host, aosoa );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( px_h( p, d ), 1.23 + p + d );

        EXPECT_DOUBLE_EQ( pm_h( p ), 3.3 + p );

        EXPECT_EQ( pc_h( p ), 5 + p );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( pf_h( p, i, j ), -1.2 + p + i + j );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, particle_test ) { particleTest(); }

TEST( TEST_CATEGORY, particle_view_test ) { particleViewTest(); }

TEST( TEST_CATEGORY, linear_algebra_test ) { linearAlgebraTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
