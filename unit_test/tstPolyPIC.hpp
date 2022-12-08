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
#include <Picasso_FieldManager.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleInterpolation.hpp>
#include <Picasso_PolyPIC.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct Foo : Field::Vector<double, 3>
{
    static std::string label() { return "foo"; }
};

struct Bar : Field::Scalar<double>
{
    static std::string label() { return "bar"; }
};

struct Baz : Field::Scalar<double>
{
    static std::string label() { return "baz"; }
};

//---------------------------------------------------------------------------//
// Interpolation Order Traits.
//---------------------------------------------------------------------------//
template <int Order>
struct OrderTraits;

template <>
struct OrderTraits<1>
{
    static constexpr int num_mode = 8;
};

template <>
struct OrderTraits<2>
{
    static constexpr int num_mode = 27;
};

template <>
struct OrderTraits<3>
{
    static constexpr int num_mode = 64;
};

//---------------------------------------------------------------------------//
// Linear
//---------------------------------------------------------------------------//
// Check the grid velocity. Computed in mathematica.
template <class GridVelocity>
void checkGridVelocity( std::integral_constant<int, 1>, const int cx,
                        const int cy, const int cz, const GridVelocity& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[8][3] = {
        { -1041.4514891968, -1036.5091579931998, -1029.5668973207999 },
        { -1039.6577396436, -1034.71540844, -1027.7731477676 },
        { -1080.689462254, -1075.7555034336, -1068.821615994 },
        { -1078.8522294412, -1073.9182706208, -1066.9843831812 },
        { -1068.8518602504003, -1063.8168117349, -1056.7818343482002 },
        { -1067.0275793381002, -1061.9925308226, -1054.9575534359 },
        { -1109.1131484693003, -1104.0866148322, -1097.0601531807001 },
        { -1107.2446441701002, -1102.2181105329998, -1095.1916488815 } };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check particle velocity. Computed in Mathematica.
template <class ParticleVelocity>
void checkParticleVelocity( std::integral_constant<int, 1>,
                            const ParticleVelocity& pu_host,
                            const double near_eps, const int array_dim )
{
    double mathematica[8][3] = {
        { -1075.397343439538, -1070.4012502238584, -1063.4052282856792 },
        { -55.68151095982523, -55.496201731800625, -55.31089370531271 },
        { -79.71673394625162, -79.73365540669937, -79.75057857527554 },
        { 3.664027073586908, 3.6640270735866807, 3.6640270735867944 },
        { -4.092313283855674, -4.092883264654347, -4.093453274253761 },
        { 0.12342806044921417, 0.12342806044875942, 0.12342806044921417 },
        { 0.17576855410425196, 0.1757685541047067, 0.1757685541047067 },
        { 0.005921018397202715, 0.005921018397202715, 0.005921018397202715 } };

    for ( int r = 0; r < 8; ++r )
        EXPECT_NEAR( pu_host( r, array_dim ), mathematica[r][array_dim],
                     near_eps );
}

//---------------------------------------------------------------------------//
// Set the particle velocity for lagrangian backtracking in the staggered
// case.
template <class ParticleVelocity>
void setParticleVelocity( std::integral_constant<int, 1>,
                          ParticleVelocity& pu_host )
{
    double mathematica[8][3] = {
        { -1075.397343439538, -1070.4012502238584, -1063.4052282856792 },
        { -55.68151095982523, -55.496201731800625, -55.31089370531271 },
        { -79.71673394625162, -79.73365540669937, -79.75057857527554 },
        { 3.664027073586908, 3.6640270735866807, 3.6640270735867944 },
        { -4.092313283855674, -4.092883264654347, -4.093453274253761 },
        { 0.12342806044921417, 0.12342806044875942, 0.12342806044921417 },
        { 0.17576855410425196, 0.1757685541047067, 0.1757685541047067 },
        { 0.005921018397202715, 0.005921018397202715, 0.005921018397202715 } };

    for ( int r = 0; r < 8; ++r )
        for ( int d = 0; d < 3; ++d )
            pu_host( r, d ) = mathematica[r][d];
}

//---------------------------------------------------------------------------//
// Check the grid momentum. Computed in mathematica.
template <class GridMomentum>
void checkGridMomentum( std::integral_constant<int, 1>, const int cx,
                        const int cy, const int cz, const GridMomentum& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[8][3] = {
        { -20.18146901560989, -20.085647705592606, -19.951047090925332 },
        { -9.485096134368252, -9.439999628365738, -9.376654037915257 },
        { -16.46182244525835, -16.386654910437038, -16.281017935061584 },
        { -7.736986121883586, -7.70160992921125, -7.6518951763516725 },
        { -33.80935969975699, -33.65007195751753, -33.42751273744686 },
        { -15.890262963360303, -15.815297313206546, -15.710556850108693 },
        { -27.577316161715743, -27.452361927441657, -27.277694410711515 },
        { -12.96137223964792, -12.902564982333763, -12.820363239278581 } };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid mass. Computed in mathematica.
template <class GridMass>
void checkGridMass( std::integral_constant<int, 1>, const int cx, const int cy,
                    const int cz, const GridMass& gm_host,
                    const double near_eps )
{
    double mathematica[8] = { 0.019390335999999897, 0.009124863999999963,
                              0.015235263999999998, 0.0071695360000000085,
                              0.031636863999999966, 0.014887936000000006,
                              0.024857536000000104, 0.011697664000000064 };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gm_host( i, j, k, 0 ), mathematica[n], near_eps );
}

//---------------------------------------------------------------------------//
template <class Location, int Order>
void collocatedTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Global mesh parameters.
    Kokkos::Array<double, 6> global_box = { -50.0, -50.0, -50.0,
                                            50.0,  50.0,  50.0 };

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "polypic_test.json" );

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh( inputs, global_box, minimum_halo_size,
                                     MPI_COMM_WORLD );
    auto local_mesh =
        Cajita::createLocalMesh<TEST_EXECSPACE>( *( mesh.localGrid() ) );

    // Time step size.
    double dt = 0.0001;

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 120;
    int cy = 85;
    int cz = 95;

    // Number of velocity modes.
    const int num_mode = OrderTraits<Order>::num_mode;

    // Particle velocity.
    Kokkos::View<double[num_mode][3], TEST_MEMSPACE> pc( "pc" );

    // Create a grid vector on the entities.
    auto grid_vector = createArray( mesh, Location(), Foo() );

    // Initialize the grid vector.
    auto gv_view = grid_vector->view();
    Cajita::grid_parallel_for(
        "fill_grid_vector", TEST_EXECSPACE(),
        grid_vector->layout()->indexSpace( Cajita::Own(), Cajita::Local() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int d ) {
            int ic = i - 2;
            int jc = j - 2;
            int kc = k - 2;
            gv_view( i, j, k, d ) =
                0.0000000001 * ( d + 1 ) * pow( ic, 5 ) -
                0.0000000012 * pow( ( d + 1 ) + jc * ic, 3 ) +
                0.0000000001 * pow( ic * jc * kc, 2 ) + pow( ( d + 1 ), 2 );
        } );

    // Check the grid velocity. Computed in Mathematica.
    auto gv_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gv_view );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 0, 0 );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 1, 1 );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 2, 2 );

    // Do G2P.
    auto gv_wrapper =
        createViewWrapper( FieldLayout<Location, Foo>(), gv_view );
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd =
                createSpline( Location(), InterpolationOrder<Order>(),
                              local_mesh, x, SplineValue(), SplineGradient() );

            LinearAlgebra::Matrix<double, num_mode, 3> modes;
            PolyPIC::g2p( gv_wrapper, modes, sd );

            for ( int r = 0; r < num_mode; ++r )
                for ( int d = 0; d < 3; ++d )
                    pc( r, d ) = modes( r, d );
        } );

    // Check particle velocity. Computed in Mathematica.
    auto pc_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pc );
    checkParticleVelocity( std::integral_constant<int, Order>(), pc_host,
                           near_eps, 0 );
    checkParticleVelocity( std::integral_constant<int, Order>(), pc_host,
                           near_eps, 1 );
    checkParticleVelocity( std::integral_constant<int, Order>(), pc_host,
                           near_eps, 2 );

    // Reset the grid view.
    Kokkos::deep_copy( gv_view, 0.0 );

    // Create grid mass on the entities.
    auto grid_mass = createArray( mesh, Location(), Baz() );
    auto gm_view = grid_mass->view();
    Kokkos::deep_copy( gm_view, 0.0 );

    // Do P2G
    auto gv_sv = Kokkos::Experimental::create_scatter_view( gv_view );
    auto gm_sv = Kokkos::Experimental::create_scatter_view( gm_view );
    Kokkos::parallel_for(
        "p2g", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline( Location(), InterpolationOrder<Order>(),
                                    local_mesh, x, SplineValue(),
                                    SplineGradient(), SplineDistance() );

            LinearAlgebra::Matrix<double, num_mode, 3> modes;
            for ( int r = 0; r < num_mode; ++r )
                for ( int d = 0; d < 3; ++d )
                    modes( r, d ) = pc( r, d );

            PolyPIC::p2g( pm, modes, modes, gv_sv, gm_sv, dt, sd );
        } );
    Kokkos::Experimental::contribute( gv_view, gv_sv );
    Kokkos::Experimental::contribute( gm_view, gm_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gv_host, gv_view );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 0, 0 );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 1, 1 );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 2, 2 );

    // Check grid mass. Computed in Mathematica.
    auto gm_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gm_view );
    checkGridMass( std::integral_constant<int, Order>(), cx, cy, cz, gm_host,
                   near_eps );
}

//---------------------------------------------------------------------------//
template <class Location, int Order>
void staggeredTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Test dimension
    const int Dim = Location::entity_type::dim;

    // Global mesh parameters.
    Kokkos::Array<double, 6> global_box = { -50.0, -50.0, -50.0,
                                            50.0,  50.0,  50.0 };

    // If we are using faces offset the non-test dimensions to align with the
    // mathematica grid.
    if ( Cajita::isFace<typename Location::entity_type>::value )
        for ( int d = 0; d < 3; ++d )
        {
            if ( d != Dim )
            {
                global_box[d] -= 0.25;
                global_box[d + 3] -= 0.25;
            }
        }

    // If we are using edges offset that dimension to align with the
    // mathematica grid for the quadratic functions.
    else if ( Cajita::isEdge<typename Location::entity_type>::value )
    {
        global_box[Dim] -= 0.25;
        global_box[Dim + 3] -= 0.25;
    }

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "polypic_test.json" );

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh( inputs, global_box, minimum_halo_size,
                                     MPI_COMM_WORLD );
    auto local_mesh =
        Cajita::createLocalMesh<TEST_EXECSPACE>( *( mesh.localGrid() ) );

    // Time step size.
    double dt = 0.0001;

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 120;
    int cy = 85;
    int cz = 95;

    // Number of velocity modes.
    const int num_mode = OrderTraits<Order>::num_mode;

    // Particle velocity.
    Kokkos::View<double[num_mode][3], TEST_MEMSPACE> pc( "pc" );

    // Create a grid scalar on the faces in the test dimension.
    auto grid_scalar = createArray( mesh, Location(), Bar() );

    // Initialize the grid scalar.
    auto gs_view = grid_scalar->view();
    Cajita::grid_parallel_for(
        "fill_grid_scalar", TEST_EXECSPACE(),
        grid_scalar->layout()->indexSpace( Cajita::Own(), Cajita::Local() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int ) {
            int ic = i - 2;
            int jc = j - 2;
            int kc = k - 2;
            gs_view( i, j, k, 0 ) =
                0.0000000001 * ( Dim + 1 ) * pow( ic, 5 ) -
                0.0000000012 * pow( ( Dim + 1 ) + jc * ic, 3 ) +
                0.0000000001 * pow( ic * jc * kc, 2 ) + pow( ( Dim + 1 ), 2 );
        } );

    // Check the grid velocity. Computed in Mathematica.
    auto gs_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gs_view );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gs_host, near_eps, Dim, 0 );

    // Do G2P.
    auto gs_wrapper =
        createViewWrapper( FieldLayout<Location, Bar>(), gs_view );
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd =
                createSpline( Location(), InterpolationOrder<Order>(),
                              local_mesh, x, SplineValue(), SplineGradient() );

            LinearAlgebra::Matrix<double, num_mode, 3> modes = 0.0;
            PolyPIC::g2p( gs_wrapper, modes, sd );

            for ( int r = 0; r < num_mode; ++r )
                for ( int d = 0; d < 3; ++d )
                    pc( r, d ) = modes( r, d );
        } );

    // Check particle velocity. Computed in Mathematica.
    auto pc_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pc );
    checkParticleVelocity( std::integral_constant<int, Order>(), pc_host,
                           near_eps, Dim );

    // Reset the particle velocity with the Mathematica results so we can do
    // the Lagrangian backtracking calculation correctly. We don't have the
    // other velocity components in this test as we check
    // dimension-by-dimension so we set them here. We just checked the
    // dimension of particle velocity we are testing so we know it was right.
    setParticleVelocity( std::integral_constant<int, Order>(), pc_host );
    Kokkos::deep_copy( pc, pc_host );

    // Reset the grid view.
    Kokkos::deep_copy( gs_view, 0.0 );

    // Create grid mass on the entities.
    auto grid_mass = createArray( mesh, Location(), Baz() );
    auto gm_view = grid_mass->view();
    Kokkos::deep_copy( gm_view, 0.0 );

    // Do P2G
    auto gs_sv = Kokkos::Experimental::create_scatter_view( gs_view );
    auto gm_sv = Kokkos::Experimental::create_scatter_view( gm_view );
    Kokkos::parallel_for(
        "p2g", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline( Location(), InterpolationOrder<Order>(),
                                    local_mesh, x, SplineValue(),
                                    SplineGradient(), SplineDistance() );

            LinearAlgebra::Matrix<double, num_mode, 3> modes;
            for ( int r = 0; r < num_mode; ++r )
                for ( int d = 0; d < 3; ++d )
                    modes( r, d ) = pc( r, d );

            PolyPIC::p2g( pm, modes, modes, gs_sv, gm_sv, dt, sd );
        } );
    Kokkos::Experimental::contribute( gs_view, gs_sv );
    Kokkos::Experimental::contribute( gm_view, gm_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gs_host, gs_view );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gs_host, near_eps, Dim, 0 );

    // Check grid mass. Computed in Mathematica.
    auto gm_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gm_view );
    checkGridMass( std::integral_constant<int, Order>(), cx, cy, cz, gm_host,
                   near_eps );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linear_collocated_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    collocatedTest<FieldLocation::Node, 1>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linear_staggered_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    staggeredTest<FieldLocation::Edge<Dim::I>, 1>();
    staggeredTest<FieldLocation::Edge<Dim::J>, 1>();
    staggeredTest<FieldLocation::Edge<Dim::K>, 1>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
