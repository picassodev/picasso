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
#include <Picasso_InputParser.hpp>
#include <Picasso_LevelSet.hpp>
#include <Picasso_MarchingCubes.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
struct ErrorField : public Field::Scalar<double>
{
    static std::string label() { return "error"; }
};

//---------------------------------------------------------------------------//
template <class Phi0>
void runTest( const Phi0& phi_0, const std::string& stl_filename )
{
    // Global parameters.
    Kokkos::Array<double, 6> global_box = { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };

    // Get inputs for mesh.
    auto inputs = parse( "level_set_redistance_test.json" );

    // Make mesh.
    int minimum_halo_size = 4;
    auto mesh = createUniformMesh( TEST_MEMSPACE{}, inputs, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );

    // Create a level set.
    auto level_set = createLevelSet<FieldLocation::Node>( inputs, mesh );

    // Signed distance fields.
    auto estimate = level_set->getDistanceEstimate();
    auto distance = level_set->getSignedDistance();

    // Field views.
    auto estimate_view = estimate->view();
    auto distance_view = distance->view();

    // Local mesh.
    auto local_mesh =
        Cajita::createLocalMesh<TEST_MEMSPACE>( *( mesh->localGrid() ) );

    // Populate the initial estimate.
    auto own_entities = mesh->localGrid()->indexSpace(
        Cajita::Own(), Cajita::Node(), Cajita::Local() );
    Kokkos::parallel_for(
        "estimate",
        Cajita::createExecutionPolicy( own_entities, TEST_EXECSPACE{} ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Get the entity index.
            int entity_index[3] = { i, j, k };

            // Get the entity location.
            double x[3];
            local_mesh.coordinates( Cajita::Node(), entity_index, x );

            // Assign the estimate value.
            estimate_view( i, j, k, 0 ) = phi_0( x[0], x[1], x[2] );
        } );

    // Redistance.
    level_set->redistance( TEST_EXECSPACE{} );

    // Create the marching cubes mesh.
    auto mc_data = MarchingCubes::createData( *mesh );
    MarchingCubes::build( TEST_EXECSPACE{}, *mesh, *distance, *mc_data );

    // Output an stl file with the marching cubes mesh.
    MarchingCubes::writeDataToSTL( *mc_data, MPI_COMM_WORLD, stl_filename );

    // Output a bov file with the level set.
    Cajita::Experimental::BovWriter::writeTimeStep( 0, 0.0, *distance );

    double area = 16.0 * std::atan( 1.0 ) * 0.25 * 0.25;
    double facet_area = 0.0;

    auto facets = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                                                       mc_data->facets );
    for ( std::size_t f = 0; f < facets.extent( 0 ); ++f )
    {
        Vec3<double> ab = { facets( f, 1, 0 ) - facets( f, 0, 0 ),
                            facets( f, 1, 1 ) - facets( f, 0, 1 ),
                            facets( f, 1, 2 ) - facets( f, 0, 2 ) };

        Vec3<double> ac = { facets( f, 2, 0 ) - facets( f, 0, 0 ),
                            facets( f, 2, 1 ) - facets( f, 0, 1 ),
                            facets( f, 2, 2 ) - facets( f, 0, 2 ) };

        auto cross = ab % ac;

        double c2 = ~cross * cross;
        facet_area += 0.5 * sqrt( c2 );
    }

    MPI_Allreduce( MPI_IN_PLACE, &facet_area, 1, MPI_DOUBLE, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_NEAR( area, facet_area, 0.01 );
}

void sphere_redistance()
{
    // Sphere with radius of 0.25 centered at (0.5,0.5,0.5)

    // Actual distance. Use this as the guess to see if we get it the same
    // thing out within our discrete error bounds.
    auto phi_r = KOKKOS_LAMBDA( const double x, const double y, const double z )
    {

        double dx = 0.5 - x;
        double dy = 0.5 - y;
        double dz = 0.5 - z;
        double r = sqrt( dx * dx + dy * dy + dz * dz );

        return r - 0.25;
    };

    // Test.
    runTest( phi_r, "spehere_mc_mesh.stl" );
}

//---------------------------------------------------------------------------//
void scaled_sphere_redistance()
{
    // Scaled sphere with radius of 0.25 centered at (0.5,0.5,0.5).

    // Initial data.
    auto phi_0 = KOKKOS_LAMBDA( const double x, const double y, const double z )
    {

        double dx = 0.5 - x;
        double dy = 0.5 - y;
        double dz = 0.5 - z;
        double r = sqrt( dx * dx + dy * dy + dz * dz );

        return 2.8 * ( exp( r - 0.25 ) - 1.0 );
    };

    // Test.
    runTest( phi_0, "scaled_spehere_mc_mesh.stl" );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sphere_redistance_good_guess_test )
{
    sphere_redistance();
}
TEST( TEST_CATEGORY, scaled_sphere_redistance_test )
{
    scaled_sphere_redistance();
}

//---------------------------------------------------------------------------//
/*
  The tests below are for level sets in which the zero isocontour is a
  cube. The first test inputs the analytic solution as the estimate will the
  second gives the distance to the closest surface as the estimate. Per the
  paper by Royston these tests should work pretty well. However, that paper
  was in 2D while these are 3D tests. As a result, a 2D cut plane down the
  center of the cube along any given Cartesian axis gets a nice result. The
  corners of the cube, however, see large errors due to the inability of the
  secant method to converge at those zero-features. We also see some errors
  along the cube edges as well.

  So, these tests point out some numerical issues with this method that was
  not exposed in the paper. Maybe the secant and projected gradient
  implementation isn't right (maybe a sign issue?) or perhaps some additional
  robustness measures can be added or another root-finding algorithm such as
  Newton leveraged.

  As a result these are commented out for now and for non-smooth isocontours a
  user should expect to encounter some issues.
 */
// //---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, box_redistance_good_guess_test )
// {
//     // Box of sides 0.5 centered at (0.5,0.5,0.5)

//     // Actual distance. Use this as the guess to see if we get it the same
//     // thing out within our discrete error bounds.
//     auto phi_r =
//         KOKKOS_LAMBDA( const double x, const double y, const double z ){

//         // Compute distance to each box face.
//         double dxlo = 0.25 - x;
//         double dxhi = x - 0.75;

//         double dylo = 0.25 - y;
//         double dyhi = y - 0.75;

//         double dzlo = 0.25 - z;
//         double dzhi = z - 0.75;

//         // Compute closest distance to the box.
//         double dx = fmax(dxlo,dxhi);
//         double dy = fmax(dylo,dyhi);
//         double dz = fmax(dzlo,dzhi);

//         // Inside.
//         if ( dx <= 0.0 && dy <= 0.0 && dz <= 0.0 )
//         {
//             return fmax( dx, fmax(dy,dz) );
//         }

//         // Outside
//         else
//         {
//             double xval = (dx > 0.0) ? dx : 0.0;
//             double yval = (dy > 0.0) ? dy : 0.0;
//             double zval = (dz > 0.0) ? dz : 0.0;
//             return sqrt( xval*xval + yval*yval + zval*zval );
//         }
//     };

//     // Test. The tolerance here is large because of the degenerate geometric
//     // features. We will need to revist this in the future.
//     runTest( phi_r, "box_good_guess_mc_mesh.stl" );
// }

// //---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, box_redistance_test )
// {
//     // Box of sides 0.5 centered at (0.5,0.5,0.5)

//     // Initial data. Only return the distance to the closest side.
//     auto phi_0 =
//         KOKKOS_LAMBDA( const double x, const double y, const double z ){

//         return -fmin( 0.25 - fabs(x-0.5),
//                       fmin( 0.25 - fabs(y-0.5), 0.25 - fabs(z-0.5) ) );
//     };

//     // Test. The tolerance here is large because of the degenerate geometric
//     // features. We will need to revisit this in the future.
//     runTest( phi_0, "box_mc_mesh.stl" );
// }

//---------------------------------------------------------------------------//

} // end namespace Test
