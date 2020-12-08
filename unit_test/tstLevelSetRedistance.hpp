#include <Picasso_Types.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_UniformMesh.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_LevelSet.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
template<class Phi0, class PhiR>
void runTest( const Phi0& phi_0, const PhiR& phi_r )
{
    // Global parameters.
    Kokkos::Array<double,6> global_box = { 0.0, 0.0, 0.0,
                                           1.0, 1.0, 1.0 };

    // Get inputs for mesh.
    InputParser parser( "level_set_redistance_test.json", "json" );
    auto pt = parser.propertyTree();

    // Make mesh.
    int minimum_halo_size = 4;
    auto mesh = createUniformMesh( TEST_MEMSPACE(), pt, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );
    auto dx = mesh->localGrid()->globalGrid().globalMesh().cellSize(0);
    auto halo_width = dx * minimum_halo_size;

    // Create a level set.
    auto level_set = createLevelSet<FieldLocation::Node>( pt, mesh );

    // Signed distance fields.
    auto estimate = level_set->getDistanceEstimate();
    auto distance = level_set->getSignedDistance();

    // Field views.
    auto estimate_view = estimate->view();
    auto distance_view = distance->view();

    // Local mesh.
    auto local_mesh = Cajita::createLocalMesh<TEST_MEMSPACE>(
        *(mesh->localGrid()) );

    // Populate the initial estimate.
    auto own_entities = mesh->localGrid()->indexSpace(
        Cajita::Own(), Cajita::Node(), Cajita::Local() );
    Kokkos::parallel_for(
        "estimate",
        Cajita::createExecutionPolicy(own_entities,TEST_EXECSPACE()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){

            // Get the entity index.
            int entity_index[3] = {i,j,k};

            // Get the entity location.
            double x[3];
            local_mesh.coordinates( Cajita::Node(), entity_index, x );

            // Assign the estimate value.
            estimate_view(i,j,k,0) = phi_0( x[0], x[1], x[2] );
        });

    // Redistance.
    level_set->redistance( TEST_EXECSPACE() );

    // Test epsilon. Our grid is pretty coarse so this is pretty large with
    // respect to the analytic value. This still means we are resolving the to
    // a fraction ofthe cell width
    double test_eps = 0.5 * dx;

    // Check results. The should be correct if the signed distance magnitude
    // is less than the halo.
    auto host_distance = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), distance_view );
    auto host_mesh = Cajita::createLocalMesh<Kokkos::HostSpace>(
        *(mesh->localGrid()) );
    Kokkos::parallel_for(
        "test",
        Cajita::createExecutionPolicy(own_entities,Kokkos::Serial()),
        [=]( const int i, const int j, const int k ){

            // Get the entity index.
            int entity_index[3] = {i,j,k};

            // Get the entity location.
            double x[3];
            host_mesh.coordinates( Cajita::Node(), entity_index, x );

            // Observed result.
            auto observed = host_distance(i,j,k,0);

            // Expected result.
            auto expected = phi_r(x[0],x[1],x[2]);

            // Check the distance result in the narrow band. The
            // narrow-banding is done with the initial coarse grid estimate
            // but we want to check against the expected result so we dont
            // want to use the estimate to determine the narrow band here as
            // it may be wrong. So here we check for correctness within the
            // narrow band minus our tolerance.
            if ( fabs(expected) < halo_width - test_eps )
            {
                EXPECT_NEAR( expected, observed, test_eps );
            }

            // Outside the narrow band we will only get the sign right.
            else
            {
                if ( expected > 0.0 )
                {
                    EXPECT_TRUE( observed > 0.0 );
                }
                else
                {
                    EXPECT_TRUE( observed < 0.0 );
                }
            }
        });

    Cajita::BovWriter::Experimental::writeTimeStep( 0, 0.0, *estimate );
    Cajita::BovWriter::Experimental::writeTimeStep( 0, 0.0, *distance );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sphere_redistance_good_guess_test )
{
    // Sphere with radius of 0.25 centered at (0.5,0.5,0.5)

    // Actual distance. Use this as the guess to see if we get it the same
    // thing out within our discrete error bounds.
    auto phi_r =
        KOKKOS_LAMBDA( const double x, const double y, const double z ){

        double dx = 0.5 - x;
        double dy = 0.5 - y;
        double dz = 0.5 - z;
        double r = sqrt( dx*dx + dy*dy + dz*dz );

        return r - 0.25;
   };

    // Test
    runTest( phi_r, phi_r );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, box_redistance_good_guess_test )
{
    // Box of sides 0.5 centered at (0.5,0.5,0.5)

    // Actual distance. Use this as the guess to see if we get it the same
    // thing out within our discrete error bounds.
    auto phi_r =
        KOKKOS_LAMBDA( const double x, const double y, const double z ){

        // Compute distance to each box face.
        double dxlo = 0.25 - x;
        double dxhi = x - 0.75;

        double dylo = 0.25 - y;
        double dyhi = y - 0.75;

        double dzlo = 0.25 - z;
        double dzhi = z - 0.75;

        // Compute closest distance to the box.
        double dx = fmax(dxlo,dxhi);
        double dy = fmax(dylo,dyhi);
        double dz = fmax(dzlo,dzhi);

        // Inside.
        if ( dx <= 0.0 && dy <= 0.0 && dz <= 0.0 )
        {
            return fmax( dx, fmax(dy,dz) );
        }

        // Outside
        else
        {
            double xval = (dx > 0.0) ? dx : 0.0;
            double yval = (dy > 0.0) ? dy : 0.0;
            double zval = (dz > 0.0) ? dz : 0.0;
            return sqrt( xval*xval + yval*yval + zval*zval );
        }
    };

    // Test
    runTest( phi_r, phi_r );
}

//---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, sphere_redistance_test )
// {
//     // Sphere with radius of 0.25 centered at (0.5,0.5,0.5)

//     // Initial data. We interpret negative values to be inside the level
//     // set. In this function we then just get inside/outside correct.
//     auto phi_0 =
//         KOKKOS_LAMBDA( const double x, const double y, const double z ){

//         double dx = 0.5 - x;
//         double dy = 0.5 - y;
//         double dz = 0.5 - z;
//         double r = sqrt( dx*dx + dy*dy + dz*dz );

//         return ( r < 0.25 ) ? -1.0 : 1.0;
//     };

//     // Actual distance.
//     auto phi_r =
//         KOKKOS_LAMBDA( const double x, const double y, const double z ){

//         double dx = 0.5 - x;
//         double dy = 0.5 - y;
//         double dz = 0.5 - z;
//         double r = sqrt( dx*dx + dy*dy + dz*dz );

//         return r - 0.25;
//    };

//     // Test
//     runTest( phi_0, phi_r );
// }

//---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, box_redistance_test )
// {
//     // Box of sides 0.5 centered at (0.5,0.5,0.5)

//     // Initial data. We interpret negative values to be inside the level
//     // set. In this function we then just get inside/outside correct.
//     auto phi_0 =
//         KOKKOS_LAMBDA( const double x, const double y, const double z ){

//         if ( 0.25 <= x && x <= 0.75 &&
//              0.25 <= y && y <= 0.75 &&
//              0.25 <= z && z <= 0.75 )
//         {
//             return -1.0;
//         }
//         else
//         {
//             return 1.0;
//         }
//     };

//     // Actual distance.
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

//     // Test
//     runTest( phi_0, phi_r );
// }

//---------------------------------------------------------------------------//

} // end namespace Test
