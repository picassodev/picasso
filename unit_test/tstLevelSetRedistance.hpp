#include <Harlow_Types.hpp>
#include <Harlow_InputParser.hpp>
#include <Harlow_UniformMesh.hpp>
#include <Harlow_LevelSetRedistance.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
template<class InitFunc>
void signedDistanceTest( const InitFunc& init_func,
                         const int num_secant_iter,
                         const int num_random,
                         const int num_eval_iter,
                         const std::string& test_name )
{
    // Get inputs.
    InputParser parser( "level_set_redistance_test.json", "json" );

    // Make mesh.
    int minimum_halo_size = 2;
    Kokkos::Array<double,6> global_box = { 0.0, 0.0, 0.0, 1.0, 1.0, 0.001953125 };
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(), global_box, minimum_halo_size, MPI_COMM_WORLD );
    auto local_mesh =
        Cajita::createLocalMesh<TEST_EXECSPACE>( *(mesh->localGrid()) );

    // Create the initial level set.
    std::string phi_0_name = test_name + "_phi_0";
    auto layout =
        Cajita::createArrayLayout( mesh->localGrid(), 1, Cajita::Node() );
    auto phi_0 =
        Cajita::createArray<double,TEST_MEMSPACE>( phi_0_name, layout );
    auto ghost_space = mesh->localGrid()->indexSpace(
        Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
    auto phi_0_view = phi_0->view();
    Kokkos::parallel_for(
        "fill_phi_0",
        Cajita::createExecutionPolicy(ghost_space,TEST_EXECSPACE()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            int index[3] = {i,j,k};
            double x[3];
            local_mesh.coordinates( Cajita::Node(), index, x );
            phi_0_view(i,j,k,0) = init_func( x );
        });

    // Write the initial level set.
    Cajita::BovWriter::writeTimeStep( 0, 0.0, *phi_0 );

    // Create the signed distance function.
    std::string phi_name = test_name + "_phi";
    auto phi =
        Cajita::createArray<double,TEST_MEMSPACE>( phi_name, layout );
    auto own_space = mesh->localGrid()->indexSpace(
        Cajita::Own(), Cajita::Node(), Cajita::Local() );
    auto phi_view = phi->view();
    double secant_tol = 1.0e-8;
    double projection_tol = 1.0e-8;
    Kokkos::parallel_for(
        "fill_phi",
        Cajita::createExecutionPolicy(own_space,TEST_EXECSPACE()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            int index[3] = {i,j,k};
            double x[3];
            local_mesh.coordinates( Cajita::Node(), index, x );
            phi_view(i,j,k,0) =
                LevelSet::redistanceEntity(
                    Cajita::Node(),
                    phi_0_view,
                    local_mesh,
                    index,
                    secant_tol,
                    num_secant_iter,
                    num_random,
                    projection_tol,
                    num_eval_iter );
        });

    // Write the signed distance function.
    Cajita::BovWriter::writeTimeStep( 0, 0.0, *phi );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, circle_test )
// {
//     auto init_func = KOKKOS_LAMBDA( const double x[3] ){
//         return exp((0.125 - pow(0.5-x[0],2) + pow(0.5-x[1],2)); };
//     signedDistanceTest( init_func, 10, 5, 100, "circle" );
// }

// TEST( TEST_CATEGORY, two_points_test )
// {
//     auto init_func = KOKKOS_LAMBDA( const double x[3] ){
//         double d1 = sqrt( pow(0.3-x[0],2) + pow(0.5-x[1],2) );
//         double d2 = sqrt( pow(0.7-x[0],2) + pow(0.5-x[1],2) );
//         return fmax( 0.25 - d1, 0.25 - d2 );
//     };
//     signedDistanceTest( init_func, 10, 5, 100, "two_points" );
// }

// TEST( TEST_CATEGORY, square_test )
// {
//     auto init_func = KOKKOS_LAMBDA( const double x[3] ){
//         return fmin( 0.25 - fabs(x[0]-0.5), 0.25 - fabs(x[1]-0.5) ); };
//     signedDistanceTest( init_func, 10, 4, 200, "square" );
// }

KOKKOS_INLINE_FUNCTION
double sine_func( const double x[3] )
{
    double pi = 4.0 * atan(1.0);
    return sin(4.0 * pi * x[0]) * sin(4.0 * pi * x[1]) - 0.01;
}

TEST( TEST_CATEGORY, sine_test )
{
    signedDistanceTest( sine_func, 10, 5, 200, "sine" );
}

//---------------------------------------------------------------------------//

} // end namespace Test
