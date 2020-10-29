#include <Picasso_UniformMesh.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_PolyPIC.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_ParticleInterpolation.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct Foo : Field::Vector<double,3>
{
    static std::string label() { return "foo"; }
};

struct Bar : Field::Scalar<double>
{
    static std::string label() { return "bar"; }
};

//---------------------------------------------------------------------------//
// Check the grid velocity. Computed in mathematica.
template<class GridVelocity>
void checkGridVelocity( const int cx, const int cy, const int cz,
                        const GridVelocity& gv_host,
                        const double near_eps,
                        const int test_dim, const int array_dim )
{
    if ( 0 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -1041.4514891968, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -1039.6577396436, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -1080.689462254, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -1078.8522294412, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -1068.8518602504, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -1067.0275793381, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -1109.1131484693, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -1107.2446441701, near_eps );
    }

    else if ( 1 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -1036.5091579932, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -1034.71540844, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -1075.7555034336, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -1073.9182706208, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -1063.8168117349, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -1061.9925308226, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -1104.0866148322, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -1102.218110533, near_eps );
    }

    else if ( 2 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -1029.5668973208, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -1027.7731477676, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -1068.821615994, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -1066.9843831812, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -1056.7818343482, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -1054.9575534359, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -1097.0601531807, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -1095.1916488815, near_eps );
    }
}

//---------------------------------------------------------------------------//
// Check particle velocity. Computed in Mathematica.
template<class ParticleVelocity>
void checkParticleVelocity( const ParticleVelocity& pc_host,
                            const double near_eps,
                            const int test_dim, const int array_dim )
{
    if ( 0 == test_dim )
    {
        EXPECT_NEAR( pc_host(0,array_dim), -1075.397343439538, near_eps );
        EXPECT_NEAR( pc_host(1,array_dim), -55.68151095982524, near_eps );
        EXPECT_NEAR( pc_host(2,array_dim), -79.71673394625162, near_eps );
        EXPECT_NEAR( pc_host(3,array_dim), 3.664027073586908, near_eps );
        EXPECT_NEAR( pc_host(4,array_dim), -4.092313283855674, near_eps );
        EXPECT_NEAR( pc_host(5,array_dim), 0.1234280604492142, near_eps );
        EXPECT_NEAR( pc_host(6,array_dim), 0.175768554104252, near_eps );
        EXPECT_NEAR( pc_host(7,array_dim), 0.005921018397202715, near_eps );
    }

    else if ( 1 == test_dim )
    {
        EXPECT_NEAR( pc_host(0,array_dim), -1070.401250223858, near_eps );
        EXPECT_NEAR( pc_host(1,array_dim), -55.49620173180062, near_eps );
        EXPECT_NEAR( pc_host(2,array_dim), -79.73365540669937, near_eps );
        EXPECT_NEAR( pc_host(3,array_dim), 3.664027073586681, near_eps );
        EXPECT_NEAR( pc_host(4,array_dim), -4.092883264654347, near_eps );
        EXPECT_NEAR( pc_host(5,array_dim), 0.1234280604487594, near_eps );
        EXPECT_NEAR( pc_host(6,array_dim), 0.1757685541047067, near_eps );
        EXPECT_NEAR( pc_host(7,array_dim), 0.005921018397202715, near_eps );
    }

    else if ( 2 == test_dim )
    {
        EXPECT_NEAR( pc_host(0,array_dim), -1063.405228285679, near_eps );
        EXPECT_NEAR( pc_host(1,array_dim), -55.31089370531271, near_eps );
        EXPECT_NEAR( pc_host(2,array_dim), -79.75057857527554, near_eps );
        EXPECT_NEAR( pc_host(3,array_dim), 3.664027073586794, near_eps );
        EXPECT_NEAR( pc_host(4,array_dim), -4.093453274253761, near_eps );
        EXPECT_NEAR( pc_host(5,array_dim), 0.1234280604492142, near_eps );
        EXPECT_NEAR( pc_host(6,array_dim), 0.1757685541047067, near_eps );
        EXPECT_NEAR( pc_host(7,array_dim), 0.005921018397202715, near_eps );
    }
}

//---------------------------------------------------------------------------//
// Check the grid momentum. Computed in mathematica.
template<class GridMomentum>
void checkGridMomentum( const int cx, const int cy, const int cz,
                        const GridMomentum& gv_host, const double near_eps,
                        const int test_dim, const int array_dim )
{
    if ( 0 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -20.18146901560989, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -9.48509613436825, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -16.46182244525835, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -7.736986121883586, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -33.80935969975699, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -15.8902629633603, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -27.57731616171574, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -12.96137223964792, near_eps );
    }

    else if ( 1 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -20.08564770559261, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -9.43999962836574, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -16.38665491043704, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -7.70160992921125, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -33.65007195751753, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -15.81529731320655, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -27.45236192744166, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -12.90256498233376, near_eps );
    }

    else if ( 2 == test_dim )
    {
        EXPECT_NEAR( gv_host(cx,cy,cz,array_dim), -19.95104709092533, near_eps );
        EXPECT_NEAR( gv_host(cx,cy,cz+1,array_dim), -9.37665403791526, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz,array_dim), -16.28101793506158, near_eps );
        EXPECT_NEAR( gv_host(cx,cy+1,cz+1,array_dim), -7.651895176351672, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz,array_dim), -33.42751273744686, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy,cz+1,array_dim), -15.71055685010869, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz,array_dim), -27.27769441071151, near_eps );
        EXPECT_NEAR( gv_host(cx+1,cy+1,cz+1,array_dim), -12.82036323927858, near_eps );
    }
}

//---------------------------------------------------------------------------//
void collocatedTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Global mesh parameters.
    Kokkos::Array<double,6> global_box = { -50.0, -50.0, -50.0,
                                           50.0, 50.0, 50.0 };

    // Get inputs for mesh.
    InputParser parser( "polypic_test.json", "json" );
    auto ptree = parser.propertyTree();

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh(
        ptree, global_box, minimum_halo_size, MPI_COMM_WORLD );
    auto local_mesh = Cajita::createLocalMesh<TEST_EXECSPACE>(
        *(mesh.localGrid()) );

    // Time step size.
    double dt = 0.0001;

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 118;
    int cy = 83;
    int cz = 93;

    // Particle velocity.
    Kokkos::View<double[8][3],TEST_MEMSPACE> pc("pc");

    // Create a grid vector on the nodes.
    auto grid_vector = createArray( mesh, FieldLocation::Node(), Foo() );

    // Initialize the grid vector.
    auto gv_view = grid_vector->view();
    Cajita::grid_parallel_for(
        "fill_grid_vector",
        TEST_EXECSPACE(),
        grid_vector->layout()->indexSpace(Cajita::Own(),Cajita::Local()),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int d ){
            gv_view(i,j,k,d) = 0.0000000001*(d+1)*pow(i,5) -
                               0.0000000012*pow((d+1)+j*i,3) +
                               0.0000000001*pow(i*j*k,2) +
                               pow((d+1),2);
        });

    // Check the grid velocity. Computed in Mathematica.
    auto gv_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), gv_view );
    checkGridVelocity( cx, cy, cz, gv_host, near_eps, 0, 0 );
    checkGridVelocity( cx, cy, cz, gv_host, near_eps, 1, 1 );
    checkGridVelocity( cx, cy, cz, gv_host, near_eps, 2, 2 );

    // Do G2P.
    Kokkos::parallel_for(
        "g2p",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int ){
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline(
                FieldLocation::Node(),
                InterpolationOrder<1>(),
                local_mesh,
                x,
                SplineValue(), SplineGradient() );

            LinearAlgebra::Matrix<double,8,3> modes;
            PolyPIC::g2p( gv_view, modes, sd );

            for ( int r = 0; r < 8; ++r )
                for ( int d = 0; d < 3; ++d )
                    pc(r,d) = modes(r,d);
        });

    // Check particle velocity. Computed in Mathematica.
    auto pc_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), pc );
    checkParticleVelocity( pc_host, near_eps, 0, 0 );
    checkParticleVelocity( pc_host, near_eps, 1, 1 );
    checkParticleVelocity( pc_host, near_eps, 2, 2 );

    // Reset the grid view.
    Kokkos::deep_copy( gv_view, 0.0 );

    // Do P2G
    auto gv_sv = Kokkos::Experimental::create_scatter_view( gv_view );
    Kokkos::parallel_for(
        "p2g",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int ){
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline(
                FieldLocation::Node(),
                InterpolationOrder<1>(),
                local_mesh,
                x,
                SplineValue(), SplineGradient(), SplineDistance() );

            LinearAlgebra::Matrix<double,8,3> modes;
            for ( int r = 0; r < 8; ++r )
                for ( int d = 0; d < 3; ++d )
                    modes(r,d) = pc(r,d);

            PolyPIC::p2g( pm, modes, gv_sv, dt, sd );
        });
    Kokkos::Experimental::contribute( gv_view, gv_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gv_host, gv_view );
    checkGridMomentum( cx, cy, cz, gv_host, near_eps, 0, 0 );
    checkGridMomentum( cx, cy, cz, gv_host, near_eps, 1, 1 );
    checkGridMomentum( cx, cy, cz, gv_host, near_eps, 2, 2 );
}

//---------------------------------------------------------------------------//
template<int Dim>
void staggeredTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Global mesh parameters. Offset in the given face dimensions by half of
    // the cell width so the faces align with the expected node locations we
    // calculated in Mathematica.
    Kokkos::Array<double,6> global_box = { -50.0, -50.0, -50.0,
                                           50.0, 50.0, 50.0 };
    for ( int d = 0; d < 3; ++d )
    {
        if ( d != Dim )
        {
            global_box[d] -= 0.25;
            global_box[d+3] -= 0.25;
        }
    }

    // Get inputs for mesh.
    InputParser parser( "polypic_test.json", "json" );
    auto ptree = parser.propertyTree();

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh(
        ptree, global_box, minimum_halo_size, MPI_COMM_WORLD );
    auto local_mesh = Cajita::createLocalMesh<TEST_EXECSPACE>(
        *(mesh.localGrid()) );

    // Time step size.
    double dt = 0.0001;

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 118;
    int cy = 83;
    int cz = 93;

    // Particle velocity.
    Kokkos::View<double[8][3],TEST_MEMSPACE> pc("pc");

    // Create a grid scalar on the faces in the test dimension.
    auto grid_scalar = createArray( mesh, FieldLocation::Face<Dim>(), Bar() );

    // Initialize the grid scalar.
    auto gs_view = grid_scalar->view();
    Cajita::grid_parallel_for(
        "fill_grid_scalar",
        TEST_EXECSPACE(),
        grid_scalar->layout()->indexSpace(Cajita::Own(),Cajita::Local()),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int ){
            gs_view(i,j,k,0) = 0.0000000001*(Dim+1)*pow(i,5) -
                               0.0000000012*pow((Dim+1)+j*i,3) +
                               0.0000000001*pow(i*j*k,2) +
                               pow((Dim+1),2);
        });

    // Check the grid velocity. Computed in Mathematica.
    auto gs_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), gs_view );
    checkGridVelocity( cx, cy, cz, gs_host, near_eps, Dim, 0 );

    // Do G2P.
    Kokkos::parallel_for(
        "g2p",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int ){
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline(
                FieldLocation::Face<Dim>(),
                InterpolationOrder<1>(),
                local_mesh,
                x,
                SplineValue(), SplineGradient() );

            LinearAlgebra::Matrix<double,8,3> modes;
            PolyPIC::g2p( gs_view, modes, sd );

            for ( int r = 0; r < 8; ++r )
                for ( int d = 0; d < 3; ++d )
                    pc(r,d) = modes(r,d);
        });

    // Check particle velocity. Computed in Mathematica.
    auto pc_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), pc );
    checkParticleVelocity( pc_host, near_eps, Dim, Dim );

    // Reset the particle velocity with the Mathematica results so we can do
    // the Lagrangian backtracking calculation correctly. We don't have the
    // other velocity components in this test as we check
    // dimension-by-dimension so we set them here. We just checked the
    // dimension of particle velocity we are testing so we know it was right.
    pc_host(0,0) = -1075.397343439538;
    pc_host(1,0) = -55.68151095982524;
    pc_host(2,0) = -79.71673394625162;
    pc_host(3,0) = 3.664027073586908;
    pc_host(4,0) = -4.092313283855674;
    pc_host(5,0) = 0.1234280604492142;
    pc_host(6,0) = 0.175768554104252;
    pc_host(7,0) = 0.005921018397202715;

    pc_host(0,1) = -1070.401250223858;
    pc_host(1,1) = -55.49620173180062;
    pc_host(2,1) = -79.73365540669937;
    pc_host(3,1) = 3.664027073586681;
    pc_host(4,1) = -4.092883264654347;
    pc_host(5,1) = 0.1234280604487594;
    pc_host(6,1) = 0.1757685541047067;
    pc_host(7,1) = 0.005921018397202715;

    pc_host(0,2) = -1063.405228285679;
    pc_host(1,2) = -55.31089370531271;
    pc_host(2,2) = -79.75057857527554;
    pc_host(3,2) = 3.664027073586794;
    pc_host(4,2) = -4.093453274253761;
    pc_host(5,2) = 0.1234280604492142;
    pc_host(6,2) = 0.1757685541047067;
    pc_host(7,2) = 0.005921018397202715;

    Kokkos::deep_copy( pc, pc_host );

    // Reset the grid view.
    Kokkos::deep_copy( gs_view, 0.0 );

    // Do P2G
    auto gs_sv = Kokkos::Experimental::create_scatter_view( gs_view );
    Kokkos::parallel_for(
        "p2g",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int ){
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline(
                FieldLocation::Face<Dim>(),
                InterpolationOrder<1>(),
                local_mesh,
                x,
                SplineValue(), SplineGradient(), SplineDistance() );

            LinearAlgebra::Matrix<double,8,3> modes;
            for ( int r = 0; r < 8; ++r )
                for ( int d = 0; d < 3; ++d )
                    modes(r,d) = pc(r,d);

            PolyPIC::p2g( pm, modes, gs_sv, dt, sd );
        });
    Kokkos::Experimental::contribute( gs_view, gs_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gs_host, gs_view );
    checkGridMomentum( cx, cy, cz, gs_host, near_eps, Dim, 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, collocated_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    collocatedTest();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, staggered_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    staggeredTest<Dim::I>();
    staggeredTest<Dim::J>();
    staggeredTest<Dim::K>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
