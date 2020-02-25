#include <Harlow_ParticleInit.hpp>
#include <Harlow_Types.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
int totalParticlesPerCell( InitUniform, int ppc )
{
    return ppc * ppc * ppc;
}

int totalParticlesPerCell( InitRandom, int ppc )
{
    return ppc;
}

//---------------------------------------------------------------------------//
template<class InitType>
void InitTest( InitType init_type, int ppc )
{
    // Create the grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 43, 32, 39 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner,
                                                        global_high_corner,
                                                        global_num_cell );
    std::array<bool,3> periodic = {false,false,false};
    auto global_grid =
        Cajita::createGlobalGrid( MPI_COMM_WORLD,
                                  global_mesh,
                                  periodic,
                                  Cajita::UniformDimPartitioner() );
    auto local_grid = Cajita::createLocalGrid( global_grid, 0 );

    // Create empty particle list.
    using particle_fields = Cabana::MemberTypes<double[3],int[3]>;
    using particle_list = Cabana::AoSoA<particle_fields,TEST_DEVICE>;
    using particle_type = typename particle_list::tuple_type;
    particle_list particles( "particles" );

    // Particle initialization functor.
    Kokkos::Array<double,6> box = { global_low_corner[Dim::I] + cell_size,
                                    global_high_corner[Dim::I] - cell_size,
                                    global_low_corner[Dim::J] + cell_size,
                                    global_high_corner[Dim::J] - cell_size,
                                    global_low_corner[Dim::K] + cell_size,
                                    global_high_corner[Dim::K] - cell_size };
    auto particle_init_func =
        KOKKOS_LAMBDA( const double x[3], particle_type& p ){
        // Put particles in a box that is one cell smaller than the global
        // mesh. This will give us a layer of empty cells.
        if ( x[Dim::I] > box[0] &&
             x[Dim::I] < box[1] &&
             x[Dim::J] > box[2] &&
             x[Dim::J] < box[3] &&
             x[Dim::K] > box[4] &&
             x[Dim::K] < box[5] )
        {
            for ( int d = 0; d < 3; ++d )
            {
                Cabana::get<0>(p,d) = x[d];
                Cabana::get<1>(p,d) = int(x[d]);
            }
            return true;
        }
        else
        {
            return false;
        }
    };

    // Initialize particles.
    initializeParticles( init_type, *local_grid, ppc, particle_init_func, particles );

    // Check that we made particles.
    int num_p = particles.size();
    EXPECT_TRUE( num_p > 0 );

    // Compute the global number of particles.
    int global_num_particle = num_p;
    MPI_Allreduce( MPI_IN_PLACE, &global_num_particle, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    int expect_num_particle =
        totalParticlesPerCell( init_type, ppc ) *
        (global_grid->globalNumEntity(Cajita::Cell(),Dim::I)-2) *
        (global_grid->globalNumEntity(Cajita::Cell(),Dim::J)-2) *
        (global_grid->globalNumEntity(Cajita::Cell(),Dim::K)-2);
    EXPECT_EQ( global_num_particle, expect_num_particle );

    // Check that all particles are in the box and got initialized correctly.
    auto host_particles = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particles );
    auto px = Cabana::slice<0>(host_particles);
    auto pi = Cabana::slice<1>(host_particles);
    for ( int p = 0; p < num_p; ++p )
    {
        EXPECT_TRUE( px(p,Dim::I) > box[0] );
        EXPECT_TRUE( px(p,Dim::I) < box[1] );
        EXPECT_TRUE( px(p,Dim::J) > box[2] );
        EXPECT_TRUE( px(p,Dim::J) < box[3] );
        EXPECT_TRUE( px(p,Dim::K) > box[4] );
        EXPECT_TRUE( px(p,Dim::K) < box[5] );

        EXPECT_EQ( int(px(p,Dim::I)), pi(p,Dim::I) );
        EXPECT_EQ( int(px(p,Dim::J)), pi(p,Dim::J) );
        EXPECT_EQ( int(px(p,Dim::K)), pi(p,Dim::K) );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, random_init_test )
{
    InitTest( InitRandom(), 17 );
}

TEST( TEST_CATEGORY, uniform_init_test )
{
    InitTest( InitUniform(), 3 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
