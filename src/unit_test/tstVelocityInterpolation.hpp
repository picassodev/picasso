#include <Cajita_GridBlock.hpp>
#include <Cajita_GridExecPolicy.hpp>
#include <Cajita_GridField.hpp>

#include <Harlow_Types.hpp>
#include <Harlow_VelocityInterpolation.hpp>
#include <Harlow_ParticleInterpolation.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
template<int SplineOrder, int ParticleOrder>
void conservationTest()
{
    // Make a cartesian grid.
    std::vector<int> num_cell = { 4, 6, 5 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.2298;
    int halo_width = 4;
    Cajita::GridBlock block( low_corner, num_cell, boundary_location,
                             periodic, cell_size, halo_width );

    // Calculate the low corners of the node primal grid. This includes the halo.
    std::vector<double> node_low_corner =
        { low_corner[Dim::I] - halo_width * cell_size,
          low_corner[Dim::J] - halo_width * cell_size,
          low_corner[Dim::K] - halo_width * cell_size };

    // Create particles near the center of each local cell.
    int num_particle = num_cell[0] * num_cell[1] * num_cell[2];
    Kokkos::View<double*[3],TEST_EXECSPACE> position( "positions", num_particle );
    auto position_mirror = Kokkos::create_mirror_view( position );
    int pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
            {
                position_mirror( pid, Dim::I ) = low_corner[Dim::I] + (i+0.499) * cell_size;
                position_mirror( pid, Dim::J ) = low_corner[Dim::J] + (j+0.499) * cell_size;
                position_mirror( pid, Dim::K ) = low_corner[Dim::K] + (k+0.499) * cell_size;
            }
    Kokkos::deep_copy( position, position_mirror );

    // Make a grid mass field.
    auto grid_mass =
        Cajita::createField<double,TEST_MEMSPACE>( block, 1, MeshEntity::Node );

    // Make a grid velocity field.
    auto grid_velocity =
        Cajita::createField<double,TEST_MEMSPACE>( block, 3, MeshEntity::Node );

    // Initialize grid velocity field
    Kokkos::parallel_for(
        "init_grid_velocity",
        Cajita::GridExecution::createEntityPolicy<TEST_EXECSPACE>(block,MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            grid_velocity(i,j,k,Dim::I) = (i-6.)*(i-6.) + 3.2*i;
            grid_velocity(i,j,k,Dim::J) = (j-10.)*(j-10.) + 1.4*j;
            grid_velocity(i,j,k,Dim::K) = (k-5.)*(k-5.) - 0.2*k;
        } );

    // Make a particle mass field.
    double pmass = 1.4;
    Kokkos::View<double*,TEST_MEMSPACE> particle_mass( "particle_mass", num_particle );
    Kokkos::deep_copy( particle_mass, pmass );

    // Project particle mass to grid.
    auto mass_p_accessor = ParticleGrid::createParticleViewAccessor( particle_mass );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, block.inverseCellSize(),
        mass_p_accessor, grid_mass );

    // Make a particle velocity field.
    const int num_mode = VelocityInterpolation::PolyPicBasisTraits<ParticleOrder>::num_mode;
    Kokkos::View<double*[num_mode][3],TEST_MEMSPACE> particle_velocity(
        "particle_velocity", num_particle );

    // Compute the grid momentum.
    auto grid_velocity_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), grid_velocity );
    auto grid_mass_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), grid_mass );
    double gm_init_x = 0.0;
    double gm_init_y = 0.0;
    double gm_init_z = 0.0;
    for ( int i = 0; i < num_cell[0]+2*halo_width; ++i )
        for ( int j = 0; j < num_cell[1]+2*halo_width; ++j )
            for ( int k = 0; k < num_cell[2]+2*halo_width; ++k )
            {
                gm_init_x += grid_mass_mirror(i,j,k,0)*grid_velocity_mirror(i,j,k,Dim::I);
                gm_init_y += grid_mass_mirror(i,j,k,0)*grid_velocity_mirror(i,j,k,Dim::J);
                gm_init_z += grid_mass_mirror(i,j,k,0)*grid_velocity_mirror(i,j,k,Dim::K);
            }

    // Interpolate grid to particle.
    VelocityInterpolation::gridToParticle<SplineOrder,ParticleOrder>(
        block, grid_velocity, position, particle_velocity );

    // Compute initial particle momentum.
    double particle_momentum_x;
    Kokkos::parallel_reduce(
        "particle_momentum_sum_x",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p, double& result )
        {
            result += particle_velocity(p,0,Dim::I) * particle_mass(p);
        },
        particle_momentum_x );
    double particle_momentum_y;
    Kokkos::parallel_reduce(
        "particle_momentum_sum_y",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p, double& result )
        {
            result += particle_velocity(p,0,Dim::J) * particle_mass(p);
        },
        particle_momentum_y );
    double particle_momentum_z;
    Kokkos::parallel_reduce(
        "particle_momentum_sum_z",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p, double& result )
        {
            result += particle_velocity(p,0,Dim::K) * particle_mass(p);
        },
        particle_momentum_z );

    // Project particle momentum to grid. We do a zero size time step in this
    // case to simulate no advection.
    double dt = 0.0;
    VelocityInterpolation::particleToGrid<SplineOrder,ParticleOrder>(
        block, dt, position, particle_mass, particle_velocity, grid_mass, grid_velocity );

    // Compute grid momentum
    Kokkos::deep_copy( grid_velocity_mirror, grid_velocity );
    grid_mass_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), grid_mass );
    double gm_final_x = 0.0;
    double gm_final_y = 0.0;
    double gm_final_z = 0.0;
    double total_grid_mass = 0.0;
    for ( int i = 0; i < num_cell[0]+2*halo_width; ++i )
        for ( int j = 0; j < num_cell[1]+2*halo_width; ++j )
            for ( int k = 0; k < num_cell[2]+2*halo_width; ++k )
            {
                gm_final_x += grid_velocity_mirror(i,j,k,Dim::I);
                gm_final_y += grid_velocity_mirror(i,j,k,Dim::J);
                gm_final_z += grid_velocity_mirror(i,j,k,Dim::K);
                total_grid_mass += grid_mass_mirror(i,j,k,0);
            }

    // Check the mass conservation.
    EXPECT_FLOAT_EQ( total_grid_mass, num_particle * pmass );

    // Check momentum conservation for grid-to-particle.
    EXPECT_FLOAT_EQ( gm_init_x, particle_momentum_x );
    EXPECT_FLOAT_EQ( gm_init_y, particle_momentum_y );
    EXPECT_FLOAT_EQ( gm_init_z, particle_momentum_z );

    // Check momentum conservation for the full transfer cycle.
    EXPECT_FLOAT_EQ( gm_init_x, gm_final_x );
    EXPECT_FLOAT_EQ( gm_init_y, gm_final_y );
    EXPECT_FLOAT_EQ( gm_init_z, gm_final_z );
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, linear_spline_test )
{
    conservationTest<FunctionOrder::Linear,FunctionOrder::Linear>();
    conservationTest<FunctionOrder::Linear,FunctionOrder::Bilinear>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, Quadratic_spline_test )
{
    conservationTest<FunctionOrder::Quadratic,FunctionOrder::Linear>();
    conservationTest<FunctionOrder::Quadratic,FunctionOrder::Bilinear>();
    conservationTest<FunctionOrder::Quadratic,FunctionOrder::Quadratic>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, Cubic_spline_test )
{
    conservationTest<FunctionOrder::Cubic,FunctionOrder::Linear>();
    conservationTest<FunctionOrder::Cubic,FunctionOrder::Bilinear>();

    // Should this work? It seems as though it should not.
    // conservationTest<FunctionOrder::Cubic,FunctionOrder::Quadratic>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
