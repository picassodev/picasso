#include <Harlow_Types.hpp>
#include <Harlow_GridBlock.hpp>
#include <Harlow_GridExecPolicy.hpp>
#include <Harlow_GridField.hpp>
#include <Harlow_ParticleInterpolation.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
template<int SplineOrder>
void fillTest()
{
    // Make a cartesian grid.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 4;
    GridBlock grid( low_corner, num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Calculate the low corners of the node primal grid. This includes the halo.
    std::vector<double> node_low_corner =
        { low_corner[Dim::I] - halo_width * cell_size,
          low_corner[Dim::J] - halo_width * cell_size,
          low_corner[Dim::K] - halo_width * cell_size };

    // Put a particle just off of the center of each local cell.
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

    // Make a node field.
    auto scalar_node_field =
        createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Node );

    // Make a particle field.
    using ScalarViewType = Kokkos::View<float*,TEST_MEMSPACE>;
    ScalarViewType scalar_p( "scalar_p", num_particle );
    Kokkos::deep_copy( scalar_p, 1.0 );

    // Interpolate to the nodes.
    auto scalar_p_accessor = ParticleGrid::createParticleViewAccessor( scalar_p );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, grid.inverseCellSize(),
        scalar_p_accessor, scalar_node_field );

    // Check that only nodes we expect to get data actually got data.
    auto scalar_node_field_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), scalar_node_field );

    // Linear case:
    if ( FunctionOrder::Linear == SplineOrder )
    {
        for ( int i = 0; i < grid.numEntity(MeshEntity::Node,Dim::I); ++i )
            for ( int j = 0; j < grid.numEntity(MeshEntity::Node,Dim::J); ++j )
                for ( int k = 0; k < grid.numEntity(MeshEntity::Node,Dim::K); ++k )
                {
                    // Negative halo nodes should be zero.
                    if ( i < halo_width || j < halo_width || k < halo_width )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Positive halo nodes not attached to a local cell should be
                    // zero.
                    else if ( i >= grid.numEntity(MeshEntity::Node,Dim::I) - halo_width ||
                              j >= grid.numEntity(MeshEntity::Node,Dim::J) - halo_width ||
                              k >= grid.numEntity(MeshEntity::Node,Dim::K) - halo_width )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Otherwise we should have gotten some data.
                    else
                        EXPECT_TRUE( scalar_node_field_mirror(i,j,k,0) > 0.0 );
                }
    }

    // Quadratic:
    if ( FunctionOrder::Quadratic == SplineOrder )
    {
        for ( int i = 0; i < grid.numEntity(MeshEntity::Node,Dim::I); ++i )
            for ( int j = 0; j < grid.numEntity(MeshEntity::Node,Dim::J); ++j )
                for ( int k = 0; k < grid.numEntity(MeshEntity::Node,Dim::K); ++k )
                {
                    // Negative halo nodes should be zero.
                    if ( i < halo_width - 1 || j < halo_width - 1 || k < halo_width - 1 )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Positive halo nodes not attached to a local cell should be
                    // zero.
                    else if ( i >= grid.numEntity(MeshEntity::Node,Dim::I) - halo_width ||
                              j >= grid.numEntity(MeshEntity::Node,Dim::J) - halo_width ||
                              k >= grid.numEntity(MeshEntity::Node,Dim::K) - halo_width )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Otherwise we should have gotten some data.
                    else
                        EXPECT_TRUE( scalar_node_field_mirror(i,j,k,0) > 0.0 );
                }
    }

    // Cubic:
    if ( FunctionOrder::Cubic == SplineOrder )
    {
        for ( int i = 0; i < grid.numEntity(MeshEntity::Node,Dim::I); ++i )
            for ( int j = 0; j < grid.numEntity(MeshEntity::Node,Dim::J); ++j )
                for ( int k = 0; k < grid.numEntity(MeshEntity::Node,Dim::K); ++k )
                {
                    // Negative halo nodes should be zero.
                    if ( i < halo_width - 1 || j < halo_width - 1 || k < halo_width - 1 )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Positive halo nodes not attached to a local cell should be
                    // zero.
                    else if ( i > grid.numEntity(MeshEntity::Node,Dim::I) - halo_width ||
                              j > grid.numEntity(MeshEntity::Node,Dim::J) - halo_width ||
                              k > grid.numEntity(MeshEntity::Node,Dim::K) - halo_width )
                        EXPECT_FLOAT_EQ( scalar_node_field_mirror(i,j,k,0), 0.0 );

                    // Otherwise we should have gotten some data.
                    else
                        EXPECT_TRUE( scalar_node_field_mirror(i,j,k,0) > 0.0 );
                }
    }
}

//---------------------------------------------------------------------------//
template<int SplineOrder>
void particleToGridTest()
{
    // Make a cartesian grid.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 4;
    GridBlock grid( low_corner, num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Calculate the low corners of the node primal grid. This includes the halo.
    std::vector<double> node_low_corner =
        { low_corner[Dim::I] - halo_width * cell_size,
          low_corner[Dim::J] - halo_width * cell_size,
          low_corner[Dim::K] - halo_width * cell_size };

    // Create particles randomly inside of the grid.
    double box_min_i = low_corner[Dim::I];
    double box_min_j = low_corner[Dim::J];
    double box_min_k = low_corner[Dim::K];
    double box_max_i = low_corner[Dim::I] + num_cell[Dim::I] * cell_size;
    double box_max_j = low_corner[Dim::J] + num_cell[Dim::J] * cell_size;
    double box_max_k = low_corner[Dim::K] + num_cell[Dim::K] * cell_size;
    int num_particle = 3204;
    Kokkos::View<double*[3],TEST_EXECSPACE> position( "positions", num_particle );
    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    Kokkos::parallel_for(
        "test random fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p )
        {
            auto gen = pool.get_state();
            position( p, Dim::I ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_i, box_max_i );
            position( p, Dim::J ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_j, box_max_j );
            position( p, Dim::K ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_k, box_max_k );
            pool.free_state( gen );
        } );

    // Scalar particle-to-grid
    // --------------

    // Make a node field.
    auto scalar_node_field =
        createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Node );

    // Make a particle field.
    using ScalarViewType = Kokkos::View<float*,TEST_MEMSPACE>;
    ScalarViewType scalar_p( "scalar_p", num_particle );
    float p_sum = 0.0;
    Kokkos::parallel_reduce(
        "scalar particle fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p, float& result )
        {
            scalar_p(p) = p + 1;
            result += scalar_p(p);
        },
        p_sum );

    // Interpolate to the nodes.
    auto scalar_p_accessor = ParticleGrid::createParticleViewAccessor( scalar_p );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, grid.inverseCellSize(),
        scalar_p_accessor, scalar_node_field );

    // Check that the node data sums to the particle data sum.
    double scalar_grid_sum = 0.0;
    Kokkos::parallel_reduce(
        "scalar grid sum",
        GridExecution::createEntityPolicy<TEST_EXECSPACE>(grid,MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result )
        {
            result += scalar_node_field(i,j,k,0);
        },
        scalar_grid_sum );
    EXPECT_FLOAT_EQ( scalar_grid_sum, p_sum );

    // Vector particle-to-grid
    // --------------

    // Make a node field.
    auto vector_node_field =
        createField<double,TEST_MEMSPACE>( grid, 2, MeshEntity::Node );

    // Make a particle field.
    using VectorViewType = Kokkos::View<float*[2],TEST_MEMSPACE>;
    VectorViewType vector_p( "vector_p", num_particle );
    Kokkos::parallel_for(
        "vector particle fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p )
        {
            vector_p(p,0) = p + 1;
            vector_p(p,1) = p + 1;
        } );

    // Interpolate to the nodes.
    auto vector_p_accessor = ParticleGrid::createParticleViewAccessor( vector_p );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, grid.inverseCellSize(),
        vector_p_accessor, vector_node_field );

    // Check that the node data sums to the particle data sum.
    double vector_grid_sum = 0.0;
    Kokkos::parallel_reduce(
        "vector grid sum",
        GridExecution::createEntityPolicy<TEST_EXECSPACE>(grid,MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result )
        {
            result += vector_node_field(i,j,k,0);
            result += vector_node_field(i,j,k,1);
        },
        vector_grid_sum );
    EXPECT_FLOAT_EQ( vector_grid_sum, 2.0 * p_sum );
}

//---------------------------------------------------------------------------//
template<int SplineOrder>
void gridToParticleTest()
{
    // Make a cartesian grid.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 4;
    GridBlock grid( low_corner, num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Calculate the low corners of the node primal grid. This includes the halo.
    std::vector<double> node_low_corner =
        { low_corner[Dim::I] - halo_width * cell_size,
          low_corner[Dim::J] - halo_width * cell_size,
          low_corner[Dim::K] - halo_width * cell_size };

    // Create particles randomly inside of the grid.
    double box_min_i = low_corner[Dim::I];
    double box_min_j = low_corner[Dim::J];
    double box_min_k = low_corner[Dim::K];
    double box_max_i = low_corner[Dim::I] + num_cell[Dim::I] * cell_size;
    double box_max_j = low_corner[Dim::J] + num_cell[Dim::J] * cell_size;
    double box_max_k = low_corner[Dim::K] + num_cell[Dim::K] * cell_size;
    int num_particle = 3204;
    Kokkos::View<double*[3],TEST_EXECSPACE> position( "positions", num_particle );
    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    Kokkos::parallel_for(
        "test random fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
        KOKKOS_LAMBDA( const int p )
        {
            auto gen = pool.get_state();
            position( p, Dim::I ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_i, box_max_i );
            position( p, Dim::J ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_j, box_max_j );
            position( p, Dim::K ) =
                Kokkos::rand<RandomType,double>::draw( gen, box_min_k, box_max_k );
            pool.free_state( gen );
        } );

    // Scalar grid-to-particle
    // --------------

    // Make a node field.
    auto scalar_node_field =
        createField<float,TEST_MEMSPACE>( grid, 1, MeshEntity::Node );
    double grid_value_0 = 1.2303;
    Kokkos::parallel_for(
        "scalar grid fill",
        GridExecution::createEntityPolicy<TEST_EXECSPACE>(grid,MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            scalar_node_field(i,j,k,0) = grid_value_0;
        } );

    // Make a particle field.
    using ScalarViewType = Kokkos::View<double*,TEST_MEMSPACE>;
    ScalarViewType scalar_p( "scalar_p", num_particle );

    // Interpolate to the nodes.
    auto scalar_g_accessor = ParticleGrid::createGridViewAccessor( scalar_node_field );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, grid.inverseCellSize(),
        scalar_g_accessor, scalar_p );

    // Check that the particles all got the grid value.
    auto scalar_p_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_p );
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_FLOAT_EQ( scalar_p_mirror(p), grid_value_0 );

    // Vector grid-to-particle
    // --------------

    // Make a node field.
    auto vector_node_field =
        createField<float,TEST_MEMSPACE>( grid, 2, MeshEntity::Node );
    double grid_value_1 = -34.32;
    Kokkos::parallel_for(
        "vector grid fill",
        GridExecution::createEntityPolicy<TEST_EXECSPACE>(grid,MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            vector_node_field(i,j,k,0) = grid_value_0;
            vector_node_field(i,j,k,1) = grid_value_1;
        } );

    // Make a particle field.
    using VectorViewType = Kokkos::View<double*[2],TEST_MEMSPACE>;
    VectorViewType vector_p( "vector_p", num_particle );

    // Interpolate to the nodes.
    auto vector_g_accessor = ParticleGrid::createGridViewAccessor( vector_node_field );
    ParticleGrid::interpolate<SplineOrder>(
        position, node_low_corner, grid.inverseCellSize(),
        vector_g_accessor, vector_p );

    // Check that the particles all got the grid value.
    auto vector_p_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_p );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_FLOAT_EQ( vector_p_mirror(p,0), grid_value_0 );
        EXPECT_FLOAT_EQ( vector_p_mirror(p,1), grid_value_1 );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, linear_test )
{
    fillTest<FunctionOrder::Linear>();
    particleToGridTest<FunctionOrder::Linear>();
    gridToParticleTest<FunctionOrder::Linear>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, quadratic_test )
{
    fillTest<FunctionOrder::Quadratic>();
    particleToGridTest<FunctionOrder::Quadratic>();
    gridToParticleTest<FunctionOrder::Quadratic>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, cubic_test )
{
    fillTest<FunctionOrder::Cubic>();
    particleToGridTest<FunctionOrder::Cubic>();
    gridToParticleTest<FunctionOrder::Cubic>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
