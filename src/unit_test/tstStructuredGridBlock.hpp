#include <Harlow_Types.hpp>
#include <Harlow_CartesianGridBlock.hpp>
#include <Harlow_StructuredGridBlock.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void fieldTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<int> num_cell = { 13, 21, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, false, false};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { 0, 0, halo_width };
    std::vector<int> local_cell_end =
        { num_cell[0], num_cell[1], num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] };
    CartesianGridBlock grid( low_corner, input_num_cell, boundary_location,
                             cell_size, halo_width );

    // Make some cell fields.
    auto scalar_cell_field = createCellField<double,TEST_MEMSPACE>( grid );
    EXPECT_EQ( scalar_cell_field.Rank, 3 );
    EXPECT_EQ( scalar_cell_field.extent(0), num_cell[0] );
    EXPECT_EQ( scalar_cell_field.extent(1), num_cell[1] );
    EXPECT_EQ( scalar_cell_field.extent(2), num_cell[2] );
    bool correct_value_type =
        std::is_same<double,
                     typename decltype(scalar_cell_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto vector_cell_field = createCellField<double[5],TEST_MEMSPACE>( grid );
    EXPECT_EQ( vector_cell_field.Rank, 4 );
    EXPECT_EQ( vector_cell_field.extent(0), num_cell[0] );
    EXPECT_EQ( vector_cell_field.extent(1), num_cell[1] );
    EXPECT_EQ( vector_cell_field.extent(2), num_cell[2] );
    EXPECT_EQ( vector_cell_field.extent(3), 5 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(vector_cell_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto tensor_cell_field = createCellField<double[2][4],TEST_MEMSPACE>( grid );
    EXPECT_EQ( tensor_cell_field.Rank, 5 );
    EXPECT_EQ( tensor_cell_field.extent(0), num_cell[0] );
    EXPECT_EQ( tensor_cell_field.extent(1), num_cell[1] );
    EXPECT_EQ( tensor_cell_field.extent(2), num_cell[2] );
    EXPECT_EQ( tensor_cell_field.extent(3), 2 );
    EXPECT_EQ( tensor_cell_field.extent(4), 4 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(tensor_cell_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    // Make some node fields.
    auto scalar_node_field = createNodeField<double,TEST_MEMSPACE>( grid );
    EXPECT_EQ( scalar_node_field.Rank, 3 );
    EXPECT_EQ( scalar_node_field.extent(0), num_node[0] );
    EXPECT_EQ( scalar_node_field.extent(1), num_node[1] );
    EXPECT_EQ( scalar_node_field.extent(2), num_node[2] );
    correct_value_type =
        std::is_same<double,
                     typename decltype(scalar_node_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto vector_node_field = createNodeField<double[5],TEST_MEMSPACE>( grid );
    EXPECT_EQ( vector_node_field.Rank, 4 );
    EXPECT_EQ( vector_node_field.extent(0), num_node[0] );
    EXPECT_EQ( vector_node_field.extent(1), num_node[1] );
    EXPECT_EQ( vector_node_field.extent(2), num_node[2] );
    EXPECT_EQ( vector_node_field.extent(3), 5 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(vector_node_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto tensor_node_field = createNodeField<double[2][4],TEST_MEMSPACE>( grid );
    EXPECT_EQ( tensor_node_field.Rank, 5 );
    EXPECT_EQ( tensor_node_field.extent(0), num_node[0] );
    EXPECT_EQ( tensor_node_field.extent(1), num_node[1] );
    EXPECT_EQ( tensor_node_field.extent(2), num_node[2] );
    EXPECT_EQ( tensor_node_field.extent(3), 2 );
    EXPECT_EQ( tensor_node_field.extent(4), 4 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(tensor_node_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );
}

//---------------------------------------------------------------------------//
void parallelTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<int> num_cell = { 13, 21, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, false, false};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { 0, 0, halo_width };
    std::vector<int> local_cell_end =
        { num_cell[0], num_cell[1], num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] };
    CartesianGridBlock grid( low_corner, input_num_cell, boundary_location,
                             cell_size, halo_width );

    // Make a cell field and a node field.
    auto cell_field = createCellField<double,TEST_MEMSPACE>( grid );
    auto node_field = createNodeField<double,TEST_MEMSPACE>( grid );

    // Change every value to 1 in both fields.
    auto cell_policy = createCellExecPolicy<TEST_EXECSPACE>( grid );
    Kokkos::parallel_for(
        "cell_fill",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            cell_field(i,j,k) = 1.0;
        });

    auto node_policy = createNodeExecPolicy<TEST_EXECSPACE>( grid );
    Kokkos::parallel_for(
        "node_fill",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            node_field(i,j,k) = 1.0;
        });

    auto cell_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cell_field );
    for ( int i = 0; i < num_cell[0]; ++i )
        for ( int j = 0; j < num_cell[1]; ++j )
            for ( int k = 0; k < num_cell[2]; ++k )
                EXPECT_EQ( cell_mirror(i,j,k), 1.0 );

    auto node_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), node_field );
    for ( int i = 0; i < num_node[0]; ++i )
        for ( int j = 0; j < num_node[1]; ++j )
            for ( int k = 0; k < num_node[2]; ++k )
                EXPECT_EQ( node_mirror(i,j,k), 1.0 );

    // Now do a reduction to sum all of the values in the fields.
    double cell_sum = 0.0;
    Kokkos::parallel_reduce(
        "cell_sum",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            result += cell_field(i,j,k); },
        cell_sum );
    double expected_cell_sum = num_cell[0] * num_cell[1] * num_cell[2];
    EXPECT_EQ( cell_sum, expected_cell_sum );

    double node_sum = 0.0;
    Kokkos::parallel_reduce(
        "node_sum",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            result += node_field(i,j,k); },
        node_sum );
    double expected_node_sum = num_node[0] * num_node[1] * num_node[2];
    EXPECT_EQ( node_sum, expected_node_sum );

    // Set a random value to be large and check that a max operation works.
    double max_val = 4.34;
    cell_mirror( 3, 5, 2 ) = max_val;
    Kokkos::deep_copy( cell_field, cell_mirror );
    double cell_max_result = 0.0;
    Kokkos::Max<double> cell_max_reducer( cell_max_result );
    Kokkos::parallel_reduce(
        "cell_max",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            if( cell_field(i,j,k) > result ) result = cell_field(i,j,k); },
        cell_max_reducer );
    EXPECT_EQ( cell_max_result, max_val );

    node_mirror( 3, 5, 2 ) = max_val;
    Kokkos::deep_copy( node_field, node_mirror );
    double node_max_result = 0.0;
    Kokkos::Max<double> node_max_reducer( node_max_result );
    Kokkos::parallel_reduce(
        "node_max",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            if( node_field(i,j,k) > result ) result = node_field(i,j,k); },
        node_max_reducer );
    EXPECT_EQ( node_max_result, max_val );

    // Check the local execution policies.
    auto local_cell_policy = createLocalCellExecPolicy<TEST_EXECSPACE>( grid );
    auto local_node_policy = createLocalNodeExecPolicy<TEST_EXECSPACE>( grid );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( local_cell_policy.m_lower[d], local_cell_begin[d] );
        EXPECT_EQ( local_cell_policy.m_upper[d], local_cell_end[d] );

        EXPECT_EQ( local_node_policy.m_lower[d], local_node_begin[d] );
        EXPECT_EQ( local_node_policy.m_upper[d], local_node_end[d] );
    }
}

//---------------------------------------------------------------------------//
void boundaryTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 14 };
    std::vector<int> num_cell = { 13, 21, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, true, true};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { 0, 0, 0 };
    std::vector<int> local_cell_end = num_cell;
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] + 1};
    CartesianGridBlock grid( low_corner, input_num_cell, boundary_location,
                             cell_size, halo_width );

    // -------
    // Check the boundary execution policies.

    // -X
    auto xm_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowX );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[0], 1 );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto xm_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowX );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[0], 1 );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[2], num_node[2] );

    // +X
    auto xp_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighX );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[0], num_cell[0] - 1 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto xp_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighX );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[0], num_node[0] - 1 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[2], num_node[2] );

    // -Y
    auto ym_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowY );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[1], 1 );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto ym_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowY );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[1], 1 );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[2], num_node[2] );

    // +Y
    auto yp_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighY );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[1], num_cell[1] - 1 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto yp_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighY );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[1], num_node[1] - 1 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[2], num_node[2] );

    // -Z
    auto zm_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowZ );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[2], 1 );

    auto zm_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowZ );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[2], 1 );

    // +Z
    auto zp_bnd_cell_policy = createCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighZ );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[2], num_cell[2] - 1 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto zp_bnd_node_policy = createNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighZ );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[2], num_node[2] - 1 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[2], num_node[2] );

    // -------
    // Check the local boundary execution policies.

    // -X
    auto xm_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowX );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[0], 1 );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto xm_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowX );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[0], 1 );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // +X
    auto xp_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighX );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[0], num_cell[0] - 1 );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto xp_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighX );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[0], num_node[0] - 1 );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // -Y
    auto ym_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowY );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[1], 1 );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto ym_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowY );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[1], 1 );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // +Y
    auto yp_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighY );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[1], num_cell[1] - 1 );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto yp_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighY );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[1], num_node[1] - 1 );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // -Z
    auto zm_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowZ );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[2], 1 );

    auto zm_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::LowZ );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[2], 1 );

    // +Z
    auto zp_local_bnd_cell_policy = createLocalCellBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighZ );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[2], num_cell[2] - 1 );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto zp_local_bnd_node_policy = createLocalNodeBoundaryExecPolicy<TEST_EXECSPACE>(
        grid, PhysicalBoundary::HighZ );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[2], num_node[2] - 1 );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[2], num_node[2] );
}

//---------------------------------------------------------------------------//
void haloTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 9, 17, 10 };
    std::vector<int> num_cell = { 13, 21, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { halo_width, halo_width, halo_width };
    std::vector<int> local_cell_end =
        { num_cell[0] - halo_width, num_cell[1] - halo_width, num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = local_cell_end;
    CartesianGridBlock grid( low_corner, input_num_cell, boundary_location,
                             cell_size, halo_width );

    // -------
    // Check the halo execution policies.
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            for ( int k = 0; k < 3; ++k )
            {
                std::vector<int> neighbor_id = {i,j,k};
                auto halo_cell_policy =
                    createHaloCellExecPolicy<TEST_EXECSPACE>( grid, neighbor_id );
                auto halo_node_policy =
                    createHaloNodeExecPolicy<TEST_EXECSPACE>( grid, neighbor_id );

                for ( int d = 0; d < 3; ++d )
                {
                    if ( LogicalBoundary::Negative == neighbor_id[d] )
                    {
                        EXPECT_EQ( halo_cell_policy.m_lower[d], 0 );
                        EXPECT_EQ( halo_cell_policy.m_upper[d], local_cell_begin[d] );

                        EXPECT_EQ( halo_node_policy.m_lower[d], 0 );
                        EXPECT_EQ( halo_node_policy.m_upper[d], local_node_begin[d] );
                    }
                    else if ( LogicalBoundary::Zero == neighbor_id[d] )
                    {
                        EXPECT_EQ( halo_cell_policy.m_lower[d], local_cell_begin[d] );
                        EXPECT_EQ( halo_cell_policy.m_upper[d], local_cell_end[d] );

                        EXPECT_EQ( halo_node_policy.m_lower[d], local_node_begin[d] );
                        EXPECT_EQ( halo_node_policy.m_upper[d], local_node_end[d] );
                    }
                    else if ( LogicalBoundary::Positive == neighbor_id[d] )
                    {
                        EXPECT_EQ( halo_cell_policy.m_lower[d], local_cell_end[d] );
                        EXPECT_EQ( halo_cell_policy.m_upper[d], num_cell[d] );

                        EXPECT_EQ( halo_node_policy.m_lower[d], local_node_end[d] );
                        EXPECT_EQ( halo_node_policy.m_upper[d], num_node[d] );
                    }
                }
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, field_test )
{
    fieldTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, parallel_test )
{
    parallelTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, boundary_test )
{
    boundaryTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, halo_test )
{
    haloTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
