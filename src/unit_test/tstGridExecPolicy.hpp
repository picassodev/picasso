#include <Harlow_Types.hpp>
#include <Harlow_GridBlock.hpp>
#include <Harlow_GridExecPolicy.hpp>
#include <Harlow_GridField.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void parallelTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<int> num_cell = { 17, 25, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { halo_width, halo_width, halo_width };
    std::vector<int> local_cell_end =
        { num_cell[0] - halo_width, num_cell[1] - halo_width, num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] + 1};
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Make a cell field and a node field.
    auto cell_field = createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Cell );
    auto node_field = createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Node );

    // Change every value to 1 in both fields.
    auto cell_policy =
        GridExecution::createEntityPolicy<TEST_EXECSPACE>( grid, MeshEntity::Cell );
    Kokkos::parallel_for(
        "cell_fill",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            cell_field(i,j,k,0) = 1.0;
        });

    auto node_policy =
        GridExecution::createEntityPolicy<TEST_EXECSPACE>( grid, MeshEntity::Node );
    Kokkos::parallel_for(
        "node_fill",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            node_field(i,j,k,0) = 1.0;
        });

    auto cell_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cell_field );
    for ( int i = 0; i < num_cell[0]; ++i )
        for ( int j = 0; j < num_cell[1]; ++j )
            for ( int k = 0; k < num_cell[2]; ++k )
                EXPECT_EQ( cell_mirror(i,j,k,0), 1.0 );

    auto node_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), node_field );
    for ( int i = 0; i < num_node[0]; ++i )
        for ( int j = 0; j < num_node[1]; ++j )
            for ( int k = 0; k < num_node[2]; ++k )
                EXPECT_EQ( node_mirror(i,j,k,0), 1.0 );

    // Now do a reduction to sum all of the values in the fields.
    double cell_sum = 0.0;
    Kokkos::parallel_reduce(
        "cell_sum",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            result += cell_field(i,j,k,0); },
        cell_sum );
    double expected_cell_sum = num_cell[0] * num_cell[1] * num_cell[2];
    EXPECT_EQ( cell_sum, expected_cell_sum );

    double node_sum = 0.0;
    Kokkos::parallel_reduce(
        "node_sum",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            result += node_field(i,j,k,0); },
        node_sum );
    double expected_node_sum = num_node[0] * num_node[1] * num_node[2];
    EXPECT_EQ( node_sum, expected_node_sum );

    // Set a random value to be large and check that a max operation works.
    double max_val = 4.34;
    cell_mirror( 3, 5, 2, 0 ) = max_val;
    Kokkos::deep_copy( cell_field, cell_mirror );
    double cell_max_result = 0.0;
    Kokkos::Max<double> cell_max_reducer( cell_max_result );
    Kokkos::parallel_reduce(
        "cell_max",
        cell_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            if( cell_field(i,j,k,0) > result ) result = cell_field(i,j,k,0); },
        cell_max_reducer );
    EXPECT_EQ( cell_max_result, max_val );

    node_mirror( 3, 5, 2, 0 ) = max_val;
    Kokkos::deep_copy( node_field, node_mirror );
    double node_max_result = 0.0;
    Kokkos::Max<double> node_max_reducer( node_max_result );
    Kokkos::parallel_reduce(
        "node_max",
        node_policy,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ){
            if( node_field(i,j,k,0) > result ) result = node_field(i,j,k,0); },
        node_max_reducer );
    EXPECT_EQ( node_max_result, max_val );

    // Check the local execution policies.
    auto local_cell_policy =
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>( grid, MeshEntity::Cell );
    auto local_node_policy =
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>( grid, MeshEntity::Node );
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
    std::vector<int> num_cell = { 17, 25, 18 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, true, true};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 2;
    std::vector<int> local_cell_begin = { halo_width, halo_width, halo_width };
    std::vector<int> local_cell_end = { num_cell[0] - halo_width,
                                        num_cell[1] - halo_width,
                                        num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] + 1};
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // -------
    // Check the boundary execution policies.

    // -X
    auto xm_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowX );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[0], halo_width );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[0], halo_width + 1 );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( xm_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( xm_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto xm_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowX );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[0], halo_width );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[0], halo_width + 1 );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( xm_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( xm_bnd_node_policy.m_upper[2], num_node[2] );

    // +X
    auto xp_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighX );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[0], num_cell[0] - halo_width - 1 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[0], num_cell[0] - halo_width );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( xp_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( xp_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto xp_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighX );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[0], num_node[0] - halo_width - 1 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[0], num_node[0] - halo_width );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( xp_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( xp_bnd_node_policy.m_upper[2], num_node[2] );

    // -Y
    auto ym_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowY );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[1], halo_width );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[1], halo_width + 1 );
    EXPECT_EQ( ym_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( ym_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto ym_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowY );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[1], halo_width );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[1], halo_width + 1 );
    EXPECT_EQ( ym_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( ym_bnd_node_policy.m_upper[2], num_node[2] );

    // +Y
    auto yp_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighY );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[1], num_cell[1] - halo_width - 1 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[1], num_cell[1] - halo_width );
    EXPECT_EQ( yp_bnd_cell_policy.m_lower[2], 0 );
    EXPECT_EQ( yp_bnd_cell_policy.m_upper[2], num_cell[2] );

    auto yp_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighY );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[1], num_node[1] - halo_width - 1 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[1], num_node[1] - halo_width );
    EXPECT_EQ( yp_bnd_node_policy.m_lower[2], 0 );
    EXPECT_EQ( yp_bnd_node_policy.m_upper[2], num_node[2] );

    // -Z
    auto zm_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowZ );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( zm_bnd_cell_policy.m_lower[2], halo_width );
    EXPECT_EQ( zm_bnd_cell_policy.m_upper[2], halo_width + 1 );

    auto zm_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowZ );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( zm_bnd_node_policy.m_lower[2], halo_width );
    EXPECT_EQ( zm_bnd_node_policy.m_upper[2], halo_width + 1 );

    // +Z
    auto zp_bnd_cell_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighZ );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[0], 0 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[0], num_cell[0] );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[1], 0 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[1], num_cell[1] );
    EXPECT_EQ( zp_bnd_cell_policy.m_lower[2], num_cell[2] - halo_width - 1 );
    EXPECT_EQ( zp_bnd_cell_policy.m_upper[2], num_cell[2] - halo_width );

    auto zp_bnd_node_policy =
        GridExecution::createBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighZ );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[0], 0 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[0], num_node[0] );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[1], 0 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[1], num_node[1] );
    EXPECT_EQ( zp_bnd_node_policy.m_lower[2], num_node[2] - halo_width - 1 );
    EXPECT_EQ( zp_bnd_node_policy.m_upper[2], num_node[2] - halo_width );

    // -------
    // Check the local boundary execution policies.

    // -X
    auto xm_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowX );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[0], halo_width );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[0], halo_width + 1 );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( xm_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto xm_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowX );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[0], halo_width );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[0], halo_width + 1 );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( xm_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // +X
    auto xp_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighX );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[0], num_cell[0] - halo_width - 1 );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[0], num_cell[0] - halo_width );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( xp_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto xp_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighX );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[0], num_node[0] - halo_width - 1 );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[0], num_node[0] - halo_width );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( xp_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // -Y
    auto ym_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowY );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[1], halo_width );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[1], halo_width + 1 );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( ym_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto ym_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowY );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[1], halo_width );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[1], halo_width + 1 );
    EXPECT_EQ( ym_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( ym_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // +Y
    auto yp_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighY );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[1], num_cell[1] - halo_width - 1 );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[1], num_cell[1] - halo_width );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_lower[2], local_cell_begin[2] );
    EXPECT_EQ( yp_local_bnd_cell_policy.m_upper[2], local_cell_end[2] );

    auto yp_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighY );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[1], num_node[1] - halo_width - 1 );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[1], num_node[1] - halo_width );
    EXPECT_EQ( yp_local_bnd_node_policy.m_lower[2], local_node_begin[2] );
    EXPECT_EQ( yp_local_bnd_node_policy.m_upper[2], local_node_end[2] );

    // -Z
    auto zm_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::LowZ );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_lower[2], halo_width );
    EXPECT_EQ( zm_local_bnd_cell_policy.m_upper[2], halo_width + 1 );

    auto zm_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::LowZ );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( zm_local_bnd_node_policy.m_lower[2], halo_width );
    EXPECT_EQ( zm_local_bnd_node_policy.m_upper[2], halo_width + 1 );

    // +Z
    auto zp_local_bnd_cell_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Cell, DomainBoundary::HighZ );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[0], local_cell_begin[0] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[0], local_cell_end[0] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[1], local_cell_begin[1] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[1], local_cell_end[1] );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_lower[2], num_cell[2] - halo_width - 1 );
    EXPECT_EQ( zp_local_bnd_cell_policy.m_upper[2], num_cell[2] - halo_width );

    auto zp_local_bnd_node_policy =
        GridExecution::createLocalBoundaryEntityPolicy<TEST_EXECSPACE>(
            grid, MeshEntity::Node, DomainBoundary::HighZ );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[0], local_node_begin[0] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[0], local_node_end[0] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[1], local_node_begin[1] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[1], local_node_end[1] );
    EXPECT_EQ( zp_local_bnd_node_policy.m_lower[2], num_node[2] - halo_width - 1 );
    EXPECT_EQ( zp_local_bnd_node_policy.m_upper[2], num_node[2] - halo_width );
}

//---------------------------------------------------------------------------//
// RUN TESTS
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

} // end namespace Test
