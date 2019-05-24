#include <Cajita_Types.hpp>
#include <Cajita_GridBlock.hpp>

#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
// Fixture
class harlow_grid_block : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};

//---------------------------------------------------------------------------//
void apiTest()
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

    // Test the API.
    EXPECT_EQ( low_corner[0] - halo_width * cell_size,
               grid.lowCorner(Dim::I) );
    EXPECT_EQ( low_corner[1] - halo_width * cell_size,
               grid.lowCorner(Dim::J) );
    EXPECT_EQ( low_corner[2] - halo_width * cell_size,
               grid.lowCorner(Dim::K) );

    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_TRUE( grid.onBoundary(2*i) );
        EXPECT_TRUE( grid.onBoundary(2*i+1) );
    }
    EXPECT_FALSE( grid.onBoundary(4) );
    EXPECT_FALSE( grid.onBoundary(5) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FALSE( grid.isPeriodic(d) );

    EXPECT_EQ( cell_size, grid.cellSize() );
    EXPECT_EQ( 1.0 / cell_size, grid.inverseCellSize() );

    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( num_cell[d], grid.numEntity(MeshEntity::Cell,d) );
        EXPECT_EQ( num_node[d], grid.numEntity(MeshEntity::Node,d) );
        EXPECT_EQ( local_cell_begin[d], grid.localEntityBegin(MeshEntity::Cell,d) );
        EXPECT_EQ( local_cell_end[d], grid.localEntityEnd(MeshEntity::Cell,d) );
        EXPECT_EQ( local_node_begin[d], grid.localEntityBegin(MeshEntity::Node,d) );
        EXPECT_EQ( local_node_end[d], grid.localEntityEnd(MeshEntity::Node,d) );
        EXPECT_EQ( input_num_cell[d], grid.localNumEntity(MeshEntity::Cell,d) );
        EXPECT_EQ( local_node_end[d] - local_node_begin[d],
                   grid.localNumEntity(MeshEntity::Node,d) );
    }
}

//---------------------------------------------------------------------------//
void assignTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 2;
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Create a new grid and assign the old one with a different halo.
    GridBlock block_2;
    int halo_width_2 = 4;
    block_2.assign( grid, halo_width_2 );

    std::vector<int> num_cell = { 21, 29, 18 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<int> local_cell_begin = { halo_width_2, halo_width_2, halo_width_2 };
    std::vector<int> local_cell_end = { num_cell[0] - halo_width_2,
                                        num_cell[1] - halo_width_2,
                                        num_cell[2] - halo_width_2 };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] + 1};

    EXPECT_EQ( low_corner[0] - halo_width_2 * cell_size,
               block_2.lowCorner(Dim::I) );
    EXPECT_EQ( low_corner[1] - halo_width_2 * cell_size,
               block_2.lowCorner(Dim::J) );
    EXPECT_EQ( low_corner[2] - halo_width_2 * cell_size,
               block_2.lowCorner(Dim::K) );

    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_TRUE( block_2.onBoundary(2*i) );
        EXPECT_TRUE( block_2.onBoundary(2*i+1) );
    }
    EXPECT_FALSE( block_2.onBoundary(4) );
    EXPECT_FALSE( block_2.onBoundary(5) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FALSE( grid.isPeriodic(d) );

    EXPECT_EQ( cell_size, block_2.cellSize() );
    EXPECT_EQ( 1.0 / cell_size, block_2.inverseCellSize() );

    for ( int d = 0; d < 3; ++ d)
    {
        EXPECT_EQ( num_cell[d], block_2.numEntity(MeshEntity::Cell,d) );
        EXPECT_EQ( num_node[d], block_2.numEntity(MeshEntity::Node,d) );
        EXPECT_EQ( local_cell_begin[d], block_2.localEntityBegin(MeshEntity::Cell,d) );
        EXPECT_EQ( local_cell_end[d], block_2.localEntityEnd(MeshEntity::Cell,d) );
        EXPECT_EQ( local_node_begin[d], block_2.localEntityBegin(MeshEntity::Node,d) );
        EXPECT_EQ( local_node_end[d], block_2.localEntityEnd(MeshEntity::Node,d) );
    }
}

//---------------------------------------------------------------------------//
void periodicTest()
{
    // Make a cartesian grid with all physical boundaries periodic.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<int> num_cell = { 17, 25, 14 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, true, false};
    std::vector<bool> periodic = {true,true,true};
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

    // Test the API.
    EXPECT_EQ( low_corner[0] - halo_width * cell_size, grid.lowCorner(Dim::I) );
    EXPECT_EQ( low_corner[1] - halo_width * cell_size, grid.lowCorner(Dim::J) );
    EXPECT_EQ( low_corner[2] - halo_width * cell_size, grid.lowCorner(Dim::K) );

    for ( int i = 0; i < 5; ++i )
    {
        EXPECT_TRUE( grid.onBoundary(i) );
    }
    EXPECT_FALSE( grid.onBoundary(5) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_TRUE( grid.isPeriodic(d) );

    EXPECT_EQ( cell_size, grid.cellSize() );
    EXPECT_EQ( 1.0 / cell_size, grid.inverseCellSize() );

    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( num_cell[d], grid.numEntity(MeshEntity::Cell,d) );
        EXPECT_EQ( num_node[d], grid.numEntity(MeshEntity::Node,d) );
        EXPECT_EQ( local_cell_begin[d], grid.localEntityBegin(MeshEntity::Cell,d) );
        EXPECT_EQ( local_cell_end[d], grid.localEntityEnd(MeshEntity::Cell,d) );
        EXPECT_EQ( local_node_begin[d], grid.localEntityBegin(MeshEntity::Node,d) );
        EXPECT_EQ( local_node_end[d], grid.localEntityEnd(MeshEntity::Node,d) );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( harlow_grid_block, api_test )
{
    apiTest();
}

//---------------------------------------------------------------------------//
TEST_F( harlow_grid_block, assign_test )
{
    assignTest();
}

//---------------------------------------------------------------------------//
TEST_F( harlow_grid_block, periodic_test )
{
    periodicTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
