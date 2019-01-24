#include <Harlow_Types.hpp>
#include <Harlow_GridBlock.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

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
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    cell_size, halo_width );

    // Test the API.
    EXPECT_EQ( low_corner[0], grid.lowCorner(Dim::I) );
    EXPECT_EQ( low_corner[1], grid.lowCorner(Dim::J) );
    EXPECT_EQ( low_corner[2] - halo_width * cell_size,
               grid.lowCorner(Dim::K) );

    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_TRUE( grid.onBoundary(2*i) );
        EXPECT_TRUE( grid.onBoundary(2*i+1) );
    }
    EXPECT_FALSE( grid.onBoundary(4) );
    EXPECT_FALSE( grid.onBoundary(5) );

    EXPECT_EQ( cell_size, grid.cellSize() );
    EXPECT_EQ( 1.0 / cell_size, grid.inverseCellSize() );

    EXPECT_EQ( num_cell[0], grid.numCell(Dim::I) );
    EXPECT_EQ( num_cell[1], grid.numCell(Dim::J) );
    EXPECT_EQ( num_cell[2], grid.numCell(Dim::K) );

    EXPECT_EQ( num_node[0], grid.numNode(Dim::I) );
    EXPECT_EQ( num_node[1], grid.numNode(Dim::J) );
    EXPECT_EQ( num_node[2], grid.numNode(Dim::K) );

    EXPECT_EQ( local_cell_begin[0], grid.localCellBegin(Dim::I) );
    EXPECT_EQ( local_cell_begin[1], grid.localCellBegin(Dim::J) );
    EXPECT_EQ( local_cell_begin[2], grid.localCellBegin(Dim::K) );

    EXPECT_EQ( local_cell_end[0], grid.localCellEnd(Dim::I) );
    EXPECT_EQ( local_cell_end[1], grid.localCellEnd(Dim::J) );
    EXPECT_EQ( local_cell_end[2], grid.localCellEnd(Dim::K) );

    EXPECT_EQ( local_node_begin[0], grid.localNodeBegin(Dim::I) );
    EXPECT_EQ( local_node_begin[1], grid.localNodeBegin(Dim::J) );
    EXPECT_EQ( local_node_begin[2], grid.localNodeBegin(Dim::K) );

    EXPECT_EQ( local_node_end[0], grid.localNodeEnd(Dim::I) );
    EXPECT_EQ( local_node_end[1], grid.localNodeEnd(Dim::J) );
    EXPECT_EQ( local_node_end[2], grid.localNodeEnd(Dim::K) );
}

//---------------------------------------------------------------------------//
void assignTest()
{
    // Make a cartesian grid.
    std::vector<int> input_num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { true, true, true, true, false, false};
    double cell_size = 0.53;
    int halo_width = 2;
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    cell_size, halo_width );

    // Create a new grid and assign the old one with a different halo.
    GridBlock block_2;
    int halo_width_2 = 4;
    block_2.assign( grid, halo_width_2 );

    std::vector<int> num_cell = { 13, 21, 18 };
    std::vector<int> num_node = { num_cell[0]+1, num_cell[1]+1, num_cell[2]+1 };
    std::vector<int> local_cell_begin = { 0, 0, halo_width_2 };
    std::vector<int> local_cell_end =
        { num_cell[0], num_cell[1], num_cell[2] - halo_width_2 };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] };

    EXPECT_EQ( low_corner[0], block_2.lowCorner(Dim::I) );
    EXPECT_EQ( low_corner[1], block_2.lowCorner(Dim::J) );
    EXPECT_EQ( low_corner[2] - halo_width_2 * cell_size,
               block_2.lowCorner(Dim::K) );

    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_TRUE( block_2.onBoundary(2*i) );
        EXPECT_TRUE( block_2.onBoundary(2*i+1) );
    }
    EXPECT_FALSE( block_2.onBoundary(4) );
    EXPECT_FALSE( block_2.onBoundary(5) );

    EXPECT_EQ( cell_size, block_2.cellSize() );
    EXPECT_EQ( 1.0 / cell_size, block_2.inverseCellSize() );

    EXPECT_EQ( num_cell[0], block_2.numCell(Dim::I) );
    EXPECT_EQ( num_cell[1], block_2.numCell(Dim::J) );
    EXPECT_EQ( num_cell[2], block_2.numCell(Dim::K) );

    EXPECT_EQ( num_node[0], block_2.numNode(Dim::I) );
    EXPECT_EQ( num_node[1], block_2.numNode(Dim::J) );
    EXPECT_EQ( num_node[2], block_2.numNode(Dim::K) );

    EXPECT_EQ( local_cell_begin[0], block_2.localCellBegin(Dim::I) );
    EXPECT_EQ( local_cell_begin[1], block_2.localCellBegin(Dim::J) );
    EXPECT_EQ( local_cell_begin[2], block_2.localCellBegin(Dim::K) );

    EXPECT_EQ( local_cell_end[0], block_2.localCellEnd(Dim::I) );
    EXPECT_EQ( local_cell_end[1], block_2.localCellEnd(Dim::J) );
    EXPECT_EQ( local_cell_end[2], block_2.localCellEnd(Dim::K) );

    EXPECT_EQ( local_node_begin[0], block_2.localNodeBegin(Dim::I) );
    EXPECT_EQ( local_node_begin[1], block_2.localNodeBegin(Dim::J) );
    EXPECT_EQ( local_node_begin[2], block_2.localNodeBegin(Dim::K) );

    EXPECT_EQ( local_node_end[0], block_2.localNodeEnd(Dim::I) );
    EXPECT_EQ( local_node_end[1], block_2.localNodeEnd(Dim::J) );
    EXPECT_EQ( local_node_end[2], block_2.localNodeEnd(Dim::K) );
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

} // end namespace Test
