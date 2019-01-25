#include <Harlow_Types.hpp>
#include <Harlow_GridBlock.hpp>
#include <Harlow_GridField.hpp>

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
    std::vector<bool> periodic = {false,false,false};
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
                    periodic, cell_size, halo_width );

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
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, field_test )
{
    fieldTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
