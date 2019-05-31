#include <Cajita_Types.hpp>
#include <Cajita_GridBlock.hpp>
#include <Cajita_Field.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void fieldTest()
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
    std::vector<int> local_cell_end = { num_cell[0] - halo_width,
                                        num_cell[1] - halo_width,
                                        num_cell[2] - halo_width };
    std::vector<int> local_node_begin = local_cell_begin;
    std::vector<int> local_node_end = { local_cell_end[0] + 1,
                                        local_cell_end[1] + 1,
                                        local_cell_end[2] };
    GridBlock grid( low_corner, input_num_cell, boundary_location,
                    periodic, cell_size, halo_width );

    // Make some cell fields.
    auto scalar_cell_field =
        createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Cell );
    EXPECT_EQ( scalar_cell_field.Rank, 4 );
    EXPECT_EQ( scalar_cell_field.extent(0), num_cell[0] );
    EXPECT_EQ( scalar_cell_field.extent(1), num_cell[1] );
    EXPECT_EQ( scalar_cell_field.extent(2), num_cell[2] );
    EXPECT_EQ( scalar_cell_field.extent(3), 1 );
    bool correct_value_type =
        std::is_same<double,
                     typename decltype(scalar_cell_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto vector_cell_field =
        createField<double,TEST_MEMSPACE>( grid, 5, MeshEntity::Cell );
    EXPECT_EQ( vector_cell_field.Rank, 4 );
    EXPECT_EQ( vector_cell_field.extent(0), num_cell[0] );
    EXPECT_EQ( vector_cell_field.extent(1), num_cell[1] );
    EXPECT_EQ( vector_cell_field.extent(2), num_cell[2] );
    EXPECT_EQ( vector_cell_field.extent(3), 5 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(vector_cell_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    // Make some node fields.
    auto scalar_node_field =
        createField<double,TEST_MEMSPACE>( grid, 1, MeshEntity::Node );
    EXPECT_EQ( scalar_node_field.Rank, 4 );
    EXPECT_EQ( scalar_node_field.extent(0), num_node[0] );
    EXPECT_EQ( scalar_node_field.extent(1), num_node[1] );
    EXPECT_EQ( scalar_node_field.extent(2), num_node[2] );
    EXPECT_EQ( scalar_node_field.extent(3), 1 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(scalar_node_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );

    auto vector_node_field =
        createField<double,TEST_MEMSPACE>( grid, 5, MeshEntity::Node );
    EXPECT_EQ( vector_node_field.Rank, 4 );
    EXPECT_EQ( vector_node_field.extent(0), num_node[0] );
    EXPECT_EQ( vector_node_field.extent(1), num_node[1] );
    EXPECT_EQ( vector_node_field.extent(2), num_node[2] );
    EXPECT_EQ( vector_node_field.extent(3), 5 );
    correct_value_type =
        std::is_same<double,
                     typename decltype(vector_node_field)::value_type>::value;
    EXPECT_TRUE( correct_value_type );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, field_test )
{
    fieldTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
