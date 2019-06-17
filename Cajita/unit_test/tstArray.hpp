#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_Array.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <numeric>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void layoutTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<bool> is_dim_periodic = {true,true,true};
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         partitioner,
                                         is_dim_periodic,
                                         global_low_corner,
                                         global_high_corner,
                                         cell_size );

    // Create an array layout on the nodes.
    int halo_width = 2;
    int dofs_per_node = 4;
    auto node_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_node, Node() );

    // Check the owned index_space.
    auto array_node_owned_space =
        node_layout->ownedIndexSpace();
    auto block_node_owned_space =
        node_layout->block().ownedIndexSpace( Node() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_owned_space.min(d),
                   block_node_owned_space.min(d) );
        EXPECT_EQ( array_node_owned_space.max(d),
                   block_node_owned_space.max(d) );
    }
    EXPECT_EQ( array_node_owned_space.min(3), 0 );
    EXPECT_EQ( array_node_owned_space.max(3), dofs_per_node );

    // Check the ghosted index_space.
    auto array_node_ghosted_space =
        node_layout->ghostedIndexSpace();
    auto block_node_ghosted_space =
        node_layout->block().ghostedIndexSpace( Node() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_ghosted_space.min(d),
                   block_node_ghosted_space.min(d) );
        EXPECT_EQ( array_node_ghosted_space.max(d),
                   block_node_ghosted_space.max(d) );
    }
    EXPECT_EQ( array_node_ghosted_space.min(3), 0 );
    EXPECT_EQ( array_node_ghosted_space.max(3), dofs_per_node );

    // Check the shared owned index_space.
    auto array_node_shared_owned_space =
        node_layout->sharedOwnedIndexSpace(-1,0,1);
    auto block_node_shared_owned_space =
        node_layout->block().sharedOwnedIndexSpace( Node(), -1, 0, 1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_shared_owned_space.min(d),
                   block_node_shared_owned_space.min(d) );
        EXPECT_EQ( array_node_shared_owned_space.max(d),
                   block_node_shared_owned_space.max(d) );
    }
    EXPECT_EQ( array_node_shared_owned_space.min(3), 0 );
    EXPECT_EQ( array_node_shared_owned_space.max(3), dofs_per_node );

    // Check the shared ghosted index_space.
    auto array_node_shared_ghosted_space =
        node_layout->sharedGhostedIndexSpace(1,-1,0);
    auto block_node_shared_ghosted_space =
        node_layout->block().sharedGhostedIndexSpace( Node(), 1, -1, 0 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_shared_ghosted_space.min(d),
                   block_node_shared_ghosted_space.min(d) );
        EXPECT_EQ( array_node_shared_ghosted_space.max(d),
                   block_node_shared_ghosted_space.max(d) );
    }
    EXPECT_EQ( array_node_shared_ghosted_space.min(3), 0 );
    EXPECT_EQ( array_node_shared_ghosted_space.max(3), dofs_per_node );

    // Create an array layout on the cells.
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );

    // Check the owned index_space.
    auto array_cell_owned_space =
        cell_layout->ownedIndexSpace();
    auto block_cell_owned_space =
        cell_layout->block().ownedIndexSpace( Cell() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_owned_space.min(d),
                   block_cell_owned_space.min(d) );
        EXPECT_EQ( array_cell_owned_space.max(d),
                   block_cell_owned_space.max(d) );
    }
    EXPECT_EQ( array_cell_owned_space.min(3), 0 );
    EXPECT_EQ( array_cell_owned_space.max(3), dofs_per_cell );

    // Check the ghosted index_space.
    auto array_cell_ghosted_space =
        cell_layout->ghostedIndexSpace();
    auto block_cell_ghosted_space =
        cell_layout->block().ghostedIndexSpace( Cell() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_ghosted_space.min(d),
                   block_cell_ghosted_space.min(d) );
        EXPECT_EQ( array_cell_ghosted_space.max(d),
                   block_cell_ghosted_space.max(d) );
    }
    EXPECT_EQ( array_cell_ghosted_space.min(3), 0 );
    EXPECT_EQ( array_cell_ghosted_space.max(3), dofs_per_cell );

    // Check the shared owned index_space.
    auto array_cell_shared_owned_space =
        cell_layout->sharedOwnedIndexSpace(0,1,-1);
    auto block_cell_shared_owned_space =
        cell_layout->block().sharedOwnedIndexSpace( Cell(), 0, 1, -1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_shared_owned_space.min(d),
                   block_cell_shared_owned_space.min(d) );
        EXPECT_EQ( array_cell_shared_owned_space.max(d),
                   block_cell_shared_owned_space.max(d) );
    }
    EXPECT_EQ( array_cell_shared_owned_space.min(3), 0 );
    EXPECT_EQ( array_cell_shared_owned_space.max(3), dofs_per_cell );

    // Check the shared ghosted index_space.
    auto array_cell_shared_ghosted_space =
        cell_layout->sharedGhostedIndexSpace(1,1,1);
    auto block_cell_shared_ghosted_space =
        cell_layout->block().sharedGhostedIndexSpace( Cell(), 1, 1, 1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_shared_ghosted_space.min(d),
                   block_cell_shared_ghosted_space.min(d) );
        EXPECT_EQ( array_cell_shared_ghosted_space.max(d),
                   block_cell_shared_ghosted_space.max(d) );
    }
    EXPECT_EQ( array_cell_shared_ghosted_space.min(3), 0 );
    EXPECT_EQ( array_cell_shared_ghosted_space.max(3), dofs_per_cell );
}

//---------------------------------------------------------------------------//
void arrayTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<bool> is_dim_periodic = {true,true,true};
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         partitioner,
                                         is_dim_periodic,
                                         global_low_corner,
                                         global_high_corner,
                                         cell_size );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array = createArray<double,TEST_DEVICE>( label, cell_layout );

    // Check the array.
    EXPECT_EQ( label, array->label() );
    auto space = cell_layout->ghostedIndexSpace();
    auto view = array->view();
    EXPECT_EQ( label, view.label() );
    EXPECT_EQ( view.size(), space.size() );
    for ( int i = 0; i < 4; ++i )
        EXPECT_EQ( view.extent(i), space.extent(i) );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( array, array_test )
{
    layoutTest();
    arrayTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
