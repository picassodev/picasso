#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <numeric>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
// Fixture
class harlow_global_grid : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};

//---------------------------------------------------------------------------//
void gridTest()
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<bool> is_dim_periodic = {false,false,false};
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    GlobalGrid global_grid( MPI_COMM_WORLD,
                            ranks_per_dim,
                            is_dim_periodic,
                            global_low_corner,
                            global_high_corner,
                            cell_size );

    // Check the dimension data.
    EXPECT_EQ( global_grid.cellSize(), cell_size );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( global_num_cell[d], global_grid.numEntity(MeshEntity::Cell,d) );
        EXPECT_EQ( global_num_cell[d] + 1, global_grid.numEntity(MeshEntity::Node,d) );
        EXPECT_EQ( global_low_corner[d], global_grid.lowCorner(d) );
        EXPECT_FALSE( global_grid.isPeriodic(d) );
    }

    // Check the communicator. The grid communicator has a Cartesian topology.
    auto grid_comm = global_grid.comm();
    int grid_comm_size;
    MPI_Comm_size( grid_comm, &grid_comm_size );
    EXPECT_EQ( grid_comm_size, comm_size );

    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_period( 3 );
    std::vector<int> cart_rank( 3 );
    MPI_Cart_get(
        grid_comm, 3, cart_dims.data(), cart_period.data(), cart_rank.data() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( cart_dims[d], ranks_per_dim[d] );
        EXPECT_EQ( cart_period[d], 0 );
    }

    // Get the grid block.
    int halo_width = 2;
    GridBlock grid_block;
    grid_block.assign( global_grid.block(), halo_width );

    // Check sizes
    EXPECT_EQ( grid_block.haloSize(), halo_width );
    EXPECT_EQ( grid_block.cellSize(), cell_size );
    EXPECT_EQ( grid_block.inverseCellSize(), 1.0 / cell_size );

    // Get the local number of cells.
    std::vector<int> local_num_cells =
        { grid_block.localEntityEnd(MeshEntity::Cell,Dim::I) -
          grid_block.localEntityBegin(MeshEntity::Cell,Dim::I),
          grid_block.localEntityEnd(MeshEntity::Cell,Dim::J) -
          grid_block.localEntityBegin(MeshEntity::Cell,Dim::J),
          grid_block.localEntityEnd(MeshEntity::Cell,Dim::K) -
          grid_block.localEntityBegin(MeshEntity::Cell,Dim::K) };

    // Compute a global set of local cell size arrays.
    std::vector<int> local_num_cell_i( ranks_per_dim[Dim::I], 0 );
    std::vector<int> local_num_cell_j( ranks_per_dim[Dim::J], 0 );
    std::vector<int> local_num_cell_k( ranks_per_dim[Dim::K], 0 );
    local_num_cell_i[ cart_rank[Dim::I] ] = local_num_cells[Dim::I];
    local_num_cell_j[ cart_rank[Dim::J] ] = local_num_cells[Dim::J];
    local_num_cell_k[ cart_rank[Dim::K] ] = local_num_cells[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_i.data(), ranks_per_dim[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_j.data(), ranks_per_dim[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_k.data(), ranks_per_dim[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total cells in each
    // dimension.
    EXPECT_EQ( global_num_cell[0],
               std::accumulate(
                   local_num_cell_i.begin(), local_num_cell_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[1],
               std::accumulate(
                   local_num_cell_j.begin(), local_num_cell_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[2],
               std::accumulate(
                   local_num_cell_k.begin(), local_num_cell_k.end(), 0 ) );

    // Get the local number of nodes.
    std::vector<int> local_num_nodes =
        { grid_block.localEntityEnd(MeshEntity::Node,Dim::I) -
          grid_block.localEntityBegin(MeshEntity::Node,Dim::I),
          grid_block.localEntityEnd(MeshEntity::Node,Dim::J) -
          grid_block.localEntityBegin(MeshEntity::Node,Dim::J),
          grid_block.localEntityEnd(MeshEntity::Node,Dim::K) -
          grid_block.localEntityBegin(MeshEntity::Node,Dim::K) };

    // Compute a global set of local node size arrays.
    std::vector<int> local_num_node_i( ranks_per_dim[Dim::I], 0 );
    std::vector<int> local_num_node_j( ranks_per_dim[Dim::J], 0 );
    std::vector<int> local_num_node_k( ranks_per_dim[Dim::K], 0 );
    local_num_node_i[ cart_rank[Dim::I] ] = local_num_nodes[Dim::I];
    local_num_node_j[ cart_rank[Dim::J] ] = local_num_nodes[Dim::J];
    local_num_node_k[ cart_rank[Dim::K] ] = local_num_nodes[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_i.data(), ranks_per_dim[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_j.data(), ranks_per_dim[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_k.data(), ranks_per_dim[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check boundary status.
    if ( cart_rank[Dim::I] == 0 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::LowX) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::LowX) );

    if ( cart_rank[Dim::I] == ranks_per_dim[Dim::I] - 1 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::HighX) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::HighX) );

    if ( cart_rank[Dim::J] == 0 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::LowY) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::LowY) );

    if ( cart_rank[Dim::J] == ranks_per_dim[Dim::J] - 1 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::HighY) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::HighY) );

    if ( cart_rank[Dim::K] == 0 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::LowZ) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::LowZ) );

    if ( cart_rank[Dim::K] == ranks_per_dim[Dim::K] - 1 )
        EXPECT_TRUE( grid_block.onBoundary(DomainBoundary::HighZ) );
    else
        EXPECT_FALSE( grid_block.onBoundary(DomainBoundary::HighZ) );

    // Check the local cell bounds.
    EXPECT_EQ( grid_block.localEntityBegin(MeshEntity::Cell,Dim::I), halo_width );
    EXPECT_EQ( grid_block.localEntityEnd(MeshEntity::Cell,Dim::I),
               local_num_cells[Dim::I] + halo_width );
    EXPECT_EQ( grid_block.localEntityBegin(MeshEntity::Cell,Dim::J), halo_width );
    EXPECT_EQ( grid_block.localEntityEnd(MeshEntity::Cell,Dim::J),
               local_num_cells[Dim::J] + halo_width );
    EXPECT_EQ( grid_block.localEntityBegin(MeshEntity::Cell,Dim::K), halo_width );
    EXPECT_EQ( grid_block.localEntityEnd(MeshEntity::Cell,Dim::K),
               local_num_cells[Dim::K] + halo_width );

    // Get another block without a halo and check the local low corner. Do an
    // exclusive scan of sizes to get the local cell offset.
    GridBlock grid_block_2;
    grid_block_2.assign( grid_block, 0 );
    int i_offset =
        std::accumulate( local_num_cell_i.begin(),
                         local_num_cell_i.begin() + cart_rank[Dim::I],
                         0 );
    int j_offset =
        std::accumulate( local_num_cell_j.begin(),
                         local_num_cell_j.begin() + cart_rank[Dim::J],
                         0 );
    int k_offset =
        std::accumulate( local_num_cell_k.begin(),
                         local_num_cell_k.begin() + cart_rank[Dim::K],
                         0 );

    EXPECT_EQ( grid_block_2.lowCorner(Dim::I),
               i_offset * cell_size + global_low_corner[Dim::I] );
    EXPECT_EQ( grid_block_2.lowCorner(Dim::J),
               j_offset * cell_size + global_low_corner[Dim::J] );
    EXPECT_EQ( grid_block_2.lowCorner(Dim::K),
               k_offset * cell_size + global_low_corner[Dim::K] );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( harlow_global_grid, grid_test )
{
    gridTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
