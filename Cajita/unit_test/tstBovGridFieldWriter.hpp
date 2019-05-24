#include <Cajita_BovGridFieldWriter.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GridField.hpp>
#include <Cajita_GridExecPolicy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <memory>
#include <fstream>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void writeTest( const std::vector<int>& ranks_per_dim )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 22, 19, 21 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    std::vector<bool> is_dim_periodic = {false,false,false};
    auto global_grid = std::make_shared<GlobalGrid>(
        MPI_COMM_WORLD,
        ranks_per_dim,
        is_dim_periodic,
        global_low_corner,
        global_high_corner,
        cell_size );

    // Get the global ijk offsets.
    auto off_i = global_grid->globalOffset( Dim::I );
    auto off_j = global_grid->globalOffset( Dim::J );
    auto off_k = global_grid->globalOffset( Dim::K );

    // Create a cell field and fill it with data.
    GridField<double,TEST_MEMSPACE> cell_field(
        global_grid, 1, MeshEntity::Cell, 0, "cell_field" );
    auto cell_data = cell_field.data();
    Kokkos::parallel_for(
        "fill_cell_field",
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>(
            global_grid->block(),MeshEntity::Cell),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            cell_data( i, j, k, 0 ) = i + off_i + j + off_j + k + off_k;
        } );

    // Create a node field and fill it with data.
    GridField<int,TEST_MEMSPACE> node_field(
        global_grid, 3, MeshEntity::Node, 0, "node_field" );
    auto node_data = node_field.data();
    Kokkos::parallel_for(
        "fill_node_field",
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>(
            global_grid->block(),MeshEntity::Node),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            node_data( i, j, k, Dim::I ) = i + off_i;
            node_data( i, j, k, Dim::J ) = j + off_j;
            node_data( i, j, k, Dim::K ) = k + off_k;
        } );

    // Write the fields to a file.
    BovGridFieldWriter::writeTimeStep( 302, 3.43, cell_field );
    BovGridFieldWriter::writeTimeStep( 1972, 12.457, node_field );

    // Read the data back in on rank 0 and make sure it is OK.
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( 0 == rank )
    {
        // Open the cell file.
        std::fstream cell_data_file;
        cell_data_file.open( "grid_cell_field_000302.dat",
                             std::fstream::in | std::fstream::binary );

        // The cell file data is ordered KJI
        double cell_value;
        int cell_id = 0;
        for ( int k = 0; k < global_num_cell[Dim::K]; ++k )
            for ( int j = 0; j < global_num_cell[Dim::J]; ++j )
                for ( int i = 0; i < global_num_cell[Dim::I]; ++i )
                {
                    cell_data_file.seekg( cell_id * sizeof(double) );
                    cell_data_file.read( (char*) &cell_value, sizeof(double) );
                    EXPECT_EQ( cell_value, i + j + k );
                    ++cell_id;
                }

        // Close the cell file.
        cell_data_file.close();

        // Open the node file.
        std::fstream node_data_file;
        node_data_file.open( "grid_node_field_001972.dat",
                             std::fstream::in | std::fstream::binary );

        // The node file data is ordered KJI
        int node_value;
        int node_id = 0;
        for ( int k = 0; k < global_num_cell[Dim::K] + 1; ++k )
            for ( int j = 0; j < global_num_cell[Dim::J] + 1; ++j )
                for ( int i = 0; i < global_num_cell[Dim::I] + 1; ++i )
                {
                    node_data_file.seekg( node_id * sizeof(int) );
                    node_data_file.read( (char*) &node_value, sizeof(int) );
                    EXPECT_EQ( node_value, i );
                    ++node_id;

                    node_data_file.seekg( node_id * sizeof(int) );
                    node_data_file.read( (char*) &node_value, sizeof(int) );
                    EXPECT_EQ( node_value, j );
                    ++node_id;

                    node_data_file.seekg( node_id * sizeof(int) );
                    node_data_file.read( (char*) &node_value, sizeof(int) );
                    EXPECT_EQ( node_value, k );
                    ++node_id;
                }

        // Close the node file.
        node_data_file.close();
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, write_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    writeTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    writeTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    writeTest( ranks_per_dim );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    writeTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    writeTest( ranks_per_dim );
}

//---------------------------------------------------------------------------//

} // end namespace Test
