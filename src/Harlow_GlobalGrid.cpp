#include <Harlow_GlobalGrid.hpp>
#include <Harlow_Types.hpp>

#include <algorithm>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Constructor.
GlobalGrid::GlobalGrid( MPI_Comm comm,
                        const std::vector<int>& ranks_per_dim,
                        const std::vector<bool>& is_dim_periodic,
                        const std::vector<double>& global_low_corner,
                        const std::vector<double>& global_high_corner,
                        const double cell_size )
    : _global_low_corner( global_low_corner )
{
    // Compute how many cells are in each dimension.
    _global_num_cell.resize( 3 );
    for ( int d = 0; d < 3; ++d )
        _global_num_cell[d] =
            (global_high_corner[d] - _global_low_corner[d]) / cell_size;

    // Check the cell size.
    for ( int d = 0; d < 3; ++d )
        if ( _global_num_cell[d] * cell_size + _global_low_corner[d] !=
             global_high_corner[d] )
            throw std::invalid_argument("Dimension not divisible by cell size");

    // Extract the periodicity of the boundary as integers.
    std::vector<int> periodic_dims =
        {is_dim_periodic[0],is_dim_periodic[1],is_dim_periodic[2]};

    // Generate a communicator with a Cartesian topology.
    int reorder_cart_ranks = 1;
    MPI_Cart_create( comm,
                     3,
                     ranks_per_dim.data(),
                     periodic_dims.data(),
                     reorder_cart_ranks,
                     &_cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    MPI_Comm_rank( _cart_comm, &linear_rank );
    std::vector<int> cart_rank( 3 );
    MPI_Cart_coords( _cart_comm, linear_rank, 3, cart_rank.data() );

    // Get the cells per dimension and the remainder.
    std::vector<int> cells_per_dim( 3 );
    std::vector<int> dim_remainder( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        cells_per_dim[d] = _global_num_cell[d] / ranks_per_dim[d];
        dim_remainder[d] = _global_num_cell[d] % ranks_per_dim[d];
    }

    // Compute the local low corner on this rank by computing the starting
    // global cell index via exclusive scan.
    std::vector<double> local_low_corner( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        int dim_offset = 0;
        for ( int r = 0; r < cart_rank[d]; ++r )
        {
            dim_offset += cells_per_dim[d];
            if ( dim_remainder[d] > r )
                ++dim_offset;
        }
        local_low_corner[d] = dim_offset * cell_size + _global_low_corner[d];
    }

    // Compute the number of local cells in this rank in each dimension.
    std::vector<int> local_num_cell( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        local_num_cell[d] = cells_per_dim[d];
        if ( dim_remainder[d] > cart_rank[d] )
            ++local_num_cell[d];
    }

    // Determine if this rank is on a physical boundary.
    std::vector<bool> boundary_location( 6, false );
    for ( int d = 0; d < 3; ++d )
    {
        if ( 0 == cart_rank[d] )
            boundary_location[2*d] = true;
        if ( ranks_per_dim[d] - 1 == cart_rank[d] )
            boundary_location[2*d+1] = true;
    }

    // Create the local grid block.
    _grid_block = GridBlock( local_low_corner, local_num_cell,
                             boundary_location, is_dim_periodic, cell_size, 0 );

    // Create the graph communicator. Start by getting the neighbors we will
    // communicatioe with. Note the ordering of the ijk loop here - I moves
    // the fastest and K moves the slowest. Note that we do not send to
    // ourselves (when each index is 0).
    std::vector<int> neighbors;
    neighbors.reserve( 26 );
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 && k==0) )
                {
                    // Set the Cartesian rank of this neighbor.
                    std::vector<int> ncr = { cart_rank[Dim::I] + i,
                                             cart_rank[Dim::J] + j,
                                             cart_rank[Dim::K] + k };

                    // Check for being outside of a non-periodic dimension.
                    bool is_null = false;
                    for ( int d = 0; d < 3; ++d )
                        if ( ncr[d] < 0 || ncr[d] >= ranks_per_dim[d] )
                            if ( !_grid_block.isPeriodic(d) )
                                is_null = true;

                    // If we were outside of a non-periodic dimension this
                    // rank is null.
                    if ( is_null )
                    {
                        neighbors.push_back( MPI_PROC_NULL );
                    }

                    // Otherwise get our real neighbor. The periodic case
                    // should wrap around when out of bounds.
                    else
                    {
                        int nr;
                        MPI_Cart_rank( _cart_comm, ncr.data(), &nr );
                        neighbors.push_back( nr );
                    }
                }

    // Build the new topology for the graph communicator.
    int reorder_graph_ranks = 1;
    MPI_Dist_graph_create_adjacent( _cart_comm,
                                    26, neighbors.data(), MPI_UNWEIGHTED,
                                    26, neighbors.data(), MPI_UNWEIGHTED,
                                    MPI_INFO_NULL,
                                    reorder_graph_ranks, &_graph_comm );
}

//---------------------------------------------------------------------------//
// Get the grid communicator.
MPI_Comm GlobalGrid::comm() const
{ return _cart_comm; }

//---------------------------------------------------------------------------//
// Get the grid communicator with a Cartesian topology.
MPI_Comm GlobalGrid::cartesianComm() const
{ return _cart_comm; }

//---------------------------------------------------------------------------//
// Get the grid communicator with a Graph topology.
MPI_Comm GlobalGrid::graphComm() const
{ return _graph_comm; }

//---------------------------------------------------------------------------//
// Get a grid block on this rank with a given halo cell width.
const GridBlock& GlobalGrid::block() const
{ return _grid_block; }

//---------------------------------------------------------------------------//
// Get whether a given logical dimension is periodic.
bool GlobalGrid::isPeriodic( const int dim ) const
{ return _grid_block.isPeriodic(dim); }

//---------------------------------------------------------------------------//
// Get the global number of cells in a given dimension.
int GlobalGrid::numCell( const int dim ) const
{ return _global_num_cell[dim]; }

//---------------------------------------------------------------------------//
// Get the global number of nodes in a given dimension.
int GlobalGrid::numNode( const int dim ) const
{ return _global_num_cell[dim] + 1; }

//---------------------------------------------------------------------------//
// Get the global low corner.
double GlobalGrid::lowCorner( const int dim ) const
{ return _global_low_corner[dim]; }

//---------------------------------------------------------------------------//
// Get the cell size.
double GlobalGrid::cellSize() const
{ return _grid_block.cellSize(); }

//---------------------------------------------------------------------------//

} // end namespace Harlow
