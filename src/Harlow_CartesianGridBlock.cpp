#include <Harlow_CartesianGridBlock.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Default constructor.
CartesianGridBlock::CartesianGridBlock()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Constructor.
CartesianGridBlock::CartesianGridBlock(
    const std::vector<double>& local_low_corner,
    const std::vector<int>& local_num_cell,
    const std::vector<bool>& boundary_location,
    const double cell_size,
    const int halo_cell_width )
    : _local_low_corner( local_low_corner )
    , _local_num_cell( local_num_cell )
    , _total_num_cell( local_num_cell )
    , _boundary_location( boundary_location )
    , _cell_size( cell_size )
    , _halo_cell_width( halo_cell_width )
{
    setHalo();
}

//---------------------------------------------------------------------------//
// Assign the local state of a cartesian grid block with a new halo size.
void CartesianGridBlock::assign(
    const CartesianGridBlock& rhs, const int halo_cell_width )
{
    _local_low_corner = rhs._local_low_corner;
    _local_num_cell = rhs._local_num_cell;
    _total_num_cell = rhs._local_num_cell;
    _boundary_location = rhs._boundary_location;
    _cell_size = rhs._cell_size;

    _halo_cell_width = halo_cell_width;

    setHalo();
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the low corner of the grid in a given
// dimension. This low corner includes the halo region.
double CartesianGridBlock::lowCorner( const int dim ) const
{ return _low_corner[dim]; }

//---------------------------------------------------------------------------//
// Given a physical boundary id return if this grid is on that boundary.
bool CartesianGridBlock::onBoundary( const int boundary_id ) const
{ return _boundary_location[boundary_id]; }

//---------------------------------------------------------------------------//
// Get the cell size.
double CartesianGridBlock::cellSize() const
{ return _cell_size; }

//---------------------------------------------------------------------------//
// Get the inverse cell size.
double CartesianGridBlock::inverseCellSize() const
{ return 1.0 / _cell_size; }

//---------------------------------------------------------------------------//
// Get the halo size.
int CartesianGridBlock::haloSize() const
{ return _halo_cell_width; }

//---------------------------------------------------------------------------//
// Get the total number of cells in a given dimension including the halo.
int CartesianGridBlock::numCell( const int dim ) const
{ return _total_num_cell[dim]; }

//---------------------------------------------------------------------------//
// Get the total number of nodes in a given dimension including the halo.
int CartesianGridBlock::numNode( const int dim ) const
{ return _total_num_cell[dim] + 1; }

//---------------------------------------------------------------------------//
// Get the beginning local cell index in a given direction. The local
// cells do not include the halo. Logical boundaries that are also
// physical boundaries do not have a halo region.
int CartesianGridBlock::localCellBegin( const int dim ) const
{ return ( _boundary_location[2*dim] ) ? 0 : _halo_cell_width; }

//---------------------------------------------------------------------------//
// Get the ending local cell index in a given direction. The local cells
// do not include the halo. Logical boundaries that are also on physical
// boundaries do not have a halo region.
int CartesianGridBlock::localCellEnd( const int dim ) const
{
    return ( _boundary_location[2*dim+1] ) ?
        _total_num_cell[dim] : _total_num_cell[dim] - _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the beginning local node index in a given direction. The local
// nodes do not include the halo. A local grid block always "owns" the
// node on the low logical boundary.
int CartesianGridBlock::localNodeBegin( const int dim ) const
{ return localCellBegin(dim); }

//---------------------------------------------------------------------------//
// Get the ending local node index in a given direction. The local nodes
// do not include the halo. The block neighbor "owns" the node on the high
// logical boundary (unless the high logical boundary is also a physical
// boundary).
int CartesianGridBlock::localNodeEnd( const int dim ) const
{
    return ( _boundary_location[2*dim+1] )
        ? localCellEnd(dim) + 1 : localCellEnd(dim);
}

//---------------------------------------------------------------------------//
// Set the halo.
void CartesianGridBlock::setHalo()
{
    // Calculate the low corner of the local block including the halo.
    _low_corner.resize( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        if ( _boundary_location[2*d] )
            _low_corner[d] = _local_low_corner[d];
        else
            _low_corner[d] =
                _local_low_corner[d] - _halo_cell_width * _cell_size;
    }

    // Add halo cells to the total counts if not on a physical boundary.
    for ( int d = 0; d < 3; ++d )
    {
        if ( !_boundary_location[2*d] )
            _total_num_cell[d] += _halo_cell_width;
        if ( !_boundary_location[2*d+1] )
            _total_num_cell[d] += _halo_cell_width;
    }
}

//---------------------------------------------------------------------------//

} // end namespace Harlow
