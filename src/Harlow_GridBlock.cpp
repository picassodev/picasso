#include <Harlow_GridBlock.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Default constructor.
GridBlock::GridBlock()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Constructor.
GridBlock::GridBlock(
    const std::vector<double>& local_low_corner,
    const std::vector<int>& local_num_cell,
    const std::vector<bool>& boundary_location,
    const std::vector<bool>& is_dim_periodic,
    const double cell_size,
    const int halo_cell_width )
    : _local_low_corner( local_low_corner )
    , _local_num_cell( local_num_cell )
    , _total_num_cell( local_num_cell )
    , _boundary_location( boundary_location )
    , _periodic( is_dim_periodic )
    , _cell_size( cell_size )
    , _halo_cell_width( halo_cell_width )
{
    setHalo();
}

//---------------------------------------------------------------------------//
// Assign the local state of a cartesian grid block with a new halo size.
void GridBlock::assign(
    const GridBlock& rhs, const int halo_cell_width )
{
    _local_low_corner = rhs._local_low_corner;
    _local_num_cell = rhs._local_num_cell;
    _total_num_cell = rhs._local_num_cell;
    _boundary_location = rhs._boundary_location;
    _periodic = rhs._periodic;
    _cell_size = rhs._cell_size;

    _halo_cell_width = halo_cell_width;

    setHalo();
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the low corner of the grid in a given
// dimension. This low corner includes the halo region.
double GridBlock::lowCorner( const int dim ) const
{ return _low_corner[dim]; }

//---------------------------------------------------------------------------//
// Given a physical boundary id return if this grid is on that boundary.
bool GridBlock::onBoundary( const int boundary_id ) const
{ return _boundary_location[boundary_id]; }

//---------------------------------------------------------------------------//
// Get whether a given logical dimension is periodic.
bool GridBlock::isPeriodic( const int dim ) const
{ return _periodic[dim]; }

//---------------------------------------------------------------------------//
// Get the cell size.
double GridBlock::cellSize() const
{ return _cell_size; }

//---------------------------------------------------------------------------//
// Get the inverse cell size.
double GridBlock::inverseCellSize() const
{ return 1.0 / _cell_size; }

//---------------------------------------------------------------------------//
// Get the halo size.
int GridBlock::haloSize() const
{ return _halo_cell_width; }

//---------------------------------------------------------------------------//
// Get the total number of cells in a given dimension including the halo.
int GridBlock::numCell( const int dim ) const
{ return _total_num_cell[dim]; }

//---------------------------------------------------------------------------//
// Get the total number of nodes in a given dimension including the halo.
int GridBlock::numNode( const int dim ) const
{ return _total_num_cell[dim] + 1; }

//---------------------------------------------------------------------------//
// Get the beginning local cell index in a given direction. The local cells do
// not include the halo. Logical boundaries that are also physical boundaries
// do not have a halo region unless the physical boundary is periodic.
int GridBlock::localCellBegin( const int dim ) const
{
    return ( _boundary_location[2*dim] && !_periodic[dim] )
        ? 0 : _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the ending local cell index in a given direction. The local cells do
// not include the halo. Logical boundaries that are also on physical
// boundaries do not have a halo region unless the physical boundary is
// periodic.
int GridBlock::localCellEnd( const int dim ) const
{
    return ( _boundary_location[2*dim+1] && !_periodic[dim] ) ?
        _total_num_cell[dim] : _total_num_cell[dim] - _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the beginning local node index in a given direction. The local
// nodes do not include the halo. A local grid block always "owns" the
// node on the low logical boundary.
int GridBlock::localNodeBegin( const int dim ) const
{ return localCellBegin(dim); }

//---------------------------------------------------------------------------//
// Get the ending local node index in a given direction. The local nodes
// do not include the halo. The block neighbor "owns" the node on the high
// logical boundary (unless the high logical boundary is also a physical
// boundary that is not periodic).
int GridBlock::localNodeEnd( const int dim ) const
{
    return ( _boundary_location[2*dim+1] && !_periodic[dim] )
        ? localCellEnd(dim) + 1 : localCellEnd(dim);
}

//---------------------------------------------------------------------------//
// Get the local number of cells in a given dimension.
int GridBlock::localNumCell( const int dim ) const
{
    return _local_num_cell[dim];
}

//---------------------------------------------------------------------------//
// Get the local number of nodes in a given dimension.
int GridBlock::localNumNode( const int dim ) const
{
    return localNodeEnd(dim) - localNodeBegin(dim);
}

//---------------------------------------------------------------------------//
// Set the halo.
void GridBlock::setHalo()
{
    // Calculate the low corner of the local block including the halo.
    _low_corner.resize( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        if ( _boundary_location[2*d] && !_periodic[d])
            _low_corner[d] = _local_low_corner[d];
        else
            _low_corner[d] =
                _local_low_corner[d] - _halo_cell_width * _cell_size;
    }

    // Add halo cells to the total counts if not on a physical boundary or if
    // we are on a periodic boundary.
    for ( int d = 0; d < 3; ++d )
    {
        if ( !_boundary_location[2*d] || _periodic[d] )
            _total_num_cell[d] += _halo_cell_width;
        if ( !_boundary_location[2*d+1] || _periodic[d] )
            _total_num_cell[d] += _halo_cell_width;
    }
}

//---------------------------------------------------------------------------//

} // end namespace Harlow
