#include <Harlow_GridBlock.hpp>
#include <Harlow_Types.hpp>

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
// Given a boundary id return if this has a halo on that boundary.
bool GridBlock::hasHalo( const int boundary_id ) const
{ return ( !onBoundary(boundary_id) || isPeriodic(boundary_id/2) ); }

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
// Get the total number of mesh entities in a given dimension including
// the halo.
int GridBlock::numEntity( const int entity_type, const int dim ) const
{
    if ( MeshEntity::Cell == entity_type )
        return _total_num_cell[dim];

    else if ( MeshEntity::Node == entity_type )
        return _total_num_cell[dim] + 1;

    else
        throw std::invalid_argument("Bad mesh entity type");
}

//---------------------------------------------------------------------------//
// Get the beginning local entity index in a given direction. The local
// entities do not include the halo.
int GridBlock::localEntityBegin( const int entity_type, const int dim ) const
{
    std::ignore = entity_type;
    std::ignore = dim;
    return _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the end local entity index in a given direction. The local
// entities do not include the halo.
//
// Node case; The local nodes do not include the halo. The local grid
// block does not "own" the node on the high logical boundary unless the
// high logical boundary is also a physical boundary that is not periodic.
int GridBlock::localEntityEnd( const int entity_type, const int dim ) const
{
    if ( MeshEntity::Cell == entity_type )
        return _total_num_cell[dim] - _halo_cell_width;

    else if ( MeshEntity::Node == entity_type )
        return ( _boundary_location[2*dim+1] && !_periodic[dim] )
            ? _total_num_cell[dim] - _halo_cell_width + 1 :
            _total_num_cell[dim] - _halo_cell_width;

    else
        throw std::invalid_argument("Bad mesh entity type");
}

//---------------------------------------------------------------------------//
// Get the local number of entities in a given dimension.
int GridBlock::localNumEntity( const int entity_type, const int dim ) const
{
    return localEntityEnd( entity_type, dim ) -
        localEntityBegin( entity_type, dim );
}

//---------------------------------------------------------------------------//
// Set the halo.
void GridBlock::setHalo()
{
    // Calculate the low corner of the local block including the halo.
    _low_corner.resize( 3 );
    for ( int d = 0; d < 3; ++d )
        _low_corner[d] =
            _local_low_corner[d] - _halo_cell_width * _cell_size;

    // Add halo cells to the total counts. Note that we always create a halo,
    // even if on a physical boundary.
    for ( int d = 0; d < 3; ++d )
        _total_num_cell[d] += 2 * _halo_cell_width;
}

//---------------------------------------------------------------------------//

} // end namespace Harlow
