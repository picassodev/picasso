#include <Cajita_Block.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Constructor.
Block::Block(
    const std::shared_ptr<GlobalGrid>& global_grid,
    const int halo_cell_width )
    : _global_grid( global_grid )
    , _halo_cell_width( halo_cell_width )
{
    // Calculate the owned low corner of the local block.
    _low_corner.resize( 3 );
    for ( int d = 0; d < 3; ++d )
        _low_corner[d] = global_grid->domain().lowCorner(d) +
                         global_grid->cellSize() * global_grid->globalOffset(d);

    // If there are halo cells in the negative direction, subtract the halo
    // width to get the low corner including the halo. There are halo cells if
    // the dimension is periodic or if this is not the left-most block in that
    // dimension.
    for ( int d = 0; d < 3; ++d )
        if ( global_grid->domain().isPeriodic(d) ||
             global_grid->dimBlockId(d) > 0 )
            _low_corner[d] -= _halo_cell_width * global_grid->cellSize();
}

//---------------------------------------------------------------------------//
// Get the global grid that owns the block.
const GlobalGrid& Block::globalGrid() const
{
    return *_global_grid;
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the low corner of the grid in a given
// dimension. This low corner includes the halo region.
double Block::lowCorner( const int dim ) const
{
    return _low_corner[dim];
}

//---------------------------------------------------------------------------//
// Get the halo size.
int Block::haloWidth() const
{
    return _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Given the relative offsets of a neighbor rank relative to this block's
// indices get the of the neighbor. If the neighbor rank is out of bounds
// return -1. Note that in the case of periodic boundaries out of bounds
// indices are allowed as the indices will be wrapped around the periodic
// boundary.
int Block::neighborRank( const int off_i,
                         const int off_j,
                         const int off_k ) const
{
    return _global_grid->blockRank( _global_grid->dimBlockId(Dim::I) + off_i,
                                    _global_grid->dimBlockId(Dim::J) + off_j,
                                    _global_grid->dimBlockId(Dim::K) + off_k );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned cells.
IndexSpace<3> Block::ownedIndexSpace( Cell ) const
{
    // Compute the lower bound.
    std::vector<long> min(3);
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic(d) ||
                   _global_grid->dimBlockId(d) > 0 )
                 ? _halo_cell_width
                 : 0;

    // Compute the upper bound.
    std::vector<long> max(3);
    for ( int d = 0; d < 3; ++d )
        max[d] = min[d] + _global_grid->ownedNumCell(d);

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted cells.
IndexSpace<3> Block::ghostedIndexSpace( Cell ) const
{
    // Compute the size.
    std::vector<long> size( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of cells.
        size[d] = _global_grid->ownedNumCell(d);

        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) <
             _global_grid->dimNumBlock(d) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local cell
// indices we own that we share with that neighbor to use as ghosts.
IndexSpace<3> Block::sharedOwnedIndexSpace( Cell,
                                            const int off_i,
                                            const int off_j,
                                            const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Cell() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min(d);

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min(d) + _halo_cell_width;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local cell
// indices owned by that neighbor that are shared with us to use as
// ghosts.
IndexSpace<3> Block::sharedGhostedIndexSpace( Cell,
                                              const int off_i,
                                              const int off_j,
                                              const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Cell() );

    // Get the ghosted index space.
    auto ghosted_space = ghostedIndexSpace( Cell() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            min[d] = ghosted_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d);
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            max[d] = owned_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = ghosted_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes.
IndexSpace<3> Block::ownedIndexSpace( Node ) const
{
    // Compute the lower bound.
    std::vector<long> min(3);
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic(d) ||
                   _global_grid->dimBlockId(d) > 0 )
                 ? _halo_cell_width
                 : 0;

    // Compute the upper bound.
    std::vector<long> max(3);
    for ( int d = 0; d < 3; ++d )
        max[d] = ( _global_grid->domain().isPeriodic(d) ||
                   _global_grid->dimBlockId(d) <
                   _global_grid->dimNumBlock(d) - 1 )
                 ? min[d] + _global_grid->ownedNumCell(d)
                 : min[d] + _global_grid->ownedNumCell(d) + 1;

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted nodes.
IndexSpace<3> Block::ghostedIndexSpace( Node ) const
{
    // Compute the size.
    std::vector<long> size( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of nodes.
        size[d] = _global_grid->ownedNumCell(d) + 1;

        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) <
             _global_grid->dimNumBlock(d) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local node
// indices we own that we share with that neighbor to use as ghosts.
IndexSpace<3> Block::sharedOwnedIndexSpace( Node,
                                            const int off_i,
                                            const int off_j,
                                            const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Node() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min(d);

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min(d) + _halo_cell_width + 1;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local node
// indices owned by that neighbor that are shared with us to use as
// ghosts.
IndexSpace<3> Block::sharedGhostedIndexSpace( Node,
                                              const int off_i,
                                              const int off_j,
                                              const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Node() );

    // Get the ghosted index space.
    auto ghosted_space = ghostedIndexSpace( Node() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            min[d] = ghosted_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d);
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            max[d] = owned_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = ghosted_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned Dir-direction faces.
template<int Dir>
IndexSpace<3> Block::ownedIndexSpace( Face<Dir> ) const
{
    // Compute the lower bound.
    std::vector<long> min(3);
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic(d) ||
                   _global_grid->dimBlockId(d) > 0 )
                 ? _halo_cell_width
                 : 0;

    // Compute the upper bound.
    std::vector<long> max(3);
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            max[d] = ( _global_grid->domain().isPeriodic(d) ||
                            _global_grid->dimBlockId(d) <
                            _global_grid->dimNumBlock(d) - 1 )
                          ? min[d] + _global_grid->ownedNumCell(d)
                          : min[d] + _global_grid->ownedNumCell(d) + 1;
        }
        else
        {
            max[d] = min[d] + _global_grid->ownedNumCell(d);
        }
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted Dir-direction faces.
template<int Dir>
IndexSpace<3> Block::ghostedIndexSpace( Face<Dir> ) const
{
    // Compute the size.
    std::vector<long> size( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dir == d)
        {
            size[d] = _global_grid->ownedNumCell(d) + 1;
        }
        else
        {
            size[d] = _global_grid->ownedNumCell(d);
        }
    }

    for ( int d = 0; d < 3; ++d )
    {
        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic(d) ||
             _global_grid->dimBlockId(d) <
             _global_grid->dimNumBlock(d) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction face indices we own that we share with that neighbor to use
// as ghosts.
template<int Dir>
IndexSpace<3> Block::sharedOwnedIndexSpace( Face<Dir>,
                                            const int off_i,
                                            const int off_j,
                                            const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Face<Dir>() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min(d);

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = (Dir == d)
                     ? owned_space.min(d) + _halo_cell_width + 1
                     : owned_space.min(d) + _halo_cell_width;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local
// Dir-direction face indices owned by that neighbor that are shared with us
// to use as ghosts.
template<int Dir>
IndexSpace<3> Block::sharedGhostedIndexSpace( Face<Dir>,
                                              const int off_i,
                                              const int off_j,
                                              const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i ||
         off_j < -1 || 1 < off_j ||
         off_k < -1 || 1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank(off_i,off_j,off_k) < 0 )
        return IndexSpace<3>( {0,0,0}, {0,0,0} );

    // Wrap the indices.
    std::vector<long> nid = { off_i, off_j, off_k };

    // Get the owned index space.
    auto owned_space = ownedIndexSpace( Face<Dir>() );

    // Get the ghosted index space.
    auto ghosted_space = ghostedIndexSpace( Face<Dir>() );

    // Compute the lower bound.
    std::vector<long> min(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            min[d] = ghosted_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            min[d] = owned_space.min(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max(d);
    }

    // Compute the upper bound.
    std::vector<long> max(3,-1);
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d])
            max[d] = owned_space.min(d);

        // Middle neighbor
        else if ( 0 == nid[d] )
            max[d] = owned_space.max(d);

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = ghosted_space.max(d);
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Face indexing explicit instantiations.
//---------------------------------------------------------------------------//
template IndexSpace<3> Block::ownedIndexSpace( Face<Dim::I> ) const;
template IndexSpace<3> Block::ghostedIndexSpace( Face<Dim::I> ) const;
template IndexSpace<3> Block::sharedOwnedIndexSpace(
    Face<Dim::I>, const int, const int, const int ) const;
template IndexSpace<3> Block::sharedGhostedIndexSpace(
    Face<Dim::I>, const int, const int, const int ) const;

template IndexSpace<3> Block::ownedIndexSpace( Face<Dim::J> ) const;
template IndexSpace<3> Block::ghostedIndexSpace( Face<Dim::J> ) const;
template IndexSpace<3> Block::sharedOwnedIndexSpace(
    Face<Dim::J>, const int, const int, const int ) const;
template IndexSpace<3> Block::sharedGhostedIndexSpace(
    Face<Dim::J>, const int, const int, const int ) const;

template IndexSpace<3> Block::ownedIndexSpace( Face<Dim::K> ) const;
template IndexSpace<3> Block::ghostedIndexSpace( Face<Dim::K> ) const;
template IndexSpace<3> Block::sharedOwnedIndexSpace(
    Face<Dim::K>, const int, const int, const int ) const;
template IndexSpace<3> Block::sharedGhostedIndexSpace(
    Face<Dim::K>, const int, const int, const int ) const;

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
std::shared_ptr<Block> createBlock(
    const std::shared_ptr<GlobalGrid>& global_grid,
    const int halo_cell_width )
{
    return std::make_shared<Block>( global_grid, halo_cell_width );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
