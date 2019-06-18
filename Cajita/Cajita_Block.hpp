#ifndef CAJITA_BLOCK_HPP
#define CAJITA_BLOCK_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <vector>
#include <memory>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Local Cartesian grid block.
//---------------------------------------------------------------------------//
class Block
{
  public:

    /*!
      \brief Constructor.
      \param global_grid The global grid from which the block will be
      constructed.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    Block( const std::shared_ptr<GlobalGrid>& global_grid,
           const int halo_cell_width );

    // Get the global grid that owns the block.
    const GlobalGrid& globalGrid() const;

    // Get the physical coordinates of the low corner of the block in a given
    // dimension. This low corner includes the halo region.
    double lowCorner( const int dim ) const;

    // Get the number of cells in the halo.
    int haloWidth() const;

    // Given the relative offsets of a neighbor rank relative to this block's
    // indices get the of the neighbor. If the neighbor rank is out of bounds
    // return -1. Note that in the case of periodic boundaries out of bounds
    // indices are allowed as the indices will be wrapped around the periodic
    // boundary.
    int neighborRank( const int off_i, const int off_j, const int off_k ) const;

    // Get the local index space of the owned cells.
    IndexSpace<3> ownedIndexSpace( Cell ) const;

    // Get the local index space of the owned and ghosted cells.
    IndexSpace<3> ghostedIndexSpace( Cell ) const;

    // Given a relative set of indices of a neighbor get the set of local cell
    // indices we own that we share with that neighbor to use as ghosts.
    IndexSpace<3> sharedOwnedIndexSpace( Cell,
                                         const int off_i,
                                         const int off_j,
                                         const int off_k ) const;

    // Given a relative set of indices of a neighbor get set of local cell
    // indices owned by that neighbor that are shared with us to use as
    // ghosts.
    IndexSpace<3> sharedGhostedIndexSpace( Cell,
                                           const int off_i,
                                           const int off_j,
                                           const int off_k ) const;

    // Get the local index space of the owned nodes.
    IndexSpace<3> ownedIndexSpace( Node ) const;

    // Get the local index space of the owned and ghosted nodes.
    IndexSpace<3> ghostedIndexSpace( Node ) const;

    // Given a relative set of indices of a neighbor get the set of local node
    // indices we own that we share with that neighbor to use as ghosts.
    IndexSpace<3> sharedOwnedIndexSpace( Node,
                                         const int off_i,
                                         const int off_j,
                                         const int off_k ) const;

    // Given a relative set of indices of a neighbor get set of local node
    // indices owned by that neighbor that are shared with us to use as
    // ghosts.
    IndexSpace<3> sharedGhostedIndexSpace( Node,
                                           const int off_i,
                                           const int off_j,
                                           const int off_k ) const;

    // Get the local index space of the owned Dir-direction faces.
    template<int Dir>
    IndexSpace<3> ownedIndexSpace( Face<Dir> ) const;

    // Get the local index space of the owned and ghosted Dir-direction faces.
    template<int Dir>
    IndexSpace<3> ghostedIndexSpace( Face<Dir> ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // Dir-direction face indices we own that we share with that neighbor to use
    // as ghosts.
    template<int Dir>
    IndexSpace<3> sharedOwnedIndexSpace( Face<Dir>,
                                         const int off_i,
                                         const int off_j,
                                         const int off_k ) const;

    // Given a relative set of indices of a neighbor get set of local
    // Dir-direction face indices owned by that neighbor that are shared with
    // us to use as ghosts.
    template<int Dir>
    IndexSpace<3> sharedGhostedIndexSpace( Face<Dir>,
                                           const int off_i,
                                           const int off_j,
                                           const int off_k ) const;

  private:

    std::shared_ptr<GlobalGrid> _global_grid;
    std::vector<double> _low_corner;
    int _halo_cell_width;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a block.
  \param global_grid The global grid from which the block will be
  constructed.
  \param halo_cell_width The number of halo cells surrounding the locally
  owned cells.
*/
std::shared_ptr<Block> createBlock(
    const std::shared_ptr<GlobalGrid>& global_grid,
    const int halo_cell_width );

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_BLOCK_HPP
