#ifndef CAJITA_GRIDBLOCK_HPP
#define CAJITA_GRIDBLOCK_HPP

#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Local Cartesian grid block representation.
//
// Note that a block always has a halo - even if it is on a physical
// boundary that is not periodic. We do this to facilitate particle deposition
// at boundaries.
//
// As a result, the boundary node/cell on a physical boundary that is not
// periodic is the first/last local node/cell depending on whether the low or
// high boundary is chosen.
//---------------------------------------------------------------------------//
class GridBlock
{
  public:

    // Default constructor.
    GridBlock();

    /*!
      \brief Constructor.
      \param local_low_corner The low corner of the space locally owned by the
      block.
      \param local_num_cell The number of cells in each dimension locally
      owned by the block.
      \param boundary_location Boolean indicating if the block is on any of
      the 6 physical boundaries {-x,+x,-y,+y,-z,+z}.
      \param is_dim_periodic Whether each logical dimension is periodic.
      \param cell_size The size of the cells in the mesh. The cells are cubes.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    GridBlock( const std::vector<double>& local_low_corner,
               const std::vector<int>& local_num_cell,
               const std::vector<bool>& boundary_location,
               const std::vector<bool>& is_dim_periodic,
               const double cell_size,
               const int halo_cell_width );

    // Assign the local state of a cartesian grid block with a new halo size.
    void assign( const GridBlock& rhs, const int halo_cell_width );

    // Get the physical coordinates of the low corner of the grid in a given
    // dimension. This low corner includes the halo region.
    double lowCorner( const int dim ) const;

    // Given a physical boundary id return if this grid is on that boundary.
    bool onBoundary( const int boundary_id ) const;

    // Get whether a given logical dimension is periodic.
    bool isPeriodic( const int dim ) const;

    // Given a boundary id return if this has a halo on that boundary. This
    // will be true when this block is not on the domain boundary or, if it
    // is, that boundary is periodic.
    bool hasHalo( const int boundary_id ) const;

    // Get the cell size.
    double cellSize() const;

    // Get the inverse cell size.
    double inverseCellSize() const;

    // Get the number of cells in the halo.
    int haloSize() const;

    // Get the total number of mesh entities in a given dimension including
    // the halo.
    int numEntity( const int entity_type, const int dim ) const;

    // Get the beginning local entity index in a given direction. The local
    // entities do not include the halo.
    //
    // Note that true local ownership is only defined in terms of cells in
    // this partitioning.
    int localEntityBegin( const int entity_type, const int dim ) const;

    // Get the end local entity index in a given direction.
    //
    // Note that true local ownership is only defined in terms of cells in
    // this partitioning.
    int localEntityEnd( const int entity_type, const int dim ) const;

    // Get the local number of entities in a given dimension.
    int localNumEntity( const int entity_type, const int dim ) const;

    // Get the beginning entity index in a given direction in the halo for a
    // neighbor of the given logical index for a requested halo size.
    //
    // Note that nodes on the cells that are at the edges of the local domain
    // are in the halo.
    int haloEntityBegin( const int entity_type,
                         const int dim,
                         const int logical_index,
                         const int halo_num_cell ) const;

    // Get the end entity index in a given direction in the halo for a
    // neighbor of the given logical index for a requested halo size.
    //
    // Note that nodes on the cells that are at the edges of the local domain
    // are in the halo.
    int haloEntityEnd( const int entity_type,
                       const int dim,
                       const int logical_index,
                       const int halo_num_cell ) const;


    // Get the number of entities in a given direction in the halo for a
    // neighbor of a given logical index for a requested halo size.
    //
    // Note that nodes on the cells that are at the edges of the local domain
    // are in the halo.
    int haloNumEntity( const int entity_type,
                       const int dim,
                       const int logical_index,
                       const int halo_num_cell ) const;

  private:

    // Set the halo.
    void setHalo();

  private:

    std::vector<double> _local_low_corner;
    std::vector<double> _low_corner;
    std::vector<int> _local_num_cell;
    std::vector<int> _total_num_cell;
    std::vector<bool> _boundary_location;
    std::vector<bool> _periodic;
    double _cell_size;
    int _halo_cell_width;
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_GRIDBLOCK_HPP
