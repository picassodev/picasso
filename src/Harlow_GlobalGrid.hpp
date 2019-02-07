#ifndef HARLOW_GLOBALGRID_HPP
#define HARLOW_GLOBALGRID_HPP

#include <Harlow_GridBlock.hpp>

#include <vector>

#include <mpi.h>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Global Cartesian grid.
//---------------------------------------------------------------------------//
class GlobalGrid
{
  public:

    /*!
     \brief Constructor.
     \param comm The communicator over which to define the grid.
     \param ranks_per_dim The number of ranks in which to partition each
     logical dimension.
     \param is_dim_periodic Whether each logical dimension is periodic.
     \param global_low_corner The low corner of the grid in physical space.
     \param global_high_corner The high corner of the grid in physical space.
     \param cell_size The size of the cells in the grid.
    */
    GlobalGrid( MPI_Comm comm,
                const std::vector<int>& ranks_per_dim,
                const std::vector<bool>& is_dim_periodic,
                const std::vector<double>& global_low_corner,
                const std::vector<double>& global_high_corner,
                const double cell_size );

    // Get the grid communicator.
    MPI_Comm comm() const;

    // Get the grid communicator with a 6-neighbor Cartesian
    // topology. Neighbors are ordered in this topology as
    // {-I,+I,-J,+J,-K,+K}.
    MPI_Comm cartesianComm() const;

    // Get the grid communicator with a 26-neighbor graph topology. Neighbors
    // are logically ordered in the 3x3 grid about centered on the local rank
    // with the I index moving the fastest and the K index moving the slowest.
    MPI_Comm graphComm() const;

    // Get the locally owned grid block for this rank.
    const GridBlock& block() const;

    // Get the number of blocks in each dimension in the global mesh.
    int numBlock( const int dim ) const;

    // Get the id of this block in a given dimension.
    int blockId( const int dim ) const;

    // Get whether a given logical dimension is periodic.
    bool isPeriodic( const int dim ) const;

    // Get the global number of entities in a given dimension.
    int numEntity( const int entity_type, const int dim ) const;

    // Get the global offset in a given dimension. This is where our block
    // starts in the global indexing scheme.
    int globalOffset( const int dim ) const;

    // Get the global low corner.
    double lowCorner( const int dim ) const;

    // Get the cell size.
    double cellSize() const;

  private:

    MPI_Comm _cart_comm;
    MPI_Comm _graph_comm;
    GridBlock _grid_block;
    std::vector<int> _ranks_per_dim;
    std::vector<int> _cart_rank;
    std::vector<double> _global_low_corner;
    std::vector<int> _global_num_cell;
    std::vector<int> _global_cell_offset;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GLOBALGRID_HPP
