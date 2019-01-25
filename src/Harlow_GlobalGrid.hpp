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

    // Get the grid communicator. This communicator has a Cartesian topology.
    MPI_Comm comm() const;

    // Get the locally owned grid block for this rank.
    const GridBlock& block() const;

    // Get whether a given logical dimension is periodic.
    bool isPeriodic( const int dim ) const;

    // Get the global number of cells in a given dimension.
    int numCell( const int dim ) const;

    // Get the global number of nodes in a given dimension.
    int numNode( const int dim ) const;

    // Get the global low corner.
    double lowCorner( const int dim ) const;

    // Get the cell size.
    double cellSize() const;

  private:

    MPI_Comm _cart_comm;
    GridBlock _grid_block;
    std::vector<double> _global_low_corner;
    std::vector<int> _global_num_cell;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GLOBALGRID_HPP
