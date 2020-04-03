#ifndef HARLOW_UNIFORMMESH_HPP
#define HARLOW_UNIFORMMESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

#include <mpi.h>

namespace Harlow
{
//---------------------------------------------------------------------------//
/*!
  \class UniformMesh
  \brief Logically and spatially uniform Cartesian mesh.
 */
template<class MemorySpace>
class UniformMesh
{
  public:

    using memory_space = MemorySpace;

    // Construct a mesh manager from the problem bounding box and a property
    // tree.
    UniformMesh( const boost::property_tree::ptree& ptree,
                 const Kokkos::Array<float,6>& global_bounding_box,
                 const int minimum_halo_cell_width,
                 MPI_Comm comm )
    {
        // Get the mesh parameters.
        const auto& mesh_params = ptree.get_child("mesh");

        // Get the global number of cells in each direction and the cell
        // size.
        std::array<int,3> global_num_cell;
        double cell_size;
        if ( mesh_params.count("cell_size") )
        {
            cell_size = mesh_params.get<double>("cell_size");
            for ( int d = 0 ; d < 3; ++d )
            {
                global_num_cell[d] =
                    (global_bounding_box[d+3] - global_bounding_box[d]) /
                    cell_size;
            }
        }
        else if ( mesh_params.count("global_num_cell") )
        {
            if ( mesh_params.get_child("global_num_cell").size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.global_num_cell" );

            int d = 0;
            for ( auto& element : mesh_params.get_child("global_num_cell") )
            {
                global_num_cell[d] = element.second.get_value<int>();
                ++d;
            }
            cell_size =
                (global_bounding_box[3] - global_bounding_box[0]) /
                global_num_cell[0];
        }

        // Because the mesh is uniform check that the domain is evenly
        // divisible by the cell size in each dimension within round-off
        // error. This will let us do cheaper math for particle location.
        for ( int d = 0; d < 3; ++d )
        {
            double extent = global_num_cell[d] * cell_size;
            if ( std::abs(
                     extent - (global_bounding_box[d+3]-
                               global_bounding_box[d]) ) >
                 double( 100.0 ) * std::numeric_limits<double>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double,3> global_low_corner = { global_bounding_box[0],
                                                   global_bounding_box[1],
                                                   global_bounding_box[2] };
        std::array<double,3> global_high_corner = { global_bounding_box[3],
                                                    global_bounding_box[4],
                                                    global_bounding_box[5] };

        // Get the periodicity.
        std::array<bool,3> periodic;
        {
            if ( mesh_params.get_child("periodic").size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.periodic" );

            int d = 0;
            for ( auto& element : mesh_params.get_child("periodic") )
            {
                periodic[d] = element.second.get_value<bool>();
                ++d;
            }
        }

        // For dimensions that are not periodic we pad by the minimum halo
        // cell width to allow for projections outside of the domain.
        for ( int d = 0; d < 3; ++d )
        {
            if ( !periodic[d] )
            {
                global_num_cell[d] += 2*minimum_halo_cell_width;
                global_low_corner[d] -= cell_size*minimum_halo_cell_width;
                global_high_corner[d] += cell_size*minimum_halo_cell_width;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );

        // Create the partitioner.
        const auto& part_params = mesh_params.get_child("partitioner");
        std::shared_ptr<Cajita::Partitioner> partitioner;
        if ( part_params.get<std::string>("type").compare(
                 "uniform_dim") == 0 )
        {
            partitioner = std::make_shared<Cajita::UniformDimPartitioner>();
        }
        else if ( part_params.get<std::string>("type").compare(
                 "manual") == 0 )
        {
            if ( part_params.get_child("ranks_per_dim").size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.partitioner.ranks_per_dim " );

            std::array<int,3> ranks_per_dim;
            int d = 0;
            for ( auto& element :
                      part_params.get_child("ranks_per_dim") )
            {
                ranks_per_dim[d] = element.second.get_value<int>();
                ++d;
            }
            partitioner =
                std::make_shared<Cajita::ManualPartitioner>(ranks_per_dim);
        }

        // Build the global grid.
        auto global_grid = Cajita::createGlobalGrid(
            comm, global_mesh, periodic, *partitioner );

        // Get the halo cell width. If the user does not assign one then it is
        // assumed the minimum halo cell width will be used.
        auto halo_cell_width = std::max(
            minimum_halo_cell_width,
            mesh_params.get<int>("halo_cell_width",0) );

        // Build the local grid.
        _local_grid = Cajita::createLocalGrid( global_grid, halo_cell_width );
    }

    // Get the local grid.
    const Cajita::LocalGrid<Cajita::UniformMesh<double>>& localGrid() const
    {
        return *_local_grid;
    }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().uniformCellSize();
    }

  public:

    std::shared_ptr<
      Cajita::LocalGrid<Cajita::UniformMesh<double>>> _local_grid;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_uniform_mesh_impl : public std::false_type
{
};

template <class MemorySpace>
struct is_uniform_mesh_impl<UniformMesh<MemorySpace>>
    : public std::true_type
{
};

template <class T>
struct is_uniform_mesh
    : public is_uniform_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_UNIFORMMESH_HPP
