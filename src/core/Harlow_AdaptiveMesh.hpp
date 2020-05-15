#ifndef HARLOW_ADAPTIVEMESH_HPP
#define HARLOW_ADAPTIVEMESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

#include <mpi.h>

namespace Harlow
{
//---------------------------------------------------------------------------//
/*!
  \class AdaptiveMesh
  \brief Logically uniform Cartesian mesh with adaptive node locations.
 */
template<class MemorySpace>
class AdaptiveMesh
{
  public:

    using memory_space = MemorySpace;

    using local_grid = Cajita::LocalGrid<Cajita::UniformMesh<double>>;

    using node_array = Cajita::Array<double,Cajita::Node,
                                     Cajita::UniformMesh<double>,MemorySpace>;

    // Construct an adaptive mesh from the problem bounding box and a property
    // tree.
    template<class ExecutionSpace>
    AdaptiveMesh( const boost::property_tree::ptree& ptree,
                  const Kokkos::Array<double,6>& global_bounding_box,
                  const int minimum_halo_cell_width,
                  MPI_Comm comm,
                  const ExecutionSpace& exec_space )
        : _minimum_halo_width( minimum_halo_cell_width )
    {
        // Get the mesh parameters.
        const auto& mesh_params = ptree.get_child("mesh");

        // Get the global number of cells and cell sizes in each direction.
        std::array<int,3> global_num_cell;
        Kokkos::Array<double,3> cell_size;
        if ( mesh_params.count("cell_size") )
        {
            if ( mesh_params.get_child("cell_size").size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.cell_size" );

            int d = 0;
            for ( auto& element : mesh_params.get_child("cell_size") )
            {
                cell_size[d] = element.second.get_value<double>();
                global_num_cell[d] =
                    std::rint(
                        (global_bounding_box[d+3] - global_bounding_box[d]) /
                        cell_size[d] );
                ++d;
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
                cell_size[d] =
                    (global_bounding_box[d+3] - global_bounding_box[d]) /
                    global_num_cell[d];
                ++d;
            }
        }

        // Create global mesh.
        std::array<double,3> global_low_corner = { 0.0, 0.0, 0.0 };
        std::array<double,3> global_high_corner =
            { static_cast<double>(global_num_cell[0]),
              static_cast<double>(global_num_cell[1]),
              static_cast<double>(global_num_cell[2]) };
        Kokkos::Array<double,3> physical_low_corner = { global_bounding_box[0],
                                                        global_bounding_box[1],
                                                        global_bounding_box[2] };

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
                global_num_cell[d] += 2*_minimum_halo_width;
                global_low_corner[d] -= _minimum_halo_width;
                global_high_corner[d] += _minimum_halo_width;
                physical_low_corner[d] -= cell_size[d]*_minimum_halo_width;
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
            _minimum_halo_width,
            mesh_params.get<int>("halo_cell_width",0) );

        // Build the local grid.
        _local_grid =
            Cajita::createLocalGrid( global_grid, halo_cell_width );

        // Create the nodes.
        buildNodes( physical_low_corner, cell_size, exec_space );
    }

    // Get the minimum required numober of cells in the halo.
    int minimumHaloWidth() const
    {
        return _minimum_halo_width;
    }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const
    {
        return _local_grid;
    }

    // Get the mesh node coordinates.
    std::shared_ptr<node_array> nodes() const
    {
        return _nodes;
    }

  public:

    // Build the mesh nodes.
    template<class ExecutionSpace>
    void buildNodes( const Kokkos::Array<double,3>& physical_low_corner,
                     const Kokkos::Array<double,3>& cell_size,
                     const ExecutionSpace& exec_space )
    {
        // Create both owned and ghosted nodes so we don't have to gather
        // initially.
        auto node_layout =
            Cajita::createArrayLayout( _local_grid, 3, Cajita::Node() );
        _nodes = Cajita::createArray<double,MemorySpace>(
            "mesh_nodes", node_layout );
        auto node_view = _nodes->view();
        auto global_space = _local_grid->indexSpace(
            Cajita::Ghost(),Cajita::Node(),Cajita::Global());
        auto local_space = _local_grid->indexSpace(
            Cajita::Ghost(),Cajita::Node(),Cajita::Local());
        Kokkos::parallel_for(
            "create_nodes",
            Cajita::createExecutionPolicy(local_space,exec_space),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                int ig = global_space.min(0) + i;
                int jg = global_space.min(1) + j;
                int kg = global_space.min(2) + k;
                node_view(i,j,k,0) = physical_low_corner[0] + cell_size[0] * ig;
                node_view(i,j,k,1) = physical_low_corner[1] + cell_size[1] * jg;
                node_view(i,j,k,2) = physical_low_corner[2] + cell_size[2] * kg;
            });
    }

  public:

    int _minimum_halo_width;
    std::shared_ptr<local_grid> _local_grid;
    std::shared_ptr<node_array> _nodes;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_adaptive_mesh_impl : public std::false_type
{
};

template <class MemorySpace>
struct is_adaptive_mesh_impl<AdaptiveMesh<MemorySpace>>
    : public std::true_type
{
};

template <class T>
struct is_adaptive_mesh
    : public is_adaptive_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_ADAPTIVEMESH_HPP
