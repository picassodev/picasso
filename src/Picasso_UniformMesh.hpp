/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_UNIFORMMESH_HPP
#define PICASSO_UNIFORMMESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <nlohmann/json.hpp>

#include <memory>

#include <mpi.h>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class UniformMesh
  \brief Logically and spatially uniform Cartesian mesh.
 */
template <class MemorySpace>
class UniformMesh
{
  public:
    using cajita_mesh = Cajita::UniformMesh<double>;

    using memory_space = MemorySpace;

    using local_grid = Cajita::LocalGrid<cajita_mesh>;

    using node_array = Cajita::Array<double, Cajita::Node,
                                     Cajita::UniformMesh<double>, MemorySpace>;

    static constexpr std::size_t num_space_dim = 3;

    // Construct a mesh manager from the problem bounding box and a property
    // tree.
    template <class ExecutionSpace>
    UniformMesh( const nlohmann::json inputs,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_cell_width, MPI_Comm comm,
                 const ExecutionSpace& exec_space )
    {
        build( inputs, global_bounding_box, minimum_halo_cell_width, comm,
               exec_space );
    }

    // Constructor that uses the default ExecutionSpace for this MemorySpace.
    UniformMesh( const nlohmann::json inputs,
                 const Kokkos::Array<double, 6>& global_bounding_box,
                 const int minimum_halo_cell_width, MPI_Comm comm )
    {
        using exec_space = typename memory_space::execution_space;

        build( inputs, global_bounding_box, minimum_halo_cell_width, comm,
               exec_space{} );
    }

  private:
    template <class ExecutionSpace>
    void build( const nlohmann::json inputs,
                const Kokkos::Array<double, 6>& global_bounding_box,
                const int minimum_halo_cell_width, MPI_Comm comm,
                const ExecutionSpace& exec_space )
    {
        Kokkos::Profiling::pushRegion( "Picasso::UniformMesh::build" );

        _minimum_halo_width = minimum_halo_cell_width;

        // Get the mesh parameters.
        auto mesh_params = inputs["mesh"];

        // Get the global number of cells in each direction and the cell size.
        std::array<int, 3> global_num_cell;
        double cell_size = 0.0;
        if ( mesh_params.count( "cell_size" ) )
        {
            cell_size = mesh_params["cell_size"];
            for ( int d = 0; d < 3; ++d )
            {
                global_num_cell[d] = std::rint(
                    ( global_bounding_box[d + 3] - global_bounding_box[d] ) /
                    cell_size );
            }
        }
        else if ( mesh_params.count( "global_num_cell" ) )
        {
            global_num_cell = mesh_params["global_num_cell"];
            if ( global_num_cell.size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.global_num_cell" );

            cell_size = ( global_bounding_box[3] - global_bounding_box[0] ) /
                        global_num_cell[0];
        }
        else
        {
            throw std::runtime_error( "Invalid uniform mesh size parameters" );
        }

        // Because the mesh is uniform check that the domain is evenly
        // divisible by the cell size in each dimension within round-off
        // error. This will let us do cheaper math for particle location.
        for ( int d = 0; d < 3; ++d )
        {
            double extent = global_num_cell[d] * cell_size;
            if ( std::abs( extent - ( global_bounding_box[d + 3] -
                                      global_bounding_box[d] ) ) >
                 std::numeric_limits<float>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        // Create global mesh bounds.
        std::array<double, 3> global_low_corner = { global_bounding_box[0],
                                                    global_bounding_box[1],
                                                    global_bounding_box[2] };
        std::array<double, 3> global_high_corner = { global_bounding_box[3],
                                                     global_bounding_box[4],
                                                     global_bounding_box[5] };

        // Get the periodicity.
        std::array<bool, 3> periodic = mesh_params["periodic"];
        if ( periodic.size() != 3 )
            throw std::runtime_error( "3 entries required for mesh.periodic" );

        // For dimensions that are not periodic we pad by the minimum halo
        // cell width to allow for projections outside of the domain.
        for ( int d = 0; d < 3; ++d )
        {
            if ( !periodic[d] )
            {
                global_num_cell[d] += 2 * _minimum_halo_width;
                global_low_corner[d] -= cell_size * _minimum_halo_width;
                global_high_corner[d] += cell_size * _minimum_halo_width;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );

        // Create the partitioner.
        auto part_params = mesh_params["partitioner"];
        std::shared_ptr<Cajita::BlockPartitioner<3>> partitioner;
        std::string type = part_params["type"];
        if ( type.compare( "uniform_dim" ) == 0 )
        {
            partitioner = std::make_shared<Cajita::DimBlockPartitioner<3>>();
        }
        else if ( type.compare( "manual" ) == 0 )
        {
            std::array<int, 3> ranks_per_dim = part_params["ranks_per_dim"];
            if ( ranks_per_dim.size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.partitioner.ranks_per_dim " );

            partitioner = std::make_shared<Cajita::ManualBlockPartitioner<3>>(
                ranks_per_dim );
        }

        // Build the global grid.
        auto global_grid = Cajita::createGlobalGrid( comm, global_mesh,
                                                     periodic, *partitioner );

        // Get the halo cell width. If the user does not assign one then it is
        // assumed the minimum halo cell width will be used.
        int read_halo_width = 0;
        if ( mesh_params.count( "halo_cell_width" ) )
            read_halo_width = mesh_params["halo_cell_width"];
        auto halo_cell_width = std::max( _minimum_halo_width, read_halo_width );

        // Build the local grid.
        _local_grid = Cajita::createLocalGrid( global_grid, halo_cell_width );

        // Create the nodes.
        buildNodes( cell_size, exec_space );

        Kokkos::Profiling::popRegion();
    }

  public:
    // Get the minimum required number of cells in the halo.
    int minimumHaloWidth() const { return _minimum_halo_width; }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const { return _local_grid; }

    // Get the mesh node coordinates.
    std::shared_ptr<node_array> nodes() const { return _nodes; }

    // Get the cell size.
    double cellSize() const
    {
        return _local_grid->globalGrid().globalMesh().cellSize( 0 );
    }

    // Build the mesh nodes.
    template <class ExecutionSpace>
    void buildNodes( const double cell_size, const ExecutionSpace& exec_space )
    {
        // Create both owned and ghosted nodes so we don't have to gather
        // initially.
        auto node_layout =
            Cajita::createArrayLayout( _local_grid, 3, Cajita::Node() );
        _nodes = Cajita::createArray<double, MemorySpace>( "mesh_nodes",
                                                           node_layout );
        auto node_view = _nodes->view();
        auto local_mesh =
            Cajita::createLocalMesh<ExecutionSpace>( *_local_grid );
        auto local_space = _local_grid->indexSpace(
            Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
        Kokkos::parallel_for(
            "Picasso::UniformMesh::create_nodes",
            Cajita::createExecutionPolicy( local_space, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                node_view( i, j, k, 0 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 0 ) + i * cell_size;
                node_view( i, j, k, 1 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 1 ) + j * cell_size;
                node_view( i, j, k, 2 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 2 ) + k * cell_size;
            } );
    }

  public:
    int _minimum_halo_width;
    std::shared_ptr<local_grid> _local_grid;
    std::shared_ptr<node_array> _nodes;
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_uniform_mesh_impl : public std::false_type
{
};

template <class MemorySpace>
struct is_uniform_mesh_impl<UniformMesh<MemorySpace>> : public std::true_type
{
};

template <class T>
struct is_uniform_mesh
    : public is_uniform_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Creation function.
template <class MemorySpace, class ExecSpace>
auto createUniformMesh( MemorySpace, const nlohmann::json inputs,
                        const Kokkos::Array<double, 6>& global_bounding_box,
                        const int minimum_halo_cell_width, MPI_Comm comm,
                        ExecSpace exec_space )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        inputs, global_bounding_box, minimum_halo_cell_width, comm,
        exec_space );
}

template <class MemorySpace>
auto createUniformMesh( MemorySpace, const nlohmann::json inputs,
                        const Kokkos::Array<double, 6>& global_bounding_box,
                        const int minimum_halo_cell_width, MPI_Comm comm )
{
    return std::make_shared<UniformMesh<MemorySpace>>(
        inputs, global_bounding_box, minimum_halo_cell_width, comm );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_UNIFORMMESH_HPP
