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

#ifndef PICASSO_ADAPTIVEMESH_HPP
#define PICASSO_ADAPTIVEMESH_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <nlohmann/json.hpp>

#include <memory>

#include <mpi.h>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class AdaptiveMesh
  \brief Logically uniform Cartesian mesh with adaptive node locations.
 */
template <class MemorySpace>
class AdaptiveMesh
{
  public:
    using cajita_mesh = Cajita::UniformMesh<double>;

    using memory_space = MemorySpace;

    using local_grid = Cajita::LocalGrid<cajita_mesh>;

    using node_array = Cajita::Array<double, Cajita::Node,
                                     Cajita::UniformMesh<double>, MemorySpace>;

    static constexpr std::size_t num_space_dim = 3;

    // Construct an adaptive mesh from the problem bounding box and a property
    // tree.
    template <class ExecutionSpace>
    AdaptiveMesh( const nlohmann::json inputs,
                  const Kokkos::Array<double, 6>& global_bounding_box,
                  const int minimum_halo_cell_width, MPI_Comm comm,
                  const ExecutionSpace& exec_space )
    {
        build( inputs, global_bounding_box, minimum_halo_cell_width, comm,
               exec_space );
    }

    // Constructor that uses the default ExecutionSpace for this MemorySpace.
    AdaptiveMesh( const nlohmann::json inputs,
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
        Kokkos::Profiling::pushRegion( "Picasso::AdaptiveMesh::build" );

        _minimum_halo_width = minimum_halo_cell_width;

        // Get the mesh parameters.
        auto mesh_params = inputs["mesh"];

        // Get the global number of cells and cell sizes in each direction.
        std::array<int, 3> global_num_cell;
        Kokkos::Array<double, 3> cell_size;
        if ( mesh_params.count( "cell_size" ) )
        {
            std::array<double, 3> std_cell_size = mesh_params["cell_size"];
            if ( std_cell_size.size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.cell_size" );

            for ( int d = 0; d < 3; ++d )
            {
                cell_size[d] = std_cell_size[d];
                global_num_cell[d] = std::rint(
                    ( global_bounding_box[d + 3] - global_bounding_box[d] ) /
                    cell_size[d] );
            }
        }
        else if ( mesh_params.count( "global_num_cell" ) )
        {
            global_num_cell = mesh_params["global_num_cell"];
            if ( global_num_cell.size() != 3 )
                throw std::runtime_error(
                    "3 entries required for mesh.global_num_cell" );

            for ( int d = 0; d < 3; ++d )
            {
                cell_size[d] =
                    ( global_bounding_box[d + 3] - global_bounding_box[d] ) /
                    global_num_cell[d];
            }
        }

        // Create global mesh.
        std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
        std::array<double, 3> global_high_corner = {
            static_cast<double>( global_num_cell[0] ),
            static_cast<double>( global_num_cell[1] ),
            static_cast<double>( global_num_cell[2] ) };

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
                global_low_corner[d] -= _minimum_halo_width;
                global_high_corner[d] += _minimum_halo_width;
            }
        }

        // Create the global mesh.
        auto global_mesh = Cajita::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );

        // Create the partitioner.
        const auto& part_params = mesh_params["partitioner"];
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
    // Get the minimum required numober of cells in the halo.
    int minimumHaloWidth() const { return _minimum_halo_width; }

    // Get the local grid.
    std::shared_ptr<local_grid> localGrid() const { return _local_grid; }

    // Get the mesh node coordinates.
    std::shared_ptr<node_array> nodes() const { return _nodes; }

    // Build the mesh nodes.
    template <class ExecutionSpace>
    void buildNodes( const Kokkos::Array<double, 3>& cell_size,
                     const ExecutionSpace& exec_space )
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
            "Picasso::AdaptiveMesh::CreateNodes",
            Cajita::createExecutionPolicy( local_space, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                node_view( i, j, k, 0 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 0 ) +
                    i * cell_size[0];
                node_view( i, j, k, 1 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 1 ) +
                    j * cell_size[1];
                node_view( i, j, k, 2 ) =
                    local_mesh.lowCorner( Cajita::Ghost(), 2 ) +
                    k * cell_size[2];
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
struct is_adaptive_mesh_impl : public std::false_type
{
};

template <class MemorySpace>
struct is_adaptive_mesh_impl<AdaptiveMesh<MemorySpace>> : public std::true_type
{
};

template <class T>
struct is_adaptive_mesh
    : public is_adaptive_mesh_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Creation functions.
template <class MemorySpace, class ExecSpace>
auto createAdaptiveMesh( MemorySpace, const nlohmann::json inputs,
                         const Kokkos::Array<double, 6>& global_bounding_box,
                         const int minimum_halo_cell_width, MPI_Comm comm,
                         ExecSpace exec_space )
{
    return std::make_shared<AdaptiveMesh<MemorySpace>>(
        inputs, global_bounding_box, minimum_halo_cell_width, comm,
        exec_space );
}

template <class MemorySpace>
auto createAdaptiveMesh( MemorySpace, const nlohmann::json inputs,
                         const Kokkos::Array<double, 6>& global_bounding_box,
                         const int minimum_halo_cell_width, MPI_Comm comm )
{
    return std::make_shared<AdaptiveMesh<MemorySpace>>(
        inputs, global_bounding_box, minimum_halo_cell_width, comm );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_ADAPTIVEMESH_HPP
