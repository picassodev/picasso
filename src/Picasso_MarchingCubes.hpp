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

#ifndef PICASSO_MARCHINGCUBES_HPP
#define PICASSO_MARCHINGCUBES_HPP

#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <fstream>
#include <string>

namespace Picasso
{
namespace MarchingCubes
{
//---------------------------------------------------------------------------//
struct LookupTable
{
    // Facet counts.
    static const short counts[256];

    // Facet offsets.
    static const short offsets[256];

    // Facet nodes.
    static const short nodes[836][3];
};

//---------------------------------------------------------------------------//
template <class MemorySpace>
struct Data
{
    // Cell facet case ids and offsets. First value is the case id, second is
    // the offset.
    Kokkos::View<int*** [2], MemorySpace> cell_case_ids_and_offsets;

    // Number of facets.
    int num_facet;

    // Facets. Facets are only created for owned cells.
    Kokkos::View<double* [3][3], MemorySpace> facets;

    // Constructor.
    template <class Mesh>
    Data( const Mesh& mesh )
    {
        auto cell_space = mesh.localGrid()->indexSpace(
            Cajita::Ghost{}, Cajita::Cell{}, Cajita::Local{} );

        cell_case_ids_and_offsets =
            Kokkos::View<int*** [2], typename Mesh::memory_space>(
                Kokkos::ViewAllocateWithoutInitializing(
                    "cell_case_ids_and_offset" ),
                cell_space.extent( 0 ), cell_space.extent( 1 ),
                cell_space.extent( 2 ) );
        Kokkos::deep_copy( cell_case_ids_and_offsets, 0 );
        num_facet = 0;
        facets = Kokkos::View<double* [3][3], typename Mesh::memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "facets" ), 0 );
    }
};

//---------------------------------------------------------------------------//
// Data creation function.
template <class Mesh>
Data<typename Mesh::memory_space> createData( const Mesh& mesh )
{
    return Data<typename Mesh::memory_space>( mesh );
}

//---------------------------------------------------------------------------//
namespace Impl
{
//---------------------------------------------------------------------------//
/*
 Compute vertex data. Ordering of cell vertices is:

    1------------3
   /|           /|
  / |          / |
 5------------7  |
 |  |         |  |
 |  |         |  |
 |  0---------|--2
 | /          | /
 |/           |/
 4------------6

in coordinate layout:

    z
    |
    |
    O------y
   /
  /
 x

*/
template <class SignedDistanceView>
KOKKOS_INLINE_FUNCTION void
vertexData( const SignedDistanceView signed_distance, const int i, const int j,
            const int k, Kokkos::Array<double, 8>& vertex_data )
{
    vertex_data[0] = signed_distance( i, j, k, 0 );
    vertex_data[1] = signed_distance( i, j, k + 1, 0 );
    vertex_data[2] = signed_distance( i, j + 1, k, 0 );
    vertex_data[3] = signed_distance( i, j + 1, k + 1, 0 );
    vertex_data[4] = signed_distance( i + 1, j, k, 0 );
    vertex_data[5] = signed_distance( i + 1, j, k + 1, 0 );
    vertex_data[6] = signed_distance( i + 1, j + 1, k, 0 );
    vertex_data[7] = signed_distance( i + 1, j + 1, k + 1, 0 );
}

//---------------------------------------------------------------------------//
// Compute vertex signs.
KOKKOS_INLINE_FUNCTION
void vertexSigns( const Kokkos::Array<double, 8>& vertex_data,
                  Kokkos::Array<int, 8>& vertex_signs )
{
    for ( int i = 0; i < 8; ++i )
        vertex_signs[i] = ( vertex_data[i] <= 0.0 ) ? 1 : 0;
}

//---------------------------------------------------------------------------//
// Compute the case id.
KOKKOS_INLINE_FUNCTION
int caseId( const Kokkos::Array<int, 8>& vertex_signs )
{
    int case_id = 0;
    for ( int i = 0; i < 8; ++i )
        if ( vertex_signs[i] )
            case_id |= 1UL << i;
    return case_id;
}

//---------------------------------------------------------------------------//
// Get vertex locations.
template <class LocalMesh>
KOKKOS_INLINE_FUNCTION void
vertexLocations( const LocalMesh& local_mesh, const int i, const int j,
                 const int k, Kokkos::Array<double, 6>& vertex_locations )
{
    int index[3] = { i, j, k };
    local_mesh.coordinates( Cajita::Node{}, index, vertex_locations.data() );
    for ( int d = 0; d < 3; ++d )
    {
        ++index[d];
    }
    local_mesh.coordinates( Cajita::Node{}, index,
                            vertex_locations.data() + 3 );
    for ( int d = 0; d < 3; ++d )
    {
        vertex_locations[d + 3] -= vertex_locations[d];
    }
}

//---------------------------------------------------------------------------//
// Compute the edges.
KOKKOS_INLINE_FUNCTION
void computeEdges( const Kokkos::Array<double, 8>& vertex_data,
                   const Kokkos::Array<int, 8>& vertex_signs,
                   Kokkos::Array<Kokkos::Array<double, 3>, 12>& edges )
{
    // Calculate edge weights via linear interpolation.
    auto compute_weight = [&]( const int v1, const int v2 ) {
        return ( vertex_signs[v1] != vertex_signs[v2] )
                   ? vertex_data[v1] / ( vertex_data[v1] - vertex_data[v2] )
                   : -1.0;
    };

    // z edges
    edges[0][0] = 0.0;
    edges[0][1] = 0.0;
    edges[0][2] = compute_weight( 0, 1 );

    edges[1][0] = 0.0;
    edges[1][1] = 1.0;
    edges[1][2] = compute_weight( 2, 3 );

    edges[2][0] = 1.0;
    edges[2][1] = 0.0;
    edges[2][2] = compute_weight( 4, 5 );

    edges[3][0] = 1.0;
    edges[3][1] = 1.0;
    edges[3][2] = compute_weight( 6, 7 );

    // y edges
    edges[4][0] = 0.0;
    edges[4][1] = compute_weight( 0, 2 );
    edges[4][2] = 0.0;

    edges[5][0] = 0.0;
    edges[5][1] = compute_weight( 1, 3 );
    edges[5][2] = 1.0;

    edges[6][0] = 1.0;
    edges[6][1] = compute_weight( 4, 6 );
    edges[6][2] = 0.0;

    edges[7][0] = 1.0;
    edges[7][1] = compute_weight( 5, 7 );
    edges[7][2] = 1.0;

    // x edges
    edges[8][0] = compute_weight( 0, 4 );
    edges[8][1] = 0.0;
    edges[8][2] = 0.0;

    edges[9][0] = compute_weight( 1, 5 );
    edges[9][1] = 0.0;
    edges[9][2] = 1.0;

    edges[10][0] = compute_weight( 2, 6 );
    edges[10][1] = 1.0;
    edges[10][2] = 0.0;

    edges[11][0] = compute_weight( 3, 7 );
    edges[11][1] = 1.0;
    edges[11][2] = 1.0;
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Build the marching cubes triangulation over the owned cells. This
  requires the signed distance values to have been gathered.
  \param exec_space The execution space to use for parallel operations.
  \param signed_distance The signed distance array from which to build the
  facets. This array must be on the nodes as this is a cell-based operation.
  \param data The resulting facet data.
  \param reset_data_memory If true, the facet data will always be
  resized/reallocated. If false, reallocation will only occur when the facet
  or cell count has exceeded the existing count limits.
*/
template <class ExecutionSpace, class Mesh, class SignedDistanceArray,
          class MemorySpace>
void build( const ExecutionSpace& exec_space, const Mesh& mesh,
            const SignedDistanceArray& signed_distance, Data<MemorySpace>& data,
            bool reset_data_memory = false )
{
    static_assert( std::is_same<typename SignedDistanceArray::entity_type,
                                Cajita::Node>::value,
                   "Marching cubes facets may only be constructed from nodal "
                   "distance fields" );

    // Get a view of the signed distance data.
    // NOTE: This array must be gathered before calling this function
    // otherwise parallel data dependencies will not be satisfied for boundary
    // cells.
    auto distance_view = signed_distance.view();

    // Get the cell space we are working on.
    auto cell_space = signed_distance.layout()->localGrid()->indexSpace(
        Cajita::Own{}, Cajita::Cell{}, Cajita::Local{} );

    // Get the case id and number of facets for each cell.
    data.num_facet = 0;
    Cajita::grid_parallel_reduce(
        "marching_cubes_facet_count", exec_space, cell_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k, int& result ) {
            Kokkos::Array<double, 8> vertex_data;
            Impl::vertexData( distance_view, i, j, k, vertex_data );
            Kokkos::Array<int, 8> vertex_signs;
            Impl::vertexSigns( vertex_data, vertex_signs );
            int case_id = Impl::caseId( vertex_signs );
            int num_facet = LookupTable::counts[case_id];
            data.cell_case_ids_and_offsets( i, j, k, 0 ) = case_id;
            data.cell_case_ids_and_offsets( i, j, k, 1 ) = num_facet;
            result += num_facet;
        },
        data.num_facet );

    // Allocate facet data if necessary.
    if ( data.num_facet > static_cast<int>( data.facets.extent( 0 ) ) ||
         reset_data_memory )
    {
        Kokkos::realloc( data.facets, data.num_facet );
    }

    // Compute cell offsets into facet data via exclusive scan.
    Kokkos::parallel_scan(
        "marching_cubes_facet_offset",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, cell_space.size() ),
        KOKKOS_LAMBDA( const int c, int& update, const bool final_pass ) {
            int ek = cell_space.extent( Dim::K );
            int ej = cell_space.extent( Dim::J );
            int ejk = ej * ek;
            int i = c / ejk;
            int j = ( c - i * ejk ) / ek;
            int k = c - i * ejk - j * ek;
            i += cell_space.min( Dim::I );
            j += cell_space.min( Dim::J );
            k += cell_space.min( Dim::K );
            const int value = data.cell_case_ids_and_offsets( i, j, k, 1 );
            if ( final_pass )
            {
                data.cell_case_ids_and_offsets( i, j, k, 1 ) = update;
            }
            update += value;
        } );

    // Fill facets.
    auto local_mesh =
        Cajita::createLocalMesh<MemorySpace>( *( mesh.localGrid() ) );
    Cajita::grid_parallel_for(
        "marching_cubes_fill_facets", exec_space, cell_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int case_id = data.cell_case_ids_and_offsets( i, j, k, 0 );
            int face_offset = data.cell_case_ids_and_offsets( i, j, k, 1 );
            int num_facet = LookupTable::counts[case_id];
            if ( num_facet > 0 )
            {
                Kokkos::Array<double, 8> vertex_data;
                Impl::vertexData( distance_view, i, j, k, vertex_data );
                Kokkos::Array<int, 8> vertex_signs;
                Impl::vertexSigns( vertex_data, vertex_signs );
                Kokkos::Array<double, 6> vertex_locations;
                Impl::vertexLocations( local_mesh, i, j, k, vertex_locations );
                Kokkos::Array<Kokkos::Array<double, 3>, 12> edges;
                Impl::computeEdges( vertex_data, vertex_signs, edges );
                int case_offset = LookupTable::offsets[case_id];
                int edge_id;
                int face_id;
                for ( int f = 0; f < num_facet; ++f )
                {
                    face_id = f + face_offset;
                    for ( int n = 0; n < 3; ++n )
                    {
                        edge_id = LookupTable::nodes[case_offset + f][n];
                        for ( int d = 0; d < 3; ++d )
                        {
                            data.facets( face_id, n, d ) =
                                vertex_locations[d] +
                                vertex_locations[d + 3] * edges[edge_id][d];
                        }
                    }
                }
            }
        } );
}

//---------------------------------------------------------------------------//
// Write facet data to an STL ASCII file for visualization.
template <class MemorySpace>
void writeDataToSTL( const Data<MemorySpace>& data, MPI_Comm comm,
                     const std::string& stl_filename )
{
    // Move facets to the host.
    auto facets =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, data.facets );

    // Create the STL header.
    std::ofstream file( stl_filename );
    if ( !file.is_open() )
        throw std::runtime_error( "Unable to open STL ASCII file" );

    // Make a single surface.
    file << "solid Surface 1" << std::endl;

    // Add each rank's facets to the file. Do a simple parallel file I/O for
    // now.
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );
    int comm_size;
    MPI_Comm_size( comm, &comm_size );
    for ( int r = 0; r < comm_size; ++r )
    {
        if ( r == comm_rank )
        {
            int num_facet = facets.extent( 0 );
            for ( int f = 0; f < num_facet; ++f )
            {
                file << "  facet normal 0.000000e+00 0.000000e+00 0.000000e+00"
                     << std::endl;
                file << "    outer loop" << std::endl;
                for ( int n = 0; n < 3; ++n )
                {
                    file << "      vertex " << facets( f, n, 0 ) << " "
                         << facets( f, n, 1 ) << " " << facets( f, n, 2 )
                         << std::endl;
                }
                file << "    endloop" << std::endl;
                file << "  endfacet" << std::endl;
            }
        }
        MPI_Barrier( comm );
    }

    // Finish the surface.
    file << "endsolid Surface 1" << std::endl;

    // Close the file.
    file.close();
}

//---------------------------------------------------------------------------//

} // end namespace MarchingCubes
} // end namespace Picasso

#endif // end PICASSO_MARCHINGCUBES_HPP
