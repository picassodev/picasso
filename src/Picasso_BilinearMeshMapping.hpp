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

#ifndef PICASSO_BILINEARMESHMAPPING_HPP
#define PICASSO_BILINEARMESHMAPPING_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_CurvilinearMesh.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <limits>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class BilinearMeshMapping
  \brief Bilinear mesh mapping function.

  All cells in the mesh are described by physical node locations and a linear
  polynomial basis representing the space continuum between them.
 */
template <class MemorySpace, std::size_t NumSpaceDim>
struct BilinearMeshMapping
{
    using memory_space = MemorySpace;

    static constexpr std::size_t num_space_dim = NumSpaceDim;

    using mesh_type =
        CurvilinearMesh<BilinearMeshMapping<MemorySpace, NumSpaceDim>>;

    using coord_view_type = std::conditional_t<
        3 == num_space_dim, Kokkos::View<double****, MemorySpace>,
        std::conditional_t<2 == num_space_dim,
                           Kokkos::View<double***, MemorySpace>, void>>;

    // Construct a bilinear mapping.
    BilinearMeshMapping( const Kokkos::Array<int, NumSpaceDim>& global_num_cell,
                         const Kokkos::Array<bool, NumSpaceDim>& periodic )
        : _global_num_cell( global_num_cell )
        , _periodic( periodic )
    {
    }

    // Set the local node coordinates including the halo region.
    void setLocalNodeCoordinates( const coord_view_type& local_node_coords )
    {
        _local_node_coords = local_node_coords;
    }

    Kokkos::Array<int, NumSpaceDim> _global_num_cell;
    Kokkos::Array<bool, NumSpaceDim> _periodic;
    Kokkos::Array<int, NumSpaceDim> _global_offset;
    coord_view_type _local_node_coords;
};

//---------------------------------------------------------------------------//
// Template interface implementation. 3D specialization
template <class MemorySpace>
class CurvilinearMeshMapping<BilinearMeshMapping<MemorySpace, 3>>
{
  public:
    using memory_space = MemorySpace;
    using mesh_mapping = BilinearMeshMapping<MemorySpace, 3>;
    static constexpr std::size_t num_space_dim = 3;

    // Get the global number of cells in given logical dimension that construct
    // the mapping.
    static int globalNumCell( const mesh_mapping& mapping, const int dim )
    {
        return mapping._global_num_cell[dim];
    }

    // Get the periodicity of a given logical dimension of the mapping.
    static bool periodic( const mesh_mapping& mapping, const int dim )
    {
        return mapping._periodic[dim];
    }

    // Forward mapping. Given coordinates in the reference frame compute the
    // coordinates in the physical frame.
    template <class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const mesh_mapping& mapping,
                        const ReferenceCoords& reference_coords,
                        PhysicalCoords& physical_coords )
    {
        using value_type = typename ReferenceCoords::value_type;

        int ri = static_cast<int>( reference_coords( Dim::I ) );
        int rj = static_cast<int>( reference_coords( Dim::J ) );
        int rk = static_cast<int>( reference_coords( Dim::K ) );

        int i = ri - mapping._global_offset[Dim::I];
        int j = rj - mapping._global_offset[Dim::J];
        int k = rk - mapping._global_offset[Dim::K];

        value_type xi = reference_coords( Dim::I ) - ri;
        value_type xj = reference_coords( Dim::J ) - rj;
        value_type xk = reference_coords( Dim::K ) - rk;

        value_type w[3][2] = {
            { 1.0 - xi, xi }, { 1.0 - xj, xj }, { 1.0 - xk, xk } };

        physical_coords = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int nk = 0; nk < 2; ++nk )
                    for ( int d = 0; d < 3; ++d )
                        physical_coords( d ) +=
                            w[Dim::I][ni] * w[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node_coords( i + ni, j + nj, k + nk,
                                                        d );
    }

    // Given coordinates in the reference frame compute the grid
    // transformation metrics. This is the jacobian of the forward mapping,
    // its determinant, and inverse.
    template <class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void transformationMetrics(
        const mesh_mapping& mapping, const ReferenceCoords& reference_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type, 3, 3>&
            jacobian,
        typename ReferenceCoords::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type, 3, 3>&
            jacobian_inv )
    {
        using value_type = typename ReferenceCoords::value_type;

        int ri = static_cast<int>( reference_coords( Dim::I ) );
        int rj = static_cast<int>( reference_coords( Dim::J ) );
        int rk = static_cast<int>( reference_coords( Dim::K ) );

        int i = ri - mapping._global_offset[Dim::I];
        int j = rj - mapping._global_offset[Dim::J];
        int k = rk - mapping._global_offset[Dim::K];

        value_type xi = reference_coords( Dim::I ) - ri;
        value_type xj = reference_coords( Dim::J ) - rj;
        value_type xk = reference_coords( Dim::K ) - rk;

        value_type w[3][2] = {
            { 1.0 - xi, xi }, { 1.0 - xj, xj }, { 1.0 - xk, xk } };
        value_type g[3][2] = { { -1.0, 1.0 }, { -1.0, 1.0 }, { -1.0, 1.0 } };

        jacobian = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int nk = 0; nk < 2; ++nk )
                    for ( int d = 0; d < 3; ++d )
                    {
                        jacobian( d, Dim::I ) +=
                            g[Dim::I][ni] * w[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node_coords( i + ni, j + nj, k + nk,
                                                        d );

                        jacobian( d, Dim::J ) +=
                            w[Dim::I][ni] * g[Dim::J][nj] * w[Dim::K][nk] *
                            mapping._local_node_coords( i + ni, j + nj, k + nk,
                                                        d );

                        jacobian( d, Dim::K ) +=
                            w[Dim::I][ni] * w[Dim::J][nj] * g[Dim::K][nk] *
                            mapping._local_node_coords( i + ni, j + nj, k + nk,
                                                        d );
                    }

        jacobian_det = !jacobian;

        jacobian_inv = LinearAlgebra::inverse( jacobian, jacobian_det );
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the reference frame. Return whether or not the mapping
    // succeeded.
    template <class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& reference_coords )
    {
        return DefaultCurvilinearMeshMapping<mesh_mapping>::mapToReferenceFrame(
            mapping, physical_coords, reference_coords );
    }
};

//---------------------------------------------------------------------------//
// Template interface implementation. 2D specialization
template <class MemorySpace>
struct CurvilinearMeshMapping<BilinearMeshMapping<MemorySpace, 2>>
{
    using memory_space = MemorySpace;
    using mesh_mapping = BilinearMeshMapping<MemorySpace, 2>;
    static constexpr std::size_t num_space_dim = 2;

    // Get the global number of cells in given logical dimension that construct
    // the mapping.
    static int globalNumCell( const mesh_mapping& mapping, const int dim )
    {
        return mapping._global_num_cell[dim];
    }

    // Get the periodicity of a given logical dimension of the mapping.
    static bool periodic( const mesh_mapping& mapping, const int dim )
    {
        return mapping._periodic[dim];
    }

    // Forward mapping. Given coordinates in the reference frame compute the
    // coordinates in the physical frame.
    template <class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const mesh_mapping& mapping,
                        const ReferenceCoords& reference_coords,
                        PhysicalCoords& physical_coords )
    {
        using value_type = typename ReferenceCoords::value_type;

        int ri = static_cast<int>( reference_coords( Dim::I ) );
        int rj = static_cast<int>( reference_coords( Dim::J ) );

        int i = ri - mapping._global_offset[Dim::I];
        int j = rj - mapping._global_offset[Dim::J];

        value_type xi = reference_coords( Dim::I ) - ri;
        value_type xj = reference_coords( Dim::J ) - rj;

        value_type w[2][2] = { { 1.0 - xi, xi }, { 1.0 - xj, xj } };

        physical_coords = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int d = 0; d < 2; ++d )
                    physical_coords( d ) +=
                        w[Dim::I][ni] * w[Dim::J][nj] *
                        mapping._local_node_coords( i + ni, j + nj, d );
    }

    // Given coordinates in the reference frame compute the grid
    // transformation metrics. This is the jacobian of the forward mapping,
    // its determinant, and inverse.
    template <class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void transformationMetrics(
        const mesh_mapping& mapping, const ReferenceCoords& reference_coords,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type, 2, 2>&
            jacobian,
        typename ReferenceCoords::value_type& jacobian_det,
        LinearAlgebra::Matrix<typename ReferenceCoords::value_type, 2, 2>&
            jacobian_inv )
    {
        using value_type = typename ReferenceCoords::value_type;

        int ri = static_cast<int>( reference_coords( Dim::I ) );
        int rj = static_cast<int>( reference_coords( Dim::J ) );

        int i = ri - mapping._global_offset[Dim::I];
        int j = rj - mapping._global_offset[Dim::J];

        value_type xi = reference_coords( Dim::I ) - ri;
        value_type xj = reference_coords( Dim::J ) - rj;

        value_type w[2][2] = { { 1.0 - xi, xi }, { 1.0 - xj, xj } };
        value_type g[2][2] = { { -1.0, 1.0 }, { -1.0, 1.0 } };

        jacobian = 0.0;

        for ( int ni = 0; ni < 2; ++ni )
            for ( int nj = 0; nj < 2; ++nj )
                for ( int d = 0; d < 2; ++d )
                {
                    jacobian( d, Dim::I ) +=
                        g[Dim::I][ni] * w[Dim::J][nj] *
                        mapping._local_node_coords( i + ni, j + nj, d );

                    jacobian( d, Dim::J ) +=
                        w[Dim::I][ni] * g[Dim::J][nj] *
                        mapping._local_node_coords( i + ni, j + nj, d );
                }

        jacobian_det = !jacobian;

        jacobian_inv = LinearAlgebra::inverse( jacobian, jacobian_det );
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the reference frame. Return whether or not the mapping
    // succeeded.
    template <class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& reference_coords )
    {
        // NOTE: I am using the default procedure here but it is possible that
        // the procedure outlined on page 329 of the 1986 Brackbill FLIP paper
        // could be more efficient here as it is a direct solve. This default
        // version could be used if that one fails.
        return DefaultCurvilinearMeshMapping<mesh_mapping>::mapToReferenceFrame(
            mapping, physical_coords, reference_coords );
    }
};

//---------------------------------------------------------------------------//
// Bilinear mesh generator template interface.
template <class Generator>
struct BilinearMeshGenerator
{
    static constexpr std::size_t num_space_dim = Generator::num_space_dim;

    // Get the global number of cell in each logical dimension of the mesh.
    static Kokkos::Array<int, num_space_dim>
    globalNumCell( const Generator& generator );

    // Get the peridicity of each logical dimension of the mesh.
    static Kokkos::Array<bool, num_space_dim>
    periodic( const Generator& generator );

    // Given the local grid compute populate the local node coordinates.
    template <class NodeCoordinateArray>
    static void createLocalNodeCoordinates( const Generator& generator,
                                            const NodeCoordinateArray& coords );
};

//---------------------------------------------------------------------------//
// Create a bilinear mesh using a generator. Creates a mapping, a mesh, a
// field manager, a coordinate array in the field manager, assigns the
// coordinate field to the mapping, and populates the coordinates. A field
// manager containing the mesh is returned.
template <class MemorySpace, class Generator, std::size_t NumSpaceDim>
auto createBilinearMesh( MemorySpace, Generator generator, const int halo_width,
                         MPI_Comm comm,
                         const std::array<int, NumSpaceDim>& ranks_per_dim )
{
    // Create the mapping.
    auto mapping =
        std::make_shared<BilinearMeshMapping<MemorySpace, NumSpaceDim>>(
            BilinearMeshGenerator<Generator>::globalNumCell( generator ),
            BilinearMeshGenerator<Generator>::periodic( generator ) );

    // Create mesh.
    auto mesh =
        createCurvilinearMesh( mapping, halo_width, comm, ranks_per_dim );

    // Add the global offset of the domain decomposition. Include the halo
    // width so we know where the ghost logical grid starts.
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        mapping->_global_offset[d] =
            mesh->localGrid()->globalGrid().globalOffset( d ) - halo_width;

    // Create field manager.
    auto manager = createFieldManager( mesh );

    // Add a node coordinates field to the manager.
    manager->add( FieldLocation::Node{},
                  Field::PhysicalPosition<NumSpaceDim>{} );

    // Assign coordinates to the mapping.
    mapping->setLocalNodeCoordinates( manager->view(
        FieldLocation::Node{}, Field::PhysicalPosition<NumSpaceDim>{} ) );

    // Generate the coordinates.
    BilinearMeshGenerator<Generator>::createLocalNodeCoordinates(
        generator,
        *( manager->array( FieldLocation::Node{},
                           Field::PhysicalPosition<NumSpaceDim>{} ) ) );

    return manager;
}

//---------------------------------------------------------------------------//
// Uniform bilinear mesh generator. Generate a uniform bilinear mesh
// starting with a global bounding box and a uniform grid.
template <std::size_t NumSpaceDim>
struct UniformBilinearMeshGenerator
{
    // Uniform cell size.
    double cell_size;

    // Global bounding box.
    Kokkos::Array<double, 2 * NumSpaceDim> global_bounding_box;

    // Boundary periodicity.
    Kokkos::Array<bool, NumSpaceDim> periodic;
};

template <std::size_t NumSpaceDim>
struct BilinearMeshGenerator<UniformBilinearMeshGenerator<NumSpaceDim>>
{
    using Generator = UniformBilinearMeshGenerator<NumSpaceDim>;

    static constexpr std::size_t num_space_dim = NumSpaceDim;

    // Get the global number of cell in each logical dimension of the mesh.
    static Kokkos::Array<int, num_space_dim>
    globalNumCell( const Generator& generator )
    {
        Kokkos::Array<int, num_space_dim> global_num_cell;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            global_num_cell[d] =
                ( generator.global_bounding_box[num_space_dim + d] -
                  generator.global_bounding_box[d] ) /
                generator.cell_size;
        }

        // Because the mesh is uniform check that the domain is evenly
        // divisible by the cell size in each dimension within round-off
        // error. This will let us do cheaper math for particle location.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            double extent = global_num_cell[d] * generator.cell_size;
            if ( std::abs( extent -
                           ( generator.global_bounding_box[num_space_dim + d] -
                             generator.global_bounding_box[d] ) ) >
                 std::numeric_limits<float>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }

        return global_num_cell;
    }

    // Get the peridicity of each logical dimension of the mesh.
    static Kokkos::Array<bool, num_space_dim>
    periodic( const Generator& generator )
    {
        return generator.periodic;
    }

    // Given the local grid compute the local node coordinates. 3D
    // Specialization.
    template <class NodeCoordinateArray, std::size_t NSD = NumSpaceDim>
    static std::enable_if_t<3 == NSD, void>
    createLocalNodeCoordinates( const Generator& generator,
                                const NodeCoordinateArray& coords )
    {
        // Create nodes on the host.
        auto coords_h =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), coords.view() );

        // Create local nodes.
        auto local_grid = coords.layout()->localGrid();
        auto local_space =
            local_grid->indexSpace( Cabana::Grid::Ghost(), Cabana::Grid::Node(),
                                    Cabana::Grid::Local() );
        auto local_mesh =
            Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( *local_grid );
        for ( int i = local_space.min( Dim::I ); i < local_space.max( Dim::I );
              ++i )
            for ( int j = local_space.min( Dim::J );
                  j < local_space.max( Dim::J ); ++j )
                for ( int k = local_space.min( Dim::K );
                      k < local_space.max( Dim::K ); ++k )
                {
                    coords_h( i, j, k, Dim::I ) =
                        generator.global_bounding_box[Dim::I] +
                        ( local_mesh.lowCorner( Cabana::Grid::Ghost(),
                                                Dim::I ) +
                          i ) *
                            generator.cell_size;
                    coords_h( i, j, k, Dim::J ) =
                        generator.global_bounding_box[Dim::J] +
                        ( local_mesh.lowCorner( Cabana::Grid::Ghost(),
                                                Dim::J ) +
                          j ) *
                            generator.cell_size;
                    coords_h( i, j, k, Dim::K ) =
                        generator.global_bounding_box[Dim::K] +
                        ( local_mesh.lowCorner( Cabana::Grid::Ghost(),
                                                Dim::K ) +
                          k ) *
                            generator.cell_size;
                }

        // Move nodes to device.
        Kokkos::deep_copy( coords.view(), coords_h );
    }

    // Given the local grid compute the local node coordinates. 2D
    // Specialization.
    template <class NodeCoordinateArray, std::size_t NSD = NumSpaceDim>
    static std::enable_if_t<2 == NSD, void>
    createLocalNodeCoordinates( const Generator& generator,
                                const NodeCoordinateArray& coords )
    {
        // Create nodes on the host.
        auto coords_h =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), coords.view() );

        // Create owned nodes.
        auto local_grid = coords.layout()->localGrid();
        auto local_space =
            local_grid->indexSpace( Cabana::Grid::Ghost(), Cabana::Grid::Node(),
                                    Cabana::Grid::Local() );
        auto local_mesh =
            Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( *local_grid );
        for ( int i = local_space.min( Dim::I ); i < local_space.max( Dim::I );
              ++i )
            for ( int j = local_space.min( Dim::J );
                  j < local_space.max( Dim::J ); ++j )
            {
                coords_h( i, j, Dim::I ) =
                    generator.global_bounding_box[Dim::I] +
                    ( local_mesh.lowCorner( Cabana::Grid::Ghost(), Dim::I ) +
                      i ) *
                        generator.cell_size;
                coords_h( i, j, Dim::J ) =
                    generator.global_bounding_box[Dim::J] +
                    ( local_mesh.lowCorner( Cabana::Grid::Ghost(), Dim::J ) +
                      j ) *
                        generator.cell_size;
            }

        // Move nodes to device.
        Kokkos::deep_copy( coords.view(), coords_h );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_BILINEARMESHMAPPING_HPP
