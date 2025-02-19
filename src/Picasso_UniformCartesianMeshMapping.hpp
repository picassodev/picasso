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

#ifndef PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP
#define PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP

#include <Picasso_CurvilinearMesh.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_Types.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <limits>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
/*!
  \class UniformCartesianMeshMapping
  \brief Uniform Cartesian mesh mapping function.
 */
template <class MemorySpace, std::size_t NumSpaceDim>
struct UniformCartesianMeshMapping
{
    using memory_space = MemorySpace;
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    double _cell_size;
    double _inv_cell_size;
    double _cell_measure;
    double _inv_cell_measure;
    Kokkos::Array<int, NumSpaceDim> _global_num_cell;
    Kokkos::Array<bool, NumSpaceDim> _periodic;
    Kokkos::Array<double, 2 * NumSpaceDim> _global_bounding_box;
};

//---------------------------------------------------------------------------//
// Template interface implementation.
template <class MemorySpace, std::size_t NumSpaceDim>
class CurvilinearMeshMapping<
    UniformCartesianMeshMapping<MemorySpace, NumSpaceDim>>
{
  public:
    using memory_space = MemorySpace;
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    using mesh_mapping = UniformCartesianMeshMapping<MemorySpace, NumSpaceDim>;

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

    // Forward mapping. Given coordinates in the reference frame
    // compute the coordinates in the physical frame.
    template <class ReferenceCoords, class PhysicalCoords>
    static KOKKOS_INLINE_FUNCTION void
    mapToPhysicalFrame( const mesh_mapping& mapping,
                        const ReferenceCoords& reference_coords,
                        PhysicalCoords& physical_coords )
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            physical_coords( d ) = mapping._global_bounding_box[d] +
                                   reference_coords( d ) * mapping._cell_size;
    }

    // Given coordinates in the reference frame compute the grid
    // transformation metrics. This is the jacobian of the forward mapping,
    // its determinant, and inverse.
    template <class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION void transformationMetrics(
        const mesh_mapping& mapping, const ReferenceCoords&,
        Cabana::LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
                                      NumSpaceDim, NumSpaceDim>& jacobian,
        typename ReferenceCoords::value_type& jacobian_det,
        Cabana::LinearAlgebra::Matrix<typename ReferenceCoords::value_type,
                                      NumSpaceDim, NumSpaceDim>& jacobian_inv )
    {
        for ( std::size_t i = 0; i < num_space_dim; ++i )
            for ( std::size_t j = 0; j < num_space_dim; ++j )
                jacobian( i, j ) = ( i == j ) ? mapping._cell_size : 0.0;

        jacobian_det = mapping._inv_cell_measure;

        for ( std::size_t i = 0; i < num_space_dim; ++i )
            for ( std::size_t j = 0; j < num_space_dim; ++j )
                jacobian_inv( i, j ) =
                    ( i == j ) ? mapping._inv_cell_size : 0.0;
    }

    // Reverse mapping. Given coordinates in the physical frame compute the
    // coordinates in the reference frame. Return whether or not the
    // mapping succeeded.
    template <class PhysicalCoords, class ReferenceCoords>
    static KOKKOS_INLINE_FUNCTION bool
    mapToReferenceFrame( const mesh_mapping& mapping,
                         const PhysicalCoords& physical_coords,
                         ReferenceCoords& reference_coords )
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            reference_coords( d ) =
                ( physical_coords( d ) - mapping._global_bounding_box[d] ) *
                mapping._inv_cell_size;
        return true;
    }
};

//---------------------------------------------------------------------------//
// Create a uniform mesh. Creates a mapping, a mesh, a field manager, a
// coordinate array in the field manager, and assigns the global mesh bounds to
// the mapping. A field manager containing the mesh is returned.
template <class MemorySpace, std::size_t NumSpaceDim>
auto createUniformCartesianMesh(
    MemorySpace, const double cell_size,
    const Kokkos::Array<double, 2 * NumSpaceDim>& global_bounding_box,
    const Kokkos::Array<bool, NumSpaceDim>& periodic, const int halo_width,
    MPI_Comm comm, const std::array<int, NumSpaceDim>& ranks_per_dim )
{
    // Create the mapping.
    auto mapping = std::make_shared<
        UniformCartesianMeshMapping<MemorySpace, NumSpaceDim>>();
    mapping->_cell_size = cell_size;
    mapping->_inv_cell_size = 1.0 / cell_size;
    mapping->_cell_measure = pow( cell_size, NumSpaceDim );
    mapping->_inv_cell_measure = 1.0 / mapping->_cell_measure;
    mapping->_periodic = periodic;
    mapping->_global_bounding_box = global_bounding_box;
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        mapping->_global_num_cell[d] = std::rint(
            ( global_bounding_box[NumSpaceDim + d] - global_bounding_box[d] ) /
            cell_size );
    }

    // Because the mesh is uniform check that the domain is evenly
    // divisible by the cell size in each dimension within round-off
    // error. This will let us do cheaper math for particle location.
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        double extent = mapping->_global_num_cell[d] * cell_size;
        if ( std::abs( extent - ( global_bounding_box[NumSpaceDim + d] -
                                  global_bounding_box[d] ) ) >
             std::numeric_limits<float>::epsilon() )
            throw std::logic_error(
                "Extent not evenly divisible by uniform cell size" );
    }

    // Create mesh.
    auto mesh =
        createCurvilinearMesh( mapping, halo_width, comm, ranks_per_dim );

    // Create field manager.
    auto manager = createFieldManager( mesh );
    return manager;
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_UNIFORMCARTESIANMESHMAPPING_HPP
