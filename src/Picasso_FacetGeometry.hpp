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

#ifndef PICASSO_FACETGEOMETRY_HPP
#define PICASSO_FACETGEOMETRY_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <nlohmann/json.hpp>

#include <cfloat>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace Picasso
{
//---------------------------------------------------------------------------//
template <class MemorySpace>
struct FacetGeometryData
{
    // Get the number of volumes.
    KOKKOS_FUNCTION
    int numVolume() const { return volume_offsets.extent( 0 ); }

    // Given a local volume id get the facets associated with the volume.
    KOKKOS_FUNCTION
    Kokkos::View<float* [4][3], MemorySpace>
    volumeFacets( const int volume_id ) const {
        Kokkos::pair<int, int> facet_bounds(
            ( 0 == volume_id ) ? 0 : volume_offsets( volume_id - 1 ),
            volume_offsets( volume_id ) );
        return Kokkos::subview( volume_facets, facet_bounds, Kokkos::ALL(),
                                Kokkos::ALL() );
    }

    // Get the number of surfaces.
    KOKKOS_FUNCTION int numSurface() const
    {
        return surface_offsets.extent( 0 );
    }

    // Given a local surface id get the facets associated with the surface.
    KOKKOS_FUNCTION
    Kokkos::View<float* [4][3], MemorySpace>
    surfaceFacets( const int surface_id ) const {
        Kokkos::pair<int, int> facet_bounds(
            ( 0 == surface_id ) ? 0 : surface_offsets( surface_id - 1 ),
            surface_offsets( surface_id ) );
        return Kokkos::subview( surface_facets, facet_bounds, Kokkos::ALL(),
                                Kokkos::ALL() );
    }

    // Volume facets. Ordered as (facet,vector,dim) where vector=0,1,2 are the
    // vertices and vector=3 is the unit normal facing outward from the
    // volume.
    Kokkos::View<float* [4][3], MemorySpace> volume_facets;

    // Volume face offsets. Inclusive scan of volume counts giving the offset
    // into the facet array for each volume.
    Kokkos::View<int*, MemorySpace> volume_offsets;

    // Surface facets. Ordered as (facet,vector,dim) where vector=0,1,2 are the
    // vertices and vector=3 is the unit normal facing outward from the
    // surface.
    Kokkos::View<float* [4][3], MemorySpace> surface_facets;

    // Surface face offsets. Inclusive scan of surface counts giving the offset
    // into the facet array for each surface.
    Kokkos::View<int*, MemorySpace> surface_offsets;

    // Local id of the volume the representes the global axis-aligned bounding
    // box.
    int global_bounding_volume_id;

    // Axis aligned bounding box for all volumes.
    Kokkos::View<float* [6], MemorySpace> volume_bounding_boxes;
};

//---------------------------------------------------------------------------//
template <class MemorySpace>
class FacetGeometry
{
  public:
    // Compute the axis-aligned bounding box of a volume.
    struct BoundingBoxReduce
    {
        typedef float value_type[];
        typedef Kokkos::View<float* [4][3], MemorySpace> facet_view_type;
        typedef Kokkos::View<int*, MemorySpace> offset_view_type;
        typedef typename facet_view_type::size_type size_type;
        size_type value_count;

        facet_view_type facets;
        offset_view_type offsets;
        int volume_id;

        BoundingBoxReduce( const facet_view_type& f, const offset_view_type& o,
                           int v )
            : value_count( 6 )
            , facets( f )
            , offsets( o )
            , volume_id( v )
        {
        }

        KOKKOS_FUNCTION
        void operator()( const size_type facet_id, value_type result ) const
        {
            using Kokkos::fmax;
            using Kokkos::fmin;

            auto f_start = ( volume_id == 0 ) ? 0 : offsets( volume_id - 1 );
            auto f = f_start + facet_id;
            result[0] =
                fmin( result[0],
                      fmin( facets( f, 0, 0 ),
                            fmin( facets( f, 1, 0 ), facets( f, 2, 0 ) ) ) );
            result[1] =
                fmin( result[1],
                      fmin( facets( f, 0, 1 ),
                            fmin( facets( f, 1, 1 ), facets( f, 2, 1 ) ) ) );
            result[2] =
                fmin( result[2],
                      fmin( facets( f, 0, 2 ),
                            fmin( facets( f, 1, 2 ), facets( f, 2, 2 ) ) ) );
            result[3] =
                fmax( result[3],
                      fmax( facets( f, 0, 0 ),
                            fmax( facets( f, 1, 0 ), facets( f, 2, 0 ) ) ) );
            result[4] =
                fmax( result[4],
                      fmax( facets( f, 0, 1 ),
                            fmax( facets( f, 1, 1 ), facets( f, 2, 1 ) ) ) );
            result[5] =
                fmax( result[5],
                      fmax( facets( f, 0, 2 ),
                            fmax( facets( f, 1, 2 ), facets( f, 2, 2 ) ) ) );
        }

        KOKKOS_FUNCTION
        void join( value_type dst, const value_type src ) const
        {
            dst[0] = Kokkos::fmin( dst[0], src[0] );
            dst[1] = Kokkos::fmin( dst[1], src[1] );
            dst[2] = Kokkos::fmin( dst[2], src[2] );
            dst[3] = Kokkos::fmax( dst[3], src[3] );
            dst[4] = Kokkos::fmax( dst[4], src[4] );
            dst[5] = Kokkos::fmax( dst[5], src[5] );
        }

        KOKKOS_FUNCTION
        void init( value_type v ) const
        {
            v[0] = FLT_MAX;
            v[1] = FLT_MAX;
            v[2] = FLT_MAX;
            v[3] = -FLT_MAX;
            v[4] = -FLT_MAX;
            v[5] = -FLT_MAX;
        }
    };

  public:
    using memory_space = MemorySpace;

    // Default constructor.
    FacetGeometry() = default;

    // Create the geometry from an ASCII STL file.
    template <class ExecutionSpace>
    FacetGeometry( const nlohmann::json inputs,
                   const ExecutionSpace& exec_space )
    {
        Kokkos::Profiling::pushRegion( "Picasso::FacetGeoemtry::create" );

        // Get the geometry parameters.
        auto params = inputs["geometry"];

        // Read the stl file and create the facet geometry.
        std::string stl_ascii_filename = params["stl_file"];

        // Containers.
        std::vector<int> volume_ids;
        std::vector<int> surface_ids;
        std::vector<float> volume_facets;
        std::vector<float> surface_facets;

        // Load the file.
        std::ifstream file( stl_ascii_filename );
        if ( !file.is_open() )
            throw std::runtime_error( "Unable to open STL ASCII file" );

        // Read the file.
        std::string buffer;
        std::vector<std::string> tokens;
        tokens.reserve( 10 );
        bool read_volume = false;
        bool read_surface = false;
        while ( !file.eof() )
        {
            // Get the current line.
            std::getline( file, buffer );

            // Break the line up into tokens.
            std::istringstream line( buffer );
            tokens.resize( 0 );
            while ( !line.eof() )
            {
                tokens.push_back( std::string() );
                line >> tokens.back();
            }

            // Parse the line.
            if ( tokens.size() > 0 )
            {
                // New solid.
                if ( tokens[0].compare( "solid" ) == 0 )
                {
                    if ( tokens.size() != 3 )
                        throw std::runtime_error(
                            "STL READER: Expected 3 solid line entries" );

                    // New volume.
                    if ( tokens[1].compare( "Volume" ) == 0 ||
                         tokens[1].compare( "Body" ) == 0 )
                    {
                        volume_ids.push_back( std::atoi( tokens[2].c_str() ) );
                        _volume_facet_count.push_back( 0 );
                        read_volume = true;
                    }

                    // New surface.
                    else if ( tokens[1].compare( "Surface" ) == 0 )
                    {
                        surface_ids.push_back( std::atoi( tokens[2].c_str() ) );
                        _surface_facet_count.push_back( 0 );
                        read_surface = true;
                    }

                    else
                    {
                        throw std::runtime_error(
                            "STL READER: Solids execpted to be Volume/Body or "
                            "Surface" );
                    }
                }

                // Read facet.
                else if ( tokens[0].compare( "facet" ) == 0 )
                {
                    if ( tokens.size() != 5 )
                        throw std::runtime_error(
                            "STL READER: Expected 5 facet line entries" );

                    // Volume facet.
                    if ( read_volume )
                    {
                        ++_volume_facet_count.back();
                    }

                    // Surface facet.
                    else if ( read_surface )
                    {
                        ++_surface_facet_count.back();
                    }
                }

                // Read vertex.
                else if ( tokens[0].compare( "vertex" ) == 0 )
                {
                    if ( tokens.size() != 4 )
                        throw std::runtime_error(
                            "STL READER: Expected 4 vertex line entries" );

                    // Volume vertex coordinates.
                    if ( read_volume )
                    {
                        volume_facets.push_back(
                            std::atof( tokens[1].c_str() ) );
                        volume_facets.push_back(
                            std::atof( tokens[2].c_str() ) );
                        volume_facets.push_back(
                            std::atof( tokens[3].c_str() ) );
                    }

                    // Surface vertex coordinates.
                    if ( read_surface )
                    {
                        surface_facets.push_back(
                            std::atof( tokens[1].c_str() ) );
                        surface_facets.push_back(
                            std::atof( tokens[2].c_str() ) );
                        surface_facets.push_back(
                            std::atof( tokens[3].c_str() ) );
                    }
                }

                // Finish reading a solid.
                else if ( tokens[0].compare( "endsolid" ) == 0 )
                {
                    read_volume = false;
                    read_surface = false;
                }
            }
        }

        // Close the file.
        file.close();

        // Put volume data on device.
        putFileDataOnDevice( volume_ids, _volume_facet_count, volume_facets,
                             _volume_ids, _data.volume_facets,
                             _data.volume_offsets );

        // Put surface data on device.
        putFileDataOnDevice( surface_ids, _surface_facet_count, surface_facets,
                             _surface_ids, _data.surface_facets,
                             _data.surface_offsets );

        // Get the volume id of the global bounding box. The user is required
        // to make an axis-aligned bounding box of their geometry that defines
        // the global bounds of the problem. The user input is the global id
        // of this volume.
        _data.global_bounding_volume_id =
            localVolumeId( params["global_bounding_volume_id"] );

        // Compute the bounding boxes of all the volumes.
        _data.volume_bounding_boxes = Kokkos::View<float* [6], MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing( "volume_bounding_boxes" ),
            volume_ids.size() );
        auto host_boxes = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), _data.volume_bounding_boxes );
        for ( int v = 0; v < _data.numVolume(); ++v )
        {
            BoundingBoxReduce reducer( _data.volume_facets,
                                       _data.volume_offsets, v );
            float box[6];
            Kokkos::parallel_reduce(
                "Picasso::FacetGeometry::VolumeBoundingBox",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                     _volume_facet_count[v] ),
                reducer, box );
            for ( int i = 0; i < 6; ++i )
                host_boxes( v, i ) = box[i];
        }
        Kokkos::deep_copy( _data.volume_bounding_boxes, host_boxes );

        // Extract the global bounding box to the host.
        auto global_box = Kokkos::subview(
            host_boxes, _data.global_bounding_volume_id, Kokkos::ALL() );
        for ( int i = 0; i < 6; ++i )
            _global_bounding_box[i] = global_box( i );

        Kokkos::Profiling::popRegion();
    }

    // Given a global volume id get the local volume id.
    int localVolumeId( const int global_id ) const
    {
        return _volume_ids.find( global_id )->second;
    }

    // Given a global surface id get the local surface id.
    int localSurfaceId( const int global_id ) const
    {
        return _surface_ids.find( global_id )->second;
    }

    // Given a local volume id get the number of facets that compose the
    // volume.
    int numVolumeFacet( const int local_id ) const
    {
        return _volume_facet_count[local_id];
    }

    // Given a local surface id get the number of facets that compose the
    // surface.
    int numSurfaceFacet( const int local_id ) const
    {
        return _surface_facet_count[local_id];
    }

    // Get the global bounding box.
    const Kokkos::Array<double, 6>& globalBoundingBox() const
    {
        return _global_bounding_box;
    }

    // Get the geometry data.
    const FacetGeometryData<MemorySpace>& data() const { return _data; }

  private:
    // Put file data on device.
    void putFileDataOnDevice(
        const std::vector<int>& solid_ids,
        const std::vector<int>& solid_facet_counts,
        const std::vector<float>& solid_facets,
        std::unordered_map<int, int>& id_map,
        Kokkos::View<float* [4][3], MemorySpace>& device_facets,
        Kokkos::View<int*, MemorySpace>& device_offsets )
    {
        // Allocate solid data.
        int num_solid = solid_ids.size();
        device_offsets = Kokkos::View<int*, MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing( "offsets" ), num_solid );

        int num_solid_facet = solid_facets.size() / 9;
        device_facets = Kokkos::View<float* [4][3], MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing( "facets" ),
            num_solid_facet );

        // Compute the offsets for the facet-to-solid mapping.
        auto host_offsets =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), device_offsets );
        for ( int i = 0; i < num_solid; ++i )
        {
            // Map global solid ids to local ids.
            id_map.emplace( solid_ids[i], i );

            // Compute the offset via inclusive scan.
            host_offsets( i ) =
                ( 0 == i ) ? solid_facet_counts[i]
                           : solid_facet_counts[i] + host_offsets( i - 1 );
        }

        // Build the facets.
        auto host_facets =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), device_facets );
        for ( int f = 0; f < num_solid_facet; ++f )
        {
            // Extract the vertices.
            for ( int v = 0; v < 3; ++v )
                for ( int d = 0; d < 3; ++d )
                    host_facets( f, v, d ) = solid_facets[9 * f + 3 * v + d];

            // Compute the normals.
            float v10[3];
            float v20[3];
            for ( int d = 0; d < 3; ++d )
            {
                v10[d] = host_facets( f, 1, d ) - host_facets( f, 0, d );
                v20[d] = host_facets( f, 2, d ) - host_facets( f, 0, d );
            }
            host_facets( f, 3, 0 ) = v10[1] * v20[2] - v10[2] * v20[1];
            host_facets( f, 3, 1 ) = v10[2] * v20[0] - v10[0] * v20[2];
            host_facets( f, 3, 2 ) = v10[0] * v20[1] - v10[1] * v20[0];

            // Scale them to make it a unit normal.
            float nmag =
                std::sqrt( host_facets( f, 3, 0 ) * host_facets( f, 3, 0 ) +
                           host_facets( f, 3, 1 ) * host_facets( f, 3, 1 ) +
                           host_facets( f, 3, 2 ) * host_facets( f, 3, 2 ) );
            for ( int d = 0; d < 3; ++d )
                host_facets( f, 3, d ) /= nmag;
        }

        // Copy to device.
        Kokkos::deep_copy( device_facets, host_facets );
        Kokkos::deep_copy( device_offsets, host_offsets );
    }

  public:
    // Volume ids - global-to-local mapping.
    std::unordered_map<int, int> _volume_ids;

    // Surface ids - global-to-local mapping.
    std::unordered_map<int, int> _surface_ids;

    // Volume facet counts.
    std::vector<int> _volume_facet_count;

    // Surface facet counts.
    std::vector<int> _surface_facet_count;

    // Global bounding box.
    Kokkos::Array<double, 6> _global_bounding_box;

    // Data.
    FacetGeometryData<MemorySpace> _data;
};

//---------------------------------------------------------------------------//
// Facet Geometry operations.
//---------------------------------------------------------------------------//
namespace FacetGeometryOps
{
//---------------------------------------------------------------------------//
// Project a point x along the given direction, r, and determine if it
// projects to the facet. If returns true, the projection solution is
// valid.
//
// y[0] = projection barycentric coordinate 1
// y[1] = projection barycentric coordinate 2
// y[2] = distance to triangle
template <class FacetView>
KOKKOS_FUNCTION bool pointFacetProjection( const float x[3], const float r[3],
                                           const FacetView& facets, const int f,
                                           float y[3] )
{
    // Build the system of equations to solve for intersection. Fire the ray
    // in the Y direction - this choice is arbitary.
    Mat3<float> A;
    Vec3<float> b;
    for ( int i = 0; i < 3; ++i )
    {
        A( i, 0 ) = facets( f, 1, i ) - facets( f, 0, i );
        A( i, 1 ) = facets( f, 2, i ) - facets( f, 0, i );
        A( i, 2 ) = -r[i];
        b( i ) = x[i] - facets( f, 0, i );
    }

    // Check the determinant of the matrix. If zero then the ray is parallel
    // so no intersection.
    auto det_A = !A;
    if ( 0.0 == det_A )
        return false;

    // Solve the system.
    VecView3<float> y_view( y, 1 );
    y_view = A ^ b;

    // Check the solution for inclusion in the triangle.
    return ( y[0] >= 0.0 && y[1] >= 0.0 && y[0] + y[1] <= 1.0 );
}

//---------------------------------------------------------------------------//
// Fire a ray from point x along the given direction, r, and determine if it
// intersects the facet.
template <class FacetView>
KOKKOS_FUNCTION bool rayFacetIntersect( const float x[3], const float r[3],
                                        const FacetView& facets, const int f )
{
    // Project the point and check the distance.
    float y[3];
    auto projects = pointFacetProjection( x, r, facets, f, y );
    return projects && ( y[2] > 0.0 );
}

//---------------------------------------------------------------------------//
// Compute the signed distance from a point to the plane defined by a facet.
template <class FacetView>
KOKKOS_FUNCTION float
distanceToFacetPlane( const float x[3], const FacetView& facets, const int f )
{
    return ( x[0] - facets( f, 0, 0 ) ) * facets( f, 3, 0 ) +
           ( x[1] - facets( f, 0, 1 ) ) * facets( f, 3, 1 ) +
           ( x[2] - facets( f, 0, 2 ) ) * facets( f, 3, 2 );
}

//---------------------------------------------------------------------------//
// Determine if a point is in a volume represented by a view of facets.
template <class FacetView>
KOKKOS_FUNCTION bool pointInVolume( const float x[3],
                                    const FacetView& volume_facets )
{
    // The choice of ray direction is arbitrary so generate a random one. This
    // could potentially help with robustness as floating point noise may
    // avoid edge and vertex intersections which could lead to multiple
    // positive intersections and therefore an incorrect point-in-volume
    // determination. The facet geometry has no connectivity so it is
    // difficult to resolve multiple intersections without accumulating the
    // intersection points and checking for duplicates within some tolerance.
    //
    // Note: Duan has indicated that doing tests with 3 different random rays
    // has been enough to be robust as one is likely to pass out of the 3 if
    // is a true intersection.
    using rand_type =
        Kokkos::Random_XorShift64<typename FacetView::device_type>;
    rand_type rng( 0 );
    float r[3] = { Kokkos::rand<rand_type, float>::draw( rng ),
                   Kokkos::rand<rand_type, float>::draw( rng ),
                   Kokkos::rand<rand_type, float>::draw( rng ) };
    float r_mag_inv = 1.0 / sqrt( r[0] * r[0] + r[1] * r[1] + r[2] * r[2] );
    for ( int d = 0; d < 3; ++d )
        r[d] *= r_mag_inv;

    // Fire rays through each facet and count intersections. If an
    // odd number of intersections, the point is in the volume. This works for
    // convex and non-convex volumes.
    int count = 0;
    for ( std::size_t f = 0; f < volume_facets.extent( 0 ); ++f )
        if ( rayFacetIntersect( x, r, volume_facets, f ) )
            ++count;
    return ( 1 == count % 2 );
}

//---------------------------------------------------------------------------//
// Given a point determine the volume in the given facet geometry in which it
// is located.If it is in the implicit complement, return -1. If it is outside
// of the entire domain, return -2;
template <class MemorySpace>
KOKKOS_FUNCTION int locatePoint( const float x[3],
                                 const FacetGeometryData<MemorySpace>& geom )
{
    // Get the global bounding volume id.
    int gbv = geom.global_bounding_volume_id;

    // Start by checking that the point is in the global bounding volume.
    if ( geom.volume_bounding_boxes( gbv, 0 ) <= x[0] &&
         geom.volume_bounding_boxes( gbv, 1 ) <= x[1] &&
         geom.volume_bounding_boxes( gbv, 2 ) <= x[2] &&
         geom.volume_bounding_boxes( gbv, 3 ) >= x[0] &&
         geom.volume_bounding_boxes( gbv, 4 ) >= x[1] &&
         geom.volume_bounding_boxes( gbv, 5 ) >= x[2] )
    {
        // Check each volume except the global bounding volume for point
        // inclusion.
        for ( int v = 0; v < geom.numVolume(); ++v )
        {
            if ( v != gbv )
            {
                // First check if the point is in the axis-aligned
                // bounding box of the volume.
                if ( geom.volume_bounding_boxes( v, 0 ) <= x[0] &&
                     geom.volume_bounding_boxes( v, 1 ) <= x[1] &&
                     geom.volume_bounding_boxes( v, 2 ) <= x[2] &&
                     geom.volume_bounding_boxes( v, 3 ) >= x[0] &&
                     geom.volume_bounding_boxes( v, 4 ) >= x[1] &&
                     geom.volume_bounding_boxes( v, 5 ) >= x[2] )
                {
                    // If in the bounding box, check against each volume
                    // facet for point inclusion.
                    if ( pointInVolume( x, geom.volumeFacets( v ) ) )
                    {
                        return v;
                    }
                }
            }
        }

        // If the point was not in any volume it is in the implicit
        // complement so return -1.
        return -1;
    }

    // Otherwise point is outside of the global domain including the
    // implicit complement so return -2.
    return -2;
}

//---------------------------------------------------------------------------//

} // end namespace FacetGeometryOps

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_FACETGEOMETRY_HPP
