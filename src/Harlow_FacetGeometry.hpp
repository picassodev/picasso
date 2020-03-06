#ifndef HARLOW_FACETGEOMETRY_HPP
#define HARLOW_FACETGEOMETRY_HPP

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <fstream>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <limits>

namespace Harlow
{
//---------------------------------------------------------------------------//
template<class MemorySpace>
class FacetGeometry
{
  public:

    // Compute the axis-aligned bounding box of a volume.
    struct BoundingBoxReduce
    {
        typedef float value_type[];
        typedef Kokkos::View<float*[4][3]> view_type;
        typedef typename view_type::size_type size_type;
        size_type value_count;

        view_type facets;

        BoundingBoxReduce( const view_type& f )
            : value_count(6)
            , facets(f)
        {}

        KOKKOS_INLINE_FUNCTION
        void operator()( const size_type f, value_type result ) const
        {
            result[0] = fmin( result[0],
                              fmin( facets(f,0,0),
                                    fmin( facets(f,1,0), facets(f,2,0) ) ) );
            result[1] = fmin( result[1],
                              fmin( facets(f,0,1),
                                    fmin( facets(f,1,1), facets(f,2,1) ) ) );
            result[2] = fmin( result[2],
                              fmin( facets(f,0,2),
                                    fmin( facets(f,1,2), facets(f,2,2) ) ) );
            result[3] = fmax( result[3],
                              fmax( facets(f,0,0),
                                    fmax( facets(f,1,0), facets(f,2,0) ) ) );
            result[4] = fmax( result[4],
                              fmax( facets(f,0,1),
                                    fmax( facets(f,1,1), facets(f,2,1) ) ) );
            result[5] = fmax( result[5],
                              fmax( facets(f,0,2),
                                    fmax( facets(f,1,2), facets(f,2,2) ) ) );
        }

        KOKKOS_INLINE_FUNCTION
        void join( volatile value_type dst, const volatile value_type src ) const
        {
            dst[0] = fmin( dst[0], src[0] );
            dst[1] = fmin( dst[1], src[1] );
            dst[2] = fmin( dst[2], src[2] );
            dst[3] = fmax( dst[3], src[3] );
            dst[4] = fmax( dst[4], src[4] );
            dst[5] = fmax( dst[5], src[5] );
        }

        KOKKOS_INLINE_FUNCTION
        void init( value_type v ) const
        {
            v[0] = std::numeric_limits<float>::max();
            v[1] = std::numeric_limits<float>::max();
            v[2] = std::numeric_limits<float>::max();
            v[3] = -std::numeric_limits<float>::max();
            v[4] = -std::numeric_limits<float>::max();
            v[5] = -std::numeric_limits<float>::max();
        }
    };

  public:

    using memory_space = MemorySpace;

    // Default constructor.
    FacetGeometry() = default;

    // Create the geometry from an ASCII STL file.
    template<class ExecutionSpace>
    FacetGeometry( const boost::property_tree::ptree& ptree,
                   const ExecutionSpace& exec_space )
    {

        // Read the stl file and create the facet geometry.
        auto stl_ascii_filename =
            ptree.get<std::string>( "geometry.stl_file" );

        // Containers.
        std::vector<int> volume_ids;
        std::vector<int> surface_ids;
        std::vector<int> volume_facet_counts;
        std::vector<int> surface_facet_counts;
        std::vector<float> volume_facets;
        std::vector<float> surface_facets;

        // Load the file.
        std::ifstream file( stl_ascii_filename );
        if ( !file.is_open() )
            throw std::runtime_error("Unable to open STL ASCII file");

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
            tokens.resize(0);
            while ( !line.eof() )
            {
                tokens.push_back( std::string() );
                line >> tokens.back();
            }

            // Parse the line.
            if ( tokens.size() > 0 )
            {
                // New solid.
                if ( tokens[0].compare("solid") == 0 )
                {
                    if ( tokens.size() != 3 )
                        throw std::runtime_error(
                            "STL READER: Expected 3 solid line entries" );

                    // New volume.
                    if ( tokens[1].compare("Volume") == 0 )
                    {
                        volume_ids.push_back( std::atoi(tokens[2].c_str()) );
                        volume_facet_counts.push_back( 0 );
                        read_volume = true;
                    }

                    // New surface.
                    else if ( tokens[1].compare("Surface") == 0 )
                    {
                        surface_ids.push_back( std::atoi(tokens[2].c_str()) );
                        surface_facet_counts.push_back( 0 );
                        read_surface = true;
                    }

                    else
                    {
                        throw std::runtime_error(
                            "STL READER: Solids execpted to be Volume or Surface" );
                    }
                }

                // Read facet.
                else if ( tokens[0].compare("facet") == 0 )
                {
                    if ( tokens.size() != 5 )
                        throw std::runtime_error(
                            "STL READER: Expected 5 facet line entries" );

                    // Volume facet.
                    if ( read_volume )
                    {
                        ++volume_facet_counts.back();
                    }

                    // Surface facet.
                    else if ( read_surface )
                    {
                        ++surface_facet_counts.back();
                    }
                }

                // Read vertex.
                else if ( tokens[0].compare("vertex") == 0 )
                {
                    if ( tokens.size() != 4 )
                        throw std::runtime_error(
                            "STL READER: Expected 4 vertex line entries" );

                    // Volume vertex coordinates.
                    if ( read_volume )
                    {
                        volume_facets.push_back( std::atof(tokens[1].c_str()) );
                        volume_facets.push_back( std::atof(tokens[2].c_str()) );
                        volume_facets.push_back( std::atof(tokens[3].c_str()) );
                    }

                    // Surface vertex coordinates.
                    if ( read_surface )
                    {
                        surface_facets.push_back( std::atof(tokens[1].c_str()) );
                        surface_facets.push_back( std::atof(tokens[2].c_str()) );
                        surface_facets.push_back( std::atof(tokens[3].c_str()) );
                    }
                }

                // Finish reading a solid.
                else if ( tokens[0].compare("endsolid") == 0 )
                {
                    read_volume = false;
                    read_surface = false;
                }
            }
        }

        // Close the file.
        file.close();

        // Put volume data on device.
        putFileDataOnDevice( volume_ids, volume_facet_counts, volume_facets,
                             _volume_ids, _volume_facets, _volume_offsets );

        // Put surface data on device.
        putFileDataOnDevice( surface_ids, surface_facet_counts, surface_facets,
                             _surface_ids, _surface_facets, _surface_offsets );

        // Get the volume id of the global bounding box. The user is required
        // to make an axis-aligned bounding box of their geometry that defines
        // the global bounds of the problem. The user input is the global id
        // of this volume.
        _global_bounding_volume_id = localVolumeId(
            ptree.get<int>("geometry.global_bounding_volume_id") );

        // Compute the bounding boxes of all the volumes.
        _volume_bounding_boxes = Kokkos::View<float*[6],MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing("volume_bounding_boxes"),
            _volume_ids.size() );
        auto host_boxes = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), _volume_bounding_boxes );
        for ( int v = 0; v < numVolume(); ++v )
        {
            auto facets = volumeFacets(v);
            BoundingBoxReduce reducer( facets );
            float box[6];
            Kokkos::parallel_reduce(
                "volume_bounding_box",
                Kokkos::RangePolicy<ExecutionSpace>(
                    exec_space,0,facets.extent(0)),
                reducer,
                box );
            for ( int i = 0; i < 6; ++i )
                host_boxes(v,i) = box[i];
        }
        Kokkos::deep_copy( _volume_bounding_boxes, host_boxes );

        // Extract the global bounding box to the host.
        auto global_box = Kokkos::subview(
            host_boxes, _global_bounding_volume_id, Kokkos::ALL() );
        for ( int i = 0; i < 6; ++i )
            _global_bounding_box[i] = global_box(i);
    }

    // Get the number of volumes.
    KOKKOS_FUNCTION
    int numVolume() const
    {
        return _volume_offsets.extent(0);
    }

    // Given a global volume id get the local volume id.
    int localVolumeId( const int global_id ) const
    {
        return _volume_ids.find(global_id)->second;
    }

    // Given a local volume id get the facets associated with the volume.
    KOKKOS_FUNCTION
    Kokkos::View<float*[4][3],MemorySpace>
    volumeFacets( const int volume_id ) const
    {
        Kokkos::pair<int,int> facet_bounds(
            (0 == volume_id) ? 0 : _volume_offsets(volume_id-1),
            _volume_offsets(volume_id) );
        return Kokkos::subview(
            _volume_facets, facet_bounds, Kokkos::ALL(), Kokkos::ALL() );
    }

    // Get the number of surfaces.
    KOKKOS_FUNCTION
    int numSurface() const
    {
        return _surface_offsets.extent(0);
    }

    // Given a global surface id get the local surface id.
    int localSurfaceId( const int global_id ) const
    {
        return _surface_ids.find(global_id)->second;
    }

    // Given a local surface id get the facets associated with the surface.
    KOKKOS_FUNCTION
    Kokkos::View<float*[4][3],MemorySpace>
    surfaceFacets( const int surface_id ) const
    {
        Kokkos::pair<int,int> facet_bounds(
            (0 == surface_id) ? 0 : _surface_offsets(surface_id-1),
            _surface_offsets(surface_id) );
        return Kokkos::subview(
            _surface_facets, facet_bounds, Kokkos::ALL(), Kokkos::ALL() );
    }

    // Get the local id of the global bounding volume.
    KOKKOS_FUNCTION
    int globalBoundingVolumeId() const
    {
        return _global_bounding_volume_id;
    }

    // Get the global bounding box of the domain.
    KOKKOS_FUNCTION
    const Kokkos::Array<float,6>& globalBoundingBox() const
    {
        return _global_bounding_box;
    }

    // Get the bounding boxes of all the volumes.
    KOKKOS_FUNCTION
    Kokkos::View<float*[6],MemorySpace> volumeBoundingBoxes() const
    {
        return _volume_bounding_boxes;
    }

  public:

    // Put file data on device.
    void putFileDataOnDevice(
        const std::vector<int>& solid_ids,
        const std::vector<int>& solid_facet_counts,
        const std::vector<float>& solid_facets,
        std::unordered_map<int,int>& id_map,
        Kokkos::View<float*[4][3],MemorySpace>& device_facets,
        Kokkos::View<int*,MemorySpace>& device_offsets )
    {
        // Allocate solid data.
        int num_solid = solid_ids.size();
        device_offsets = Kokkos::View<int*,MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing("offsets"),
            num_solid );

        int num_solid_facet = solid_facets.size() / 9;
        device_facets = Kokkos::View<float*[4][3],MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing("facets"),
            num_solid_facet );

        // Populate the geometry data.
        auto host_offsets = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), device_offsets );
        for ( int i = 0; i < num_solid; ++i )
        {
            // Map global solid ids to local ids.
            id_map.emplace( solid_ids[i], i );

            // Compute the offset via inclusive scan.
            host_offsets(i) =
                (0 == i) ? solid_facet_counts[i]
                : solid_facet_counts[i] + host_offsets(i-1);
        }
        auto host_facets = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), device_facets );
        for ( int f = 0; f < num_solid_facet; ++f )
        {
            // Extract the vertices.
            for ( int v = 0; v < 3; ++v )
                for ( int d = 0; d < 3; ++d )
                    host_facets(f,v,d) = solid_facets[9*f+3*v+d];

            // Compute the normals.
            float v10[3];
            float v20[3];
            for ( int d = 0; d < 3; ++d )
            {
                v10[d] = host_facets(f,1,d)-host_facets(f,0,d);
                v20[d] = host_facets(f,2,d)-host_facets(f,0,d);
            }
            host_facets(f,3,0) = v10[1]*v20[2]-v10[2]*v20[1];
            host_facets(f,3,1) = v10[2]*v20[0]-v10[0]*v20[2];
            host_facets(f,3,2) = v10[0]*v20[1]-v10[1]*v20[0];

            // Scale them to make it a unit normal.
            float nmag = std::sqrt(
                host_facets(f,3,0)*host_facets(f,3,0) +
                host_facets(f,3,1)*host_facets(f,3,1) +
                host_facets(f,3,2)*host_facets(f,3,2) );
            for ( int d = 0; d < 3; ++d )
                host_facets(f,3,d) /= nmag;
        }

        // Copy to device.
        Kokkos::deep_copy( device_facets, host_facets );
        Kokkos::deep_copy( device_offsets, host_offsets );
    }

  public:

    // Volume ids - global-to-local mapping.
    std::unordered_map<int,int> _volume_ids;

    // Volume facets. Ordered as (facet,vector,dim) where vector=0,1,2 are the
    // vertices and vector=3 is the unit normal facing outward from the
    // volume.
    Kokkos::View<float*[4][3],MemorySpace> _volume_facets;

    // Volume face offsets. Inclusive scan of volume counts giving the offset
    // into the facet array for each volume.
    Kokkos::View<int*,MemorySpace> _volume_offsets;

    // Surface ids - global-to-local mapping.
    std::unordered_map<int,int> _surface_ids;

    // Surface facets. Ordered as (facet,vector,dim) where vector=0,1,2 are the
    // vertices and vector=3 is the unit normal facing outward from the
    // surface.
    Kokkos::View<float*[4][3],MemorySpace> _surface_facets;

    // Surface face offsets. Inclusive scan of surface counts giving the offset
    // into the facet array for each surface.
    Kokkos::View<int*,MemorySpace> _surface_offsets;

    // Local id of the volume the representes the global axis-aligned bounding
    // box.
    int _global_bounding_volume_id;

    // The global axis-aligned bounding box.
    Kokkos::Array<float,6> _global_bounding_box;

    // Axis aligned bounding box for all volumes.
    Kokkos::View<float*[6],MemorySpace> _volume_bounding_boxes;
};

//---------------------------------------------------------------------------//
// Facet Geometry operations.
//---------------------------------------------------------------------------//
namespace FacetGeometryOps
{
//---------------------------------------------------------------------------//
// Compute the signed distance from a point to the plane defined by a facet.
template<class FacetView>
KOKKOS_FUNCTION
float
distanceToFacetPlane( const float x[3], const FacetView& facets, const int f )
{
    return
        (x[0]-facets(f,0,0))*facets(f,3,0) +
        (x[1]-facets(f,0,1))*facets(f,3,1) +
        (x[2]-facets(f,0,2))*facets(f,3,2);
}

//---------------------------------------------------------------------------//
// Determine if a point is in a volume represented by a view of facets.
template<class FacetView>
KOKKOS_FUNCTION
bool pointInVolume( const float x[3], const FacetView& volume_facets )
{
    for ( std::size_t f = 0; f < volume_facets.extent(0); ++f )
    {
        if ( distanceToFacetPlane(x,volume_facets,f) > 0.0 )
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
// Given a point determine the volume in the given facet geometry in which it
// is located. We require the input point to be inside of the bounding box
// volume. If it is in the implicit complement, return -1. If it is outside of
// the entire domain, return -2;
template<class MemorySpace>
KOKKOS_FUNCTION
int locatePoint( const float x[3], const FacetGeometry<MemorySpace>& geometry )
{
    auto boxes = geometry.volumeBoundingBoxes();
    auto global_bounding_volume_id = geometry.globalBoundingVolumeId();

    // Start by checking that the point is in the global bounding volume.
    if ( boxes(global_bounding_volume_id,0) <= x[0] &&
         boxes(global_bounding_volume_id,1) <= x[1] &&
         boxes(global_bounding_volume_id,2) <= x[2] &&
         boxes(global_bounding_volume_id,3) >= x[0] &&
         boxes(global_bounding_volume_id,4) >= x[1] &&
         boxes(global_bounding_volume_id,5) >= x[2] )
    {
        // Check each volume except the global bounding volume for point
        // inclusion.
        for ( int g = 0; g < geometry.numVolume(); ++g )
        {
            if ( g != global_bounding_volume_id )
            {
                // First check if the point is in the axis-aligned
                // bounding box of the volume.
                if ( boxes(g,0) <= x[0] &&
                     boxes(g,1) <= x[1] &&
                     boxes(g,2) <= x[2] &&
                     boxes(g,3) >= x[0] &&
                     boxes(g,4) >= x[1] &&
                     boxes(g,5) >= x[2] )
                {
                    // If in the bounding box, check against each volume
                    // facet for point inclusion.
                    if ( pointInVolume(x,geometry.volumeFacets(g)) )
                    {
                        return g;
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

} // end namespace Harlow

#endif // end HARLOW_FACETGEOMETRY_HPP
