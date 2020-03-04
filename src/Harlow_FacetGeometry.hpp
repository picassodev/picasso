#ifndef HARLOW_FACETGEOMETRY_HPP
#define HARLOW_FACETGEOMETRY_HPP

#include <Kokkos_Core.hpp>

#include <fstream>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace Harlow
{
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct FacetGeometry
{
  public:

    // Create the geometry from an ASCII STL file.
    FacetGeometry( const std::string& stl_ascii_filename )
    {
        // Containers.
        std::vector<int> volume_ids;
        std::vector<int> surface_ids;
        std::vector<int> volume_facet_counts;
        std::vector<int> surface_facet_counts;
        std::vector<float> volume_facets;
        std::vector<float> surface_facets;
        std::vector<float> volume_normals;
        std::vector<float> surface_normals;

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

                    // Volume facet normal.
                    if ( read_volume )
                    {
                        ++volume_facet_counts.back();
                        volume_normals.push_back( std::atoi(tokens[2].c_str()) );
                        volume_normals.push_back( std::atoi(tokens[3].c_str()) );
                        volume_normals.push_back( std::atoi(tokens[4].c_str()) );
                    }

                    // Surface facet normal.
                    else if ( read_surface )
                    {
                        ++surface_facet_counts.back();
                        surface_normals.push_back( std::atoi(tokens[2].c_str()) );
                        surface_normals.push_back( std::atoi(tokens[3].c_str()) );
                        surface_normals.push_back( std::atoi(tokens[4].c_str()) );
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
                             volume_normals,
                             _volume_ids, _volume_facets, _volume_offsets );

        // Put surface data on device.
        putFileDataOnDevice( surface_ids, surface_facet_counts, surface_facets,
                             surface_normals,
                             _surface_ids, _surface_facets, _surface_offsets );
    }

    // Get the number of volumes.
    int numVolume() const
    {
        return _volume_ids.size();
    }

    // Given a global volume id get the local volume id.
    int localVolumeId( const int global_id ) const
    {
        return _volume_ids.find(global_id)->second;
    }

    // Given a local volume id get the facets associated with the volume.
    KOKKOS_INLINE_FUNCTION
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
    int numSurface() const
    {
        return _surface_ids.size();
    }

    // Given a global surface id get the local surface id.
    int localSurfaceId( const int global_id ) const
    {
        return _surface_ids.find(global_id)->second;
    }

    // Given a local surface id get the facets associated with the surface.
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<float*[4][3],MemorySpace>
    surfaceFacets( const int surface_id ) const
    {
        Kokkos::pair<int,int> facet_bounds(
            (0 == surface_id) ? 0 : _surface_offsets(surface_id-1),
            _surface_offsets(surface_id) );
        return Kokkos::subview(
            _surface_facets, facet_bounds, Kokkos::ALL(), Kokkos::ALL() );
    }

  public:

    // Put file data on device.
    void putFileDataOnDevice( const std::vector<int>& solid_ids,
                              const std::vector<int>& solid_facet_counts,
                              const std::vector<float>& solid_facets,
                              const std::vector<float>& solid_normals,
                              std::unordered_map<int,int>& id_map,
                              Kokkos::View<float*[4][3],MemorySpace>& device_facets,
                              Kokkos::View<int*,MemorySpace>& device_offsets )
    {
        // Allocate solid data.
        int num_solid = solid_ids.size();
        device_offsets = Kokkos::View<int*,MemorySpace>(
            Kokkos::ViewAllocateWithoutInitializing("offsets"),
            num_solid );

        int num_solid_facet = solid_normals.size() / 3;
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

            // Extract the normals.
            for ( int d = 0; d < 3; ++d )
                host_facets(f,3,d) = solid_normals[3*f+d];

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
};

//---------------------------------------------------------------------------//
// Facet Geometry operations.
//---------------------------------------------------------------------------//
namespace FacetGeometryOps
{
//---------------------------------------------------------------------------//
// Compute the signed distance from a point to the plane defined by a facet.
template<class FacetView>
KOKKOS_INLINE_FUNCTION
float
distanceToFacetPlane( const float x[3], const FacetView& facets, const int f )
{
    return
        (x[0]-facets(f,0,0))*facets(f,3,0) +
        (x[1]-facets(f,0,1))*facets(f,3,1) +
        (x[2]-facets(f,0,2))*facets(f,3,2);
}

//---------------------------------------------------------------------------//

} // end namespace FacetGeometryOps

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_FACETGEOMETRY_HPP
