#include <Harlow_FacetGeometry.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void constructionTest()
{
    // Create the geometry from the test file. It contains a sphere of radius
    // 10 centered at the origin. The sphere is saved as both a surface and
    // volume.
    FacetGeometry<TEST_MEMSPACE> geometry( "stl_reader_test.stl" );

    // Check that we got the right number of volumes and surfaces.
    EXPECT_EQ( geometry.numVolume(), 1 );
    EXPECT_EQ( geometry.numSurface(), 1 );

    // Check the global-to-local id conversion.
    EXPECT_EQ( geometry.localVolumeId(1), 0 );
    EXPECT_EQ( geometry.localSurfaceId(1), 0 );

    // Get the facets for the volume.
    auto volume_facets = geometry.volumeFacets( 0 );
    auto num_volume_facet = volume_facets.extent(0);
    EXPECT_TRUE( num_volume_facet > 0 );

    // Check that a point is in the volume. All normals point outward if a
    // point is in the sphere all distances should be negative.
    Kokkos::Array<float,3> point_in = {-1.2, 0.4, 1.9 };
    int volume_inside = 0;
    Kokkos::parallel_reduce(
        "check_inside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_volume_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_in.data(),
                                                       volume_facets, f);
            if ( dist < 0.0 )
                ++result;
        },
        volume_inside );
    EXPECT_EQ( volume_inside, num_volume_facet );

    // Check that a point is outside the volume. Some distances will be
    // negative, some will be positive. Check that some positive distances
    // were computed indicating an outside point.
    Kokkos::Array<float,3> point_out = {12.1, 1.4, -0.9 };
    int volume_outside = 0;
    Kokkos::parallel_reduce(
        "check_outside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_volume_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_out.data(),
                                                       volume_facets, f);
            if ( dist > 0.0 )
                ++result;
        },
        volume_outside );
    EXPECT_TRUE( volume_outside > 0 );

    // Get the facets for the surface.
    auto surface_facets = geometry.surfaceFacets( 0 );
    auto num_surface_facet = surface_facets.extent(0);
    EXPECT_TRUE( num_surface_facet > 0 );

    // Check that a point is in the surface. All normals point outward if a
    // point is in the sphere all distances should be negative.
    int surface_inside = 0;
    Kokkos::parallel_reduce(
        "check_inside_surface",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_surface_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_in.data(),
                                                       surface_facets, f);
            if ( dist < 0.0 )
                ++result;
        },
        surface_inside );
    EXPECT_EQ( surface_inside, num_surface_facet );

    // Check that a point is outside the surface. Some distances will be
    // negative, some will be positive. Check that some positive distances
    // were computed indicating an outside point.
    int surface_outside = 0;
    Kokkos::parallel_reduce(
        "check_outside_surface",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_surface_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_out.data(),
                                                       surface_facets, f);
            if ( dist > 0.0 )
                ++result;
        },
        surface_outside );
    EXPECT_TRUE( surface_outside > 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test )
{
    constructionTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
