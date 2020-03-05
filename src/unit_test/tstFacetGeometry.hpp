#include <Harlow_FacetGeometry.hpp>
#include <Harlow_Types.hpp>

#include <Harlow_ParticleInit.hpp>
#include <Harlow_SiloParticleWriter.hpp>

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
void initExample()
{
    std::array<int,3> global_num_cell = { 24, 24, 24 };
    std::array<double,3> global_low_corner = { -12.0, -12.0, -12.0 };
    std::array<double,3> global_high_corner = { 12.0, 12.0, 12.0 };
    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner,
                                                        global_high_corner,
                                                        global_num_cell );
    std::array<bool,3> periodic = {false,false,false};
    auto global_grid =
        Cajita::createGlobalGrid( MPI_COMM_WORLD,
                                  global_mesh,
                                  periodic,
                                  Cajita::UniformDimPartitioner() );
    auto local_grid = Cajita::createLocalGrid( global_grid, 0 );

    using particle_fields = Cabana::MemberTypes<double[3],int>;
    using particle_list = Cabana::AoSoA<particle_fields,TEST_DEVICE>;
    using particle_type = typename particle_list::tuple_type;
    particle_list particles( "particles" );

    FacetGeometry<TEST_MEMSPACE> geometry( "stl_reader_test.stl" );
    auto facets = geometry.volumeFacets( 0 );
    auto init_func =
        KOKKOS_LAMBDA( const double x[3], particle_type& p ){
        for ( int d = 0; d < 3; ++d )
            Cabana::get<0>(p,d) = x[d];
        Cabana::get<1>(p) = 0;
        bool inside = true;
        for ( std::size_t f = 0; f < facets.extent(0); ++f )
        {
            float xf[3] = {float(x[0]),float(x[1]),float(x[2])};
            if ( FacetGeometryOps::distanceToFacetPlane(xf,facets,f) > 0.0 )
            {
                inside = false;
                break;
            }
        }
        return inside;
    };
    initializeParticles( InitUniform(), *local_grid, 3, init_func, particles );

    SiloParticleWriter::writeTimeStep(
        *global_grid, 0, 0.0,
        Cabana::slice<0>(particles,"x"), Cabana::slice<1>(particles,"i") );
}

TEST( TEST_CATEGORY, init_example )
{
    initExample();
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
