#include <Harlow_FacetGeometry.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_InputParser.hpp>

#include <Harlow_ParticleInit.hpp>
#include <Harlow_SiloParticleWriter.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void rayFireTestXY()
{
    // Create a facet in the XY plane
    Kokkos::View<float*[4][3],Kokkos::HostSpace> facet( "facet", 1 );

    facet(0,0,0) = 0.0;
    facet(0,0,1) = 0.0;
    facet(0,0,2) = 0.0;

    facet(0,1,0) = 1.0;
    facet(0,1,1) = 0.0;
    facet(0,1,2) = 0.0;

    facet(0,2,0) = 0.0;
    facet(0,2,1) = 1.0;
    facet(0,2,2) = 0.0;

    // Rays will be perpendicular in z.
    float r[3] = { 0.0, 0.0, 1.0 };

    // Pick a point in the triangle first.
    float p[3] = { 0.25, 0.25, 0.0 };

    // Check the projection.
    float y[3];
    auto result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 0.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle.
    p[2] = -1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point above the triangle.
    p[2] = 1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( -1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle at the  origin.
    p[0] = 0.0;
    p[1] = 0.0;
    p[2] = -1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );
}

//---------------------------------------------------------------------------//
void rayFireTestYZ()
{
    // Create a facet in the YZ plane
    Kokkos::View<float*[4][3],Kokkos::HostSpace> facet( "facet", 1 );

    facet(0,0,0) = 0.0;
    facet(0,0,1) = 0.0;
    facet(0,0,2) = 0.0;

    facet(0,1,0) = 0.0;
    facet(0,1,1) = 1.0;
    facet(0,1,2) = 0.0;

    facet(0,2,0) = 0.0;
    facet(0,2,1) = 0.0;
    facet(0,2,2) = 1.0;

    // Rays will be perpendicular in x.
    float r[3] = { 1.0, 0.0, 0.0 };

    // Pick a point in the triangle first.
    float p[3] = { 0.0, 0.25, 0.25 };

    // Check the projection.
    float y[3];
    auto result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 0.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle.
    p[0] = -1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point above the triangle.
    p[0] = 1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( -1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle at the  origin.
    p[0] = -1.0;
    p[1] = 0.0;
    p[2] = 0.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );
}

//---------------------------------------------------------------------------//
void rayFireTestXZ()
{
    // Create a facet in the XZ plane
    Kokkos::View<float*[4][3],Kokkos::HostSpace> facet( "facet", 1 );

    facet(0,0,0) = 0.0;
    facet(0,0,1) = 0.0;
    facet(0,0,2) = 0.0;

    facet(0,1,0) = 1.0;
    facet(0,1,1) = 0.0;
    facet(0,1,2) = 0.0;

    facet(0,2,0) = 0.0;
    facet(0,2,1) = 0.0;
    facet(0,2,2) = 1.0;

    // Rays will be perpendicular in y.
    float r[3] = { 0.0, 1.0, 0.0 };

    // Pick a point in the triangle first.
    float p[3] = { 0.25, 0.0, 0.25 };

    // Check the projection.
    float y[3];
    auto result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 0.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle.
    p[1] = -1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point above the triangle.
    p[1] = 1.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( -1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_FALSE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );

    // Move the point below the triangle at the  origin.
    p[0] = 0.0;
    p[1] = -1.0;
    p[2] = 0.0;

    // Check the projection.
    result =
        FacetGeometryOps::pointFacetProjection( p, r, facet, 0, y );
    EXPECT_TRUE( result );

    // Check distance.
    EXPECT_EQ( 1.0, y[2] );

    // Check the ray facet intersection - a point on the surface is not
    // included.
    EXPECT_TRUE( FacetGeometryOps::rayFacetIntersect(p,r,facet,0) );
}

//---------------------------------------------------------------------------//
void constructionTest()
{
    // Create inputs.
    InputParser parser( "facet_geometry_test.json", "json" );
    auto pt = parser.propertyTree();

    // Create the geometry from the test file. It contains a sphere of radius
    // 10 centered at the origin and a 4x4x4 cube centered at (15,15,15).
    // The sphere and box are saved as both surfaces and volumes. The global
    // bounding volume is from (-12,20) in each dimension.
    FacetGeometry<TEST_MEMSPACE> geometry( pt, TEST_EXECSPACE() );

    // Get the device data.
    const auto& geom_data = geometry.data();

    // Check that we got the right number of volumes and surfaces.
    EXPECT_EQ( geom_data.numVolume(), 3 );
    EXPECT_EQ( geom_data.numSurface(), 13 );

    // Check the global-to-local id conversion.
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( geometry.localVolumeId(i+1), i );
    for ( int i = 0; i < 13; ++i )
        EXPECT_EQ( geometry.localSurfaceId(i+1), i );

    // Get the facets for the sphere volume.
    auto num_volume_facet = geometry.numVolumeFacet( 1 );
    EXPECT_TRUE( num_volume_facet > 0 );

    // Check that a point is in the volume. All normals point outward if a
    // point is in the sphere all distances should be negative.
    Kokkos::Array<float,3> point_in = {-1.2, 0.4, 1.9 };
    int volume_inside = 0;
    Kokkos::parallel_reduce(
        "check_inside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_volume_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto volume_facets = geom_data.volumeFacets( 1 );
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_in.data(),
                                                       volume_facets, f);
            if ( dist < 0.0 )
                ++result;
        },
        volume_inside );
    EXPECT_EQ( volume_inside, num_volume_facet );

    // Check again with the point-in-volume primitive.
    volume_inside = 0;
    Kokkos::parallel_reduce(
        "check_inside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            auto volume_facets = geom_data.volumeFacets( 1 );
            if ( FacetGeometryOps::pointInVolume(
                     point_in.data(),volume_facets) )
                ++result;
        },
        volume_inside );
    EXPECT_EQ( volume_inside, 1 );

    // Check that a point is outside the volume. Some distances will be
    // negative, some will be positive. Check that some positive distances
    // were computed indicating an outside point.
    Kokkos::Array<float,3> point_out = {22.1, 1.4, -0.9 };
    int volume_outside = 0;
    Kokkos::parallel_reduce(
        "check_outside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_volume_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto volume_facets = geom_data.volumeFacets( 1 );
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_out.data(),
                                                       volume_facets, f);
            if ( dist > 0.0 )
                ++result;
        },
        volume_outside );
    EXPECT_TRUE( volume_outside > 0 );

    // Check again with the point-in-volume primitive.
    volume_outside = 0;
    Kokkos::parallel_reduce(
        "check_outside_volume",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            auto volume_facets = geom_data.volumeFacets( 1 );
            if ( !FacetGeometryOps::pointInVolume(
                     point_out.data(),volume_facets) )
                ++result;
        },
        volume_outside );
    EXPECT_EQ( volume_outside, 1 );

    // Get the facets for the sphere surface.
    auto num_surface_facet = geometry.numSurfaceFacet( 6 );
    EXPECT_TRUE( num_surface_facet > 0 );

    // Check that a point is in the surface. All normals point outward if a
    // point is in the sphere all distances should be negative.
    int surface_inside = 0;
    Kokkos::parallel_reduce(
        "check_inside_surface",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_surface_facet),
        KOKKOS_LAMBDA( const int f, int& result ){
            auto surface_facets = geom_data.surfaceFacets( 6 );
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
            auto surface_facets = geom_data.surfaceFacets( 6 );
            auto dist =
                FacetGeometryOps::distanceToFacetPlane(point_out.data(),
                                                       surface_facets, f);
            if ( dist > 0.0 )
                ++result;
        },
        surface_outside );
    EXPECT_TRUE( surface_outside > 0 );

    // Check the global bounding volume.
    EXPECT_EQ( geom_data.global_bounding_volume_id, 2 );

    auto global_box = geometry.globalBoundingBox();

    EXPECT_TRUE( global_box[0] <= point_in[0] );
    EXPECT_TRUE( global_box[1] <= point_in[1] );
    EXPECT_TRUE( global_box[2] <= point_in[2] );
    EXPECT_TRUE( global_box[3] >= point_in[0] );
    EXPECT_TRUE( global_box[4] >= point_in[1] );
    EXPECT_TRUE( global_box[5] >= point_in[2] );

    EXPECT_TRUE( global_box[0] <= point_out[0] );
    EXPECT_TRUE( global_box[1] <= point_out[1] );
    EXPECT_TRUE( global_box[2] <= point_out[2] );
    EXPECT_FALSE( global_box[3] >= point_out[0] );
    EXPECT_TRUE( global_box[4] >= point_out[1] );
    EXPECT_TRUE( global_box[5] >= point_out[2] );

    // Locate a point in the sphere.
    int volume_id = -100;
    Kokkos::Array<float,3> p1 = {0.0,0.0,0.0};
    Kokkos::parallel_reduce(
        "check_point_location",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            result =
                FacetGeometryOps::locatePoint(p1.data(),geom_data);
        },
        volume_id );
    EXPECT_EQ( volume_id, 1 );

    // Locate a point in the box.
    volume_id = -100;
    Kokkos::Array<float,3> p2 = {14.0,14.0,14.0};
    Kokkos::parallel_reduce(
        "check_point_location",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            result =
                FacetGeometryOps::locatePoint(p2.data(),geom_data);
        },
        volume_id );
    EXPECT_EQ( volume_id, 0 );

    // Locate a point in the implicit complement
    volume_id = -100;
    Kokkos::Array<float,3> p3 = {-11.0, -11.0, -11.0};
    Kokkos::parallel_reduce(
        "check_point_location",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            result =
                FacetGeometryOps::locatePoint(p3.data(),geom_data);
        },
        volume_id );
    EXPECT_EQ( volume_id, -1 );

    // Locate a point outside the domain.
    volume_id = -100;
    Kokkos::Array<float,3> p4 = {-20.0, 0.0, 0.0};
    Kokkos::parallel_reduce(
        "check_point_location",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,1),
        KOKKOS_LAMBDA( const int, int& result ){
            result =
                FacetGeometryOps::locatePoint(p4.data(),geom_data);
        },
        volume_id );
    EXPECT_EQ( volume_id, -2 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, ray_fire_test )
{
    rayFireTestXY();
    rayFireTestYZ();
    rayFireTestXZ();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test )
{
    constructionTest();
}

//---------------------------------------------------------------------------//
template<class MemorySpace>
struct LocateFunctor
{
    FacetGeometryData<MemorySpace> geom;

    template<class ParticleType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( const double x[3], ParticleType& p ) const
    {
        float xf[3] = {float(x[0]),float(x[1]),float(x[2])};
        for ( int d = 0; d < 3; ++d )
        {
            Cabana::get<0>(p,d) = x[d];
        }
        Cabana::get<1>(p) = FacetGeometryOps::locatePoint(xf,geom);
        return (Cabana::get<1>(p) > -2);
    }
};

void initExample()
{
    std::array<int,3> global_num_cell = { 40, 40, 40 };
    std::array<double,3> global_low_corner = { -12.0, -12.0, -12.0 };
    std::array<double,3> global_high_corner = { 20.0, 20.0, 20.0 };
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
    particle_list particles( "particles" );

    InputParser parser( "facet_geometry_test.json", "json" );
    auto pt = parser.propertyTree();

    FacetGeometry<TEST_MEMSPACE> geometry( pt, TEST_EXECSPACE() );

    LocateFunctor<TEST_MEMSPACE> init_func;
    init_func.geom = geometry.data();
    initializeParticles(
        InitUniform(), TEST_EXECSPACE(), *local_grid, 3, init_func, particles );

    SiloParticleWriter::writeTimeStep(
        *global_grid, 0, 0.0,
        Cabana::slice<0>(particles,"x"), Cabana::slice<1>(particles,"i") );
}

TEST( TEST_CATEGORY, init_example )
{
    initExample();
}

//---------------------------------------------------------------------------//

} // end namespace Test
