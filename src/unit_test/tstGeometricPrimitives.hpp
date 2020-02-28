#include <Harlow_GeometricPrimitives.hpp>
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
    // Create primitives. Note this array is on the host but we will actually
    // make the primitives on the device.
    Kokkos::View<Geometry::Primitives::Object<TEST_MEMSPACE>*,Kokkos::HostSpace>
        objects( "geometry", 5 );

    // Create brick 1.
    Kokkos::Array<double,3> b1e = { 1.001, 1.001, 1.001 };
    Geometry::Primitives::create(
        objects(0),
        Geometry::Primitives::BrickBuilder<TEST_MEMSPACE>(),
        b1e );

    // Create brick 2.
    Kokkos::Array<double,3> b2e = { 2.001, 2.001, 2.001 };
    Geometry::Primitives::create(
        objects(1),
        Geometry::Primitives::BrickBuilder<TEST_MEMSPACE>(),
        b2e );

    // Unite the bricks.
    Geometry::Primitives::create(
        objects(2),
        Geometry::Primitives::UnionBuilder<TEST_MEMSPACE>(),
        objects(0), objects(1) );

    // Difference the bricks ( 2 minus 1 or "in 2 but not in 1").
    Geometry::Primitives::create(
        objects(3),
        Geometry::Primitives::DifferenceBuilder<TEST_MEMSPACE>(),
        objects(1), objects(0) );

    // Intersect the bricks.
    Geometry::Primitives::create(
        objects(4),
        Geometry::Primitives::IntersectionBuilder<TEST_MEMSPACE>(),
        objects(0), objects(1) );

    // Make the objects device accessible.
    auto objects_device = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), objects );

    // Evaluate the objects on device.
    Kokkos::View<bool*[2],TEST_MEMSPACE> inside_results( "inside_results", 5 );
    Kokkos::View<double*[6],TEST_MEMSPACE> box_results( "box_results", 5 );
    Kokkos::parallel_for(
        "test_objects",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ){
            double x[3];
            double box[6];

            // Check brick 1
            x[0] = 0.25;
            x[1] = 0.25;
            x[2] = -0.25;
            inside_results(0,0) = objects_device(0).inside(x);

            x[0] = 0.55;
            inside_results(0,1) = objects_device(0).inside(x);

            objects_device(0).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(0,i) = box[i];

            // Check brick 2
            x[0] = -0.85;
            x[1] = -0.75;
            x[2] = -0.25;
            inside_results(1,0) = objects_device(1).inside(x);

            x[1] = -1.5;
            inside_results(1,1) = objects_device(1).inside(x);

            objects_device(1).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(1,i) = box[i];

            // Check the union.
            x[0] = -0.25;
            x[1] = -0.75;
            x[2] = -0.25;
            inside_results(2,0) = objects_device(2).inside(x);

            x[1] = -22.2;
            inside_results(2,1) = objects_device(2).inside(x);

            objects_device(2).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(2,i) = box[i];

            // Check the difference.
            x[0] = -0.7;
            x[1] = -0.1;
            x[2] = -0.1;
            inside_results(3,0) = objects_device(3).inside(x);

            x[1] = -22.2;
            inside_results(3,1) = objects_device(3).inside(x);

            objects_device(3).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(3,i) = box[i];

            // Check the intersection.
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            inside_results(4,0) = objects_device(4).inside(x);

            x[1] = -22.2;
            inside_results(4,1) = objects_device(4).inside(x);

            objects_device(4).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(4,i) = box[i];
        });

    // Check the results.
    auto inside_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), inside_results );
    auto box_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), box_results );

    // brick 1
    EXPECT_TRUE( inside_host(0,0) );
    EXPECT_FALSE( inside_host(0,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_FLOAT_EQ( box_host(0,d), -0.5*b1e[d] );
        EXPECT_FLOAT_EQ( box_host(0,d+3), 0.5*b1e[d] );
    }

    // brick 2
    EXPECT_TRUE( inside_host(1,0) );
    EXPECT_FALSE( inside_host(1,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_FLOAT_EQ( box_host(1,d), -0.5*b2e[d] );
        EXPECT_FLOAT_EQ( box_host(1,d+3), 0.5*b2e[d] );
    }

    // union
    EXPECT_TRUE( inside_host(2,0) );
    EXPECT_FALSE( inside_host(2,1) );
    EXPECT_FLOAT_EQ( box_host(2,0), -0.5*b2e[0] );
    EXPECT_FLOAT_EQ( box_host(2,1), -0.5*b2e[1] );
    EXPECT_FLOAT_EQ( box_host(2,2), -0.5*b2e[2] );
    EXPECT_FLOAT_EQ( box_host(2,3), 0.5*b2e[0] );
    EXPECT_FLOAT_EQ( box_host(2,4), 0.5*b2e[1] );
    EXPECT_FLOAT_EQ( box_host(2,5), 0.5*b2e[2] );

    // difference
    EXPECT_TRUE( inside_host(3,0) );
    EXPECT_FALSE( inside_host(3,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_FLOAT_EQ( box_host(3,d), -0.5*b2e[d] );
        EXPECT_FLOAT_EQ( box_host(3,d+3), 0.5*b2e[d] );
    }

    // intersection
    EXPECT_TRUE( inside_host(4,0) );
    EXPECT_FALSE( inside_host(4,1) );
    EXPECT_FLOAT_EQ( box_host(4,0), -0.5*b1e[0] );
    EXPECT_FLOAT_EQ( box_host(4,1), -0.5*b1e[1] );
    EXPECT_FLOAT_EQ( box_host(4,2), -0.5*b1e[2] );
    EXPECT_FLOAT_EQ( box_host(4,3), 0.5*b1e[0] );
    EXPECT_FLOAT_EQ( box_host(4,4), 0.5*b1e[1] );
    EXPECT_FLOAT_EQ( box_host(4,5), 0.5*b1e[2] );

    // Cleanup.
    for ( int i = 0; i < 5; ++i )
        Geometry::Primitives::destroy( objects(i) );
}

//---------------------------------------------------------------------------//
void moveTest()
{
    // Create primitives. Note this array is on the host but we will actually
    // make the primitives on the device.
    int nprimitive = 2;
    Kokkos::View<Geometry::Primitives::Object<TEST_MEMSPACE>*,Kokkos::HostSpace>
        objects( "geometry", nprimitive );

    // Create sphere
    double radius = 1.001;
    Geometry::Primitives::create(
        objects(0),
        Geometry::Primitives::SphereBuilder<TEST_MEMSPACE>(),
        radius );

    // Move the sphere.
    Kokkos::Array<double,3> distance = { -2.001, 2.001, -2.001 };
    Geometry::Primitives::create(
        objects(1),
        Geometry::Primitives::MoveBuilder<TEST_MEMSPACE>(),
        objects(0), distance );

    // Make the objects device accessible.
    auto objects_device = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), objects );

    // Evaluate the objects on device.
    Kokkos::View<bool*[2],TEST_MEMSPACE> inside_results( "inside_results", nprimitive );
    Kokkos::View<double*[6],TEST_MEMSPACE> box_results( "box_results", nprimitive );
    Kokkos::parallel_for(
        "test_objects",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ){
            double box[6];

            // In sphere but not move
            double x0[3] = { -0.85, 0.35, -0.1 };

            // In move but not sphere
            double x1[3] = { -1.2, 1.99, -1.75 };

            // Check sphere
            inside_results(0,0) = objects_device(0).inside(x0);
            inside_results(0,1) = objects_device(0).inside(x1);

            objects_device(0).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(0,i) = box[i];

            // Check move
            inside_results(1,0) = objects_device(1).inside(x1);
            inside_results(1,1) = objects_device(1).inside(x0);

            objects_device(1).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(1,i) = box[i];
        });

    // Check the results.
    auto inside_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), inside_results );
    auto box_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), box_results );

    // sphere
    EXPECT_TRUE( inside_host(0,0) );
    EXPECT_FALSE( inside_host(0,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_FLOAT_EQ( box_host(0,d), -radius );
        EXPECT_FLOAT_EQ( box_host(0,d+3), radius );
    }

    // move
    EXPECT_TRUE( inside_host(1,0) );
    EXPECT_FALSE( inside_host(1,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_FLOAT_EQ( box_host(1,d), -radius + distance[d] );
        EXPECT_FLOAT_EQ( box_host(1,d+3), radius + distance[d] );
    }

    // Cleanup.
    for ( int i = 0; i < nprimitive; ++i )
        Geometry::Primitives::destroy( objects(i) );
}

//---------------------------------------------------------------------------//
void rotateTest()
{
    // Create primitives. Note this array is on the host but we will actually
    // make the primitives on the device.
    int nprimitive = 4;
    Kokkos::View<Geometry::Primitives::Object<TEST_MEMSPACE>*,Kokkos::HostSpace>
        objects( "geometry", nprimitive );

    // Create brick at the origin
    Kokkos::Array<double,3> be = { 1.001, 2.001, 3.001 };
    Geometry::Primitives::create(
        objects(0),
        Geometry::Primitives::BrickBuilder<TEST_MEMSPACE>(),
        be );

    // Rotate the brick 90 degrees about x.
    double angle1 = 2.0 * atan(1.0);
    Kokkos::Array<double,3> axis1 = { 2.0, 0.0, 0.0 };
    Geometry::Primitives::create(
        objects(1),
        Geometry::Primitives::RotateBuilder<TEST_MEMSPACE>(),
        objects(0), angle1, axis1 );

    // Rotate the brick 90 degrees about y.
    double angle2 = 2.0 * atan(1.0);
    Kokkos::Array<double,3> axis2 = { 0.0, 2.0, 0.0 };
    Geometry::Primitives::create(
        objects(2),
        Geometry::Primitives::RotateBuilder<TEST_MEMSPACE>(),
        objects(0), angle2, axis2 );

    // Rotate the brick 90 degrees about z.
    double angle3 = 2.0 * atan(1.0);
    Kokkos::Array<double,3> axis3 = { 0.0, 0.0, 2.0 };
    Geometry::Primitives::create(
        objects(3),
        Geometry::Primitives::RotateBuilder<TEST_MEMSPACE>(),
        objects(0), angle3, axis3 );

    // Make the objects device accessible.
    auto objects_device = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), objects );

    // Evaluate the objects on device.
    Kokkos::View<bool*[2],TEST_MEMSPACE> inside_results( "inside_results", 3 );
    Kokkos::View<double*[6],TEST_MEMSPACE> box_results( "box_results", 3 );
    Kokkos::parallel_for(
        "test_objects",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ){
            double box[6];
            double x0[3];
            double x1[3];

            // Check rotate 1
            x0[0] = -0.5;
            x0[1] = 1.49;
            x0[2] = 0.99;
            inside_results(0,0) = objects_device(1).inside(x0);
            x1[0] = -0.5;
            x1[1] = 1.99;
            x1[2] = 1.49;
            inside_results(0,1) = objects_device(1).inside(x1);

            objects_device(1).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(0,i) = box[i];

            // Check rotate 2
            x0[0] = 1.49;
            x0[1] = 0.5;
            x0[2] = 0.49;
            inside_results(1,0) = objects_device(2).inside(x0);
            x1[0] = 0.49;
            x1[1] = 0.5;
            x1[2] = 1.49;
            inside_results(1,1) = objects_device(2).inside(x1);

            objects_device(2).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(1,i) = box[i];

            // Check rotate 3
            x0[0] = 0.99;
            x0[1] = 0.49;
            x0[2] = 0.5;
            inside_results(2,0) = objects_device(3).inside(x0);
            x1[0] = 0.49;
            x1[1] = 0.99;
            x1[2] = 0.5;
            inside_results(2,1) = objects_device(3).inside(x1);

            objects_device(3).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(2,i) = box[i];
        });

    // Check the results.
    auto inside_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), inside_results );
    auto box_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), box_results );

    // rotate 1
    EXPECT_TRUE( inside_host(0,0) );
    EXPECT_FALSE( inside_host(0,1) );
    EXPECT_FLOAT_EQ( box_host(0,0), -0.5 * be[0] );
    EXPECT_FLOAT_EQ( box_host(0,1), -0.5 * be[2] );
    EXPECT_FLOAT_EQ( box_host(0,2), -0.5 * be[1] );
    EXPECT_FLOAT_EQ( box_host(0,3), 0.5 * be[0] );
    EXPECT_FLOAT_EQ( box_host(0,4), 0.5 * be[2] );
    EXPECT_FLOAT_EQ( box_host(0,5), 0.5 * be[1] );

    // rotate 2
    EXPECT_TRUE( inside_host(1,0) );
    EXPECT_FALSE( inside_host(1,1) );
    EXPECT_FLOAT_EQ( box_host(1,0), -0.5 * be[2] );
    EXPECT_FLOAT_EQ( box_host(1,1), -0.5 * be[1] );
    EXPECT_FLOAT_EQ( box_host(1,2), -0.5 * be[0] );
    EXPECT_FLOAT_EQ( box_host(1,3), 0.5 * be[2] );
    EXPECT_FLOAT_EQ( box_host(1,4), 0.5 * be[1] );
    EXPECT_FLOAT_EQ( box_host(1,5), 0.5 * be[0] );

    // rotate 3
    EXPECT_TRUE( inside_host(2,0) );
    EXPECT_FALSE( inside_host(2,1) );
    EXPECT_FLOAT_EQ( box_host(2,0), -0.5 * be[1] );
    EXPECT_FLOAT_EQ( box_host(2,1), -0.5 * be[0] );
    EXPECT_FLOAT_EQ( box_host(2,2), -0.5 * be[2] );
    EXPECT_FLOAT_EQ( box_host(2,3), 0.5 * be[1] );
    EXPECT_FLOAT_EQ( box_host(2,4), 0.5 * be[0] );
    EXPECT_FLOAT_EQ( box_host(2,5), 0.5 * be[2] );

    // Cleanup.
    for ( int i = 0; i < nprimitive; ++i )
        Geometry::Primitives::destroy( objects(i) );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test )
{
    constructionTest();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, move_test )
{
    moveTest();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, rotate_test )
{
    rotateTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
