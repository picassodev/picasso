#include <Harlow_GeometricPrimitives.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

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
    Kokkos::Array<double,3> b1e = { 1.0, 1.0, 1.0 };
    Kokkos::Array<double,3> b1o = { -0.5, 0.5, 0.5 };
    Geometry::Primitives::create(
        objects(0),
        Geometry::Primitives::BrickBuilder<TEST_MEMSPACE>(),
        b1e, b1o );

    // Create brick 2.
    Kokkos::Array<double,3> b2e = { 2.0, 2.0, 2.0 };
    Kokkos::Array<double,3> b2o = { -1.0, -1.0, 0.5 };
    Geometry::Primitives::create(
        objects(1),
        Geometry::Primitives::BrickBuilder<TEST_MEMSPACE>(),
        b2e, b2o );

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
            double box[3];

            // Check brick 1
            x[0] = -0.25;
            x[1] = 0.75;
            x[2] = 0.25;
            inside_results(0,0) = objects_device(0).inside(x);

            x[0] = -22.2;
            inside_results(0,1) = objects_device(0).inside(x);

            objects_device(0).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(0,i) = box[i];

            // Check brick 2
            x[0] = -1.25;
            x[1] = -0.75;
            x[2] = -0.25;
            inside_results(1,0) = objects_device(1).inside(x);

            x[1] = -22.2;
            inside_results(1,1) = objects_device(1).inside(x);

            objects_device(1).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(1,i) = box[i];

            // Check the union.
            x[0] = -1.25;
            x[1] = -0.75;
            x[2] = -0.25;
            inside_results(2,0) = objects_device(2).inside(x);

            x[1] = -22.2;
            inside_results(2,1) = objects_device(2).inside(x);

            objects_device(2).boundingBox(box);
            for ( int i = 0; i < 6; ++i )
                box_results(2,i) = box[i];

            // Check the difference.
            x[0] = -1.7;
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
        EXPECT_EQ( box_host(0,d), b1o[d] - 0.5*b1e[d] );
        EXPECT_EQ( box_host(0,d+3), b1o[d] + 0.5*b1e[d] );
    }

    // brick 2
    EXPECT_TRUE( inside_host(1,0) );
    EXPECT_FALSE( inside_host(1,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( box_host(1,d), b2o[d] - 0.5*b2e[d] );
        EXPECT_EQ( box_host(1,d+3), b2o[d] + 0.5*b2e[d] );
    }

    // union
    EXPECT_TRUE( inside_host(2,0) );
    EXPECT_FALSE( inside_host(2,1) );
    EXPECT_EQ( box_host(2,0), b2o[0] - 0.5*b2e[0] );
    EXPECT_EQ( box_host(2,1), b2o[1] - 0.5*b2e[1] );
    EXPECT_EQ( box_host(2,2), b2o[2] - 0.5*b2e[2] );
    EXPECT_EQ( box_host(2,3), b1o[0] + 0.5*b1e[0] );
    EXPECT_EQ( box_host(2,4), b1o[1] + 0.5*b1e[1] );
    EXPECT_EQ( box_host(2,5), b2o[2] + 0.5*b2e[2] );

    // difference
    EXPECT_TRUE( inside_host(3,0) );
    EXPECT_FALSE( inside_host(3,1) );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( box_host(3,d), b2o[d] - 0.5*b2e[d] );
        EXPECT_EQ( box_host(3,d+3), b2o[d] + 0.5*b2e[d] );
    }

    // intersection
    EXPECT_TRUE( inside_host(4,0) );
    EXPECT_FALSE( inside_host(4,1) );
    EXPECT_EQ( box_host(4,0), b1o[0] - 0.5*b1e[0] );
    EXPECT_EQ( box_host(4,1), b1o[1] - 0.5*b1e[1] );
    EXPECT_EQ( box_host(4,2), b1o[2] - 0.5*b1e[2] );
    EXPECT_EQ( box_host(4,3), b2o[0] + 0.5*b2e[0] );
    EXPECT_EQ( box_host(4,4), b2o[1] + 0.5*b2e[1] );
    EXPECT_EQ( box_host(4,5), b1o[2] + 0.5*b1e[2] );

    // Cleanup.
    for ( int i = 0; i < 5; ++i )
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

} // end namespace Test
