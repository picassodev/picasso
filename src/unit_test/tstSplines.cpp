#include <Harlow_Splines.hpp>
#include <Harlow_Types.hpp>

#include <vector>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( harlow_splines, linear_spline_test )
{
    // Check partition of unity for the linear spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[2];

    double x0 = Spline<FunctionOrder::Linear>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Linear>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<FunctionOrder::Linear>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Linear>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<FunctionOrder::Linear>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Linear>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int cell_id = 4;
    xp = low_x + (cell_id + 0.5) * dx;
    x0 = Spline<FunctionOrder::Linear>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[2];
    Spline<FunctionOrder::Linear>::stencil( offsets );
    EXPECT_EQ( int(x0) + offsets[0], cell_id );
    EXPECT_EQ( int(x0) + offsets[1], cell_id + 1);
}

TEST( harlow_splines, quadratic_spline_test )
{
    // Check partition of unity for the quadratic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[3];

    double x0 = Spline<FunctionOrder::Quadratic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Quadratic>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<FunctionOrder::Quadratic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Quadratic>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<FunctionOrder::Quadratic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Quadratic>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a dual cell (on a
    // node).
    int node_id = 4;
    xp = low_x + node_id * dx;
    x0 = Spline<FunctionOrder::Quadratic>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[3];
    Spline<FunctionOrder::Quadratic>::stencil( offsets );
    EXPECT_EQ( int(x0) + offsets[0], node_id - 1);
    EXPECT_EQ( int(x0) + offsets[1], node_id);
    EXPECT_EQ( int(x0) + offsets[2], node_id + 1);
}

TEST( harlow_splines, cubic_spline_test )
{
    // Check partition of unity for the cubic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[4];

    double x0 = Spline<FunctionOrder::Cubic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Cubic>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<FunctionOrder::Cubic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Cubic>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<FunctionOrder::Cubic>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<FunctionOrder::Cubic>::value( x0, values );
    sum = 0.0;
    for ( auto x : values ) sum += x;
    EXPECT_DOUBLE_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int cell_id = 4;
    xp = low_x + (cell_id + 0.5) * dx;
    x0 = Spline<FunctionOrder::Cubic>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[4];
    Spline<FunctionOrder::Cubic>::stencil( offsets );
    EXPECT_EQ( int(x0) + offsets[0], cell_id - 1 );
    EXPECT_EQ( int(x0) + offsets[1], cell_id );
    EXPECT_EQ( int(x0) + offsets[2], cell_id + 1 );
    EXPECT_EQ( int(x0) + offsets[3], cell_id + 2 );
}

} // end namespace Test
