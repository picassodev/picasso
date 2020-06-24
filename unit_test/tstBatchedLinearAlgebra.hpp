#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void dataTest()
{
    // Check a basic matrix.
    LinearAlgebra::Matrix<double,2,3> a = { {1.2, -3.5, 5.4},
                                            {8.6, 2.6, -0.1} };
    EXPECT_EQ( a.stride_0(), 3 );
    EXPECT_EQ( a.stride_1(), 1 );
    EXPECT_EQ( a.extent(0), 2 );
    EXPECT_EQ( a.extent(1), 3 );

    EXPECT_EQ( a(0,0), 1.2 );
    EXPECT_EQ( a(0,1), -3.5 );
    EXPECT_EQ( a(0,2), 5.4 );
    EXPECT_EQ( a(1,0), 8.6 );
    EXPECT_EQ( a(1,1), 2.6 );
    EXPECT_EQ( a(1,2), -0.1 );

    // Check a deep copy
    auto a_c = a;
    EXPECT_NE( a.data(), a_c.data() );
    EXPECT_EQ( a_c.stride_0(), 3 );
    EXPECT_EQ( a_c.stride_1(), 1 );
    EXPECT_EQ( a_c.extent(0), 2 );
    EXPECT_EQ( a_c.extent(1), 3 );

    EXPECT_EQ( a_c(0,0), 1.2 );
    EXPECT_EQ( a_c(0,1), -3.5 );
    EXPECT_EQ( a_c(0,2), 5.4 );
    EXPECT_EQ( a_c(1,0), 8.6 );
    EXPECT_EQ( a_c(1,1), 2.6 );
    EXPECT_EQ( a_c(1,2), -0.1 );

    // Check a transpose deep copy.
    LinearAlgebra::Matrix<double,3,2> a_t = ~a;
    EXPECT_EQ( a_t.stride_0(), 2 );
    EXPECT_EQ( a_t.stride_1(), 1 );
    EXPECT_EQ( a_t.extent(0), 3 );
    EXPECT_EQ( a_t.extent(1), 2 );

    EXPECT_EQ( a_t(0,0), 1.2 );
    EXPECT_EQ( a_t(1,0), -3.5 );
    EXPECT_EQ( a_t(2,0), 5.4 );
    EXPECT_EQ( a_t(0,1), 8.6 );
    EXPECT_EQ( a_t(1,1), 2.6 );
    EXPECT_EQ( a_t(2,1), -0.1 );

    // Check scalar assignment and operator()
    a = 43.3;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
        {
            EXPECT_EQ( a(i,j), 43.3 );

            a(i,j) = -10.2;
            EXPECT_EQ( a(i,j), -10.2 );
        }

    // Check default initialization.
    LinearAlgebra::Matrix<double,1,2> b;
    EXPECT_EQ( b.stride_0(), 2 );
    EXPECT_EQ( b.stride_1(), 1 );
    EXPECT_EQ( b.extent(0), 1 );
    EXPECT_EQ( b.extent(1), 2 );

    // Make a basic vector.
    LinearAlgebra::Vector<double,3> x = { 1.2, -3.5, 5.4};
    EXPECT_EQ( x.stride_0(), 1 );
    EXPECT_EQ( x.extent(0), 3 );

    EXPECT_EQ( x(0), 1.2 );
    EXPECT_EQ( x(1), -3.5 );
    EXPECT_EQ( x(2), 5.4 );

    // Check a deep copy
    auto x_c = x;
    EXPECT_NE( x.data(), x_c.data() );
    EXPECT_EQ( x_c.stride_0(), 1 );
    EXPECT_EQ( x_c.extent(0), 3 );

    EXPECT_EQ( x_c(0), 1.2 );
    EXPECT_EQ( x_c(1), -3.5 );
    EXPECT_EQ( x_c(2), 5.4 );

    // Check scalar assignment and operator()
    x = 43.3;
    for ( int i = 0; i < 2; ++i )
    {
        EXPECT_EQ( x(i), 43.3 );

        x(i) = -10.2;
        EXPECT_EQ( x(i), -10.2 );
    }

    // Check default initialization
    LinearAlgebra::Vector<double,2> y;
    EXPECT_EQ( y.stride_0(), 1 );
    EXPECT_EQ( y.extent(0), 2 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, data_test )
{
    dataTest();
}
//---------------------------------------------------------------------------//

} // end namespace Test
