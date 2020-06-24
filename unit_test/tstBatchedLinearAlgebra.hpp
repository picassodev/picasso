#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void matrixTest()
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
}

//---------------------------------------------------------------------------//
void vectorTest()
{
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
void matMatTest()
{
    // Square test.
    LinearAlgebra::Matrix<double,2,2> a = { {2.0, 1.0}, {2.0, 1.0} };
    LinearAlgebra::Matrix<double,2,2> b = { {2.0, 3.0}, {2.0, -1.0} };

    auto c = a * b;
    EXPECT_EQ( c.extent(0), 2 );
    EXPECT_EQ( c.extent(1), 2 );
    EXPECT_EQ( c(0,0), 6.0 );
    EXPECT_EQ( c(0,1), 5.0 );
    EXPECT_EQ( c(1,0), 6.0 );
    EXPECT_EQ( c(1,1), 5.0 );

    c = ~a * b;
    EXPECT_EQ( c(0,0), 8.0 );
    EXPECT_EQ( c(0,1), 4.0 );
    EXPECT_EQ( c(1,0), 4.0 );
    EXPECT_EQ( c(1,1), 2.0 );

    c = a * ~b;
    EXPECT_EQ( c(0,0), 7.0 );
    EXPECT_EQ( c(0,1), 3.0 );
    EXPECT_EQ( c(1,0), 7.0 );
    EXPECT_EQ( c(1,1), 3.0 );

    c = ~a * ~b;
    EXPECT_EQ( c(0,0), 10.0 );
    EXPECT_EQ( c(0,1), 2.0 );
    EXPECT_EQ( c(1,0), 5.0 );
    EXPECT_EQ( c(1,1), 1.0 );

    // Non square test.
    LinearAlgebra::Matrix<double,2,1> f = { {3.0}, {1.0} };
    LinearAlgebra::Matrix<double,1,2> g = { {2.0, 1.0} };

    auto h = f * g;
    EXPECT_EQ( h.extent(0), 2 );
    EXPECT_EQ( h.extent(1), 2 );
    EXPECT_EQ( h(0,0), 6.0 );
    EXPECT_EQ( h(0,1), 3.0 );
    EXPECT_EQ( h(1,0), 2.0 );
    EXPECT_EQ( h(1,1), 1.0 );

    auto j = f * ~f;
    EXPECT_EQ( j.extent(0), 2 );
    EXPECT_EQ( j.extent(1), 2 );
    EXPECT_EQ( j(0,0), 9.0 );
    EXPECT_EQ( j(0,1), 3.0 );
    EXPECT_EQ( j(1,0), 3.0 );
    EXPECT_EQ( j(1,1), 1.0 );

    auto k = ~f * f;
    EXPECT_EQ( k.extent(0), 1 );
    EXPECT_EQ( k.extent(1), 1 );
    EXPECT_EQ( k(0,0), 10.0 );

    auto m = ~f * ~g;
    EXPECT_EQ( m.extent(0), 1 );
    EXPECT_EQ( m.extent(1), 1 );
    EXPECT_EQ( m(0,0), 7.0 );
}

//---------------------------------------------------------------------------//
void matVecTest()
{
    // Square test.
    LinearAlgebra::Matrix<double,2,2> a = { {3.0, 2.0}, {1.0, 2.0} };
    LinearAlgebra::Vector<double,2> x = { 3.0, 1.0 };

    auto y = a * x;
    EXPECT_EQ( y.extent(0), 2 );
    EXPECT_EQ( y(0), 11.0 );
    EXPECT_EQ( y(1), 5.0 );

    y = ~a * x;
    EXPECT_EQ( y(0), 10.0 );
    EXPECT_EQ( y(1), 8.0 );

    auto b = ~x * a;
    EXPECT_EQ( b.extent(0), 1 );
    EXPECT_EQ( b.extent(1), 2 );
    EXPECT_EQ( b(0,0), 10.0 );
    EXPECT_EQ( b(0,1), 8.0 );

    b = ~x * ~a;
    EXPECT_EQ( b(0,0), 11.0 );
    EXPECT_EQ( b(0,1), 5.0 );

    // Non square test.
    LinearAlgebra::Matrix<double,1,2> c = { {1.0, 2.0} };
    LinearAlgebra::Vector<double,2> f = { 3.0, 2.0 };

    auto g = c * f;
    EXPECT_EQ( g.extent(0), 1 );
    EXPECT_EQ( g(0), 7.0 );

    auto h = ~f * ~c;
    EXPECT_EQ( h.extent(0), 1 );
    EXPECT_EQ( h.extent(1), 1 );
    EXPECT_EQ( h(0,0), 7.0 );

    LinearAlgebra::Matrix<double,2,1> j = { {1.0}, {2.0} };

    auto k = ~j * f;
    EXPECT_EQ( k.extent(0), 1 );
    EXPECT_EQ( k(0), 7.0 );

    auto l = ~f * j;
    EXPECT_EQ( l.extent(0), 1 );
    EXPECT_EQ( k(0), 7.0 );
}

//---------------------------------------------------------------------------//
void vecVecTest()
{
    LinearAlgebra::Vector<double,2> x = { 1.0, 2.0 };
    LinearAlgebra::Vector<double,2> y = { 2.0, 3.0 };

    auto dot = ~x * y;
    EXPECT_EQ( dot, 8.0 );

    auto inner = x * ~y;
    EXPECT_EQ( inner.extent(0), 2 );
    EXPECT_EQ( inner.extent(1), 2 );
    EXPECT_EQ( inner(0,0), 2.0 );
    EXPECT_EQ( inner(0,1), 3.0 );
    EXPECT_EQ( inner(1,0), 4.0 );
    EXPECT_EQ( inner(1,1), 6.0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, matrix_test )
{
    matrixTest();
}

TEST( TEST_CATEGORY, vector_test )
{
    vectorTest();
}

TEST( TEST_CATEGORY, matmat_test )
{
    matMatTest();
}

TEST( TEST_CATEGORY, matVec_test )
{
    matVecTest();
}

TEST( TEST_CATEGORY, vecVec_test )
{
    vecVecTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
