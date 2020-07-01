#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <random>

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

    // Check a deep copy.
    auto a_c = a;
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

    // Check a shallow transpose copy.
    auto a_t = ~a;
    EXPECT_EQ( a_t.extent(0), 3 );
    EXPECT_EQ( a_t.extent(1), 2 );

    EXPECT_EQ( a_t(0,0), 1.2 );
    EXPECT_EQ( a_t(1,0), -3.5 );
    EXPECT_EQ( a_t(2,0), 5.4 );
    EXPECT_EQ( a_t(0,1), 8.6 );
    EXPECT_EQ( a_t(1,1), 2.6 );
    EXPECT_EQ( a_t(2,1), -0.1 );

    // Check transpose of transpose shallow copy.
    auto a_t_t = ~a_t;
    EXPECT_EQ( a_t_t.extent(0), 2 );
    EXPECT_EQ( a_t_t.extent(1), 3 );

    EXPECT_EQ( a_t_t(0,0), 1.2 );
    EXPECT_EQ( a_t_t(0,1), -3.5 );
    EXPECT_EQ( a_t_t(0,2), 5.4 );
    EXPECT_EQ( a_t_t(1,0), 8.6 );
    EXPECT_EQ( a_t_t(1,1), 2.6 );
    EXPECT_EQ( a_t_t(1,2), -0.1 );

    // Check a transpose deep copy.
    LinearAlgebra::Matrix<double,3,2> a_t_c = ~a;
    EXPECT_EQ( a_t_c.stride_0(), 2 );
    EXPECT_EQ( a_t_c.stride_1(), 1 );
    EXPECT_EQ( a_t_c.extent(0), 3 );
    EXPECT_EQ( a_t_c.extent(1), 2 );

    EXPECT_EQ( a_t_c(0,0), 1.2 );
    EXPECT_EQ( a_t_c(1,0), -3.5 );
    EXPECT_EQ( a_t_c(2,0), 5.4 );
    EXPECT_EQ( a_t_c(0,1), 8.6 );
    EXPECT_EQ( a_t_c(1,1), 2.6 );
    EXPECT_EQ( a_t_c(2,1), -0.1 );

    // Check scalar assignment and operator().
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

    // Check scalar constructor.
    LinearAlgebra::Matrix<double,2,3> c = 32.3;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( c(i,j), 32.3 );

    // Check scalar multiplication.
    auto d = 2.0 * c;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( d(i,j), 64.6 );

    // Check scalr division.
    auto e = d / 2.0;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( e(i,j), 32.3 );
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
    EXPECT_EQ( x_c.stride_0(), 1 );
    EXPECT_EQ( x_c.extent(0), 3 );

    EXPECT_EQ( x_c(0), 1.2 );
    EXPECT_EQ( x_c(1), -3.5 );
    EXPECT_EQ( x_c(2), 5.4 );

    // Check scalar assignment and operator()
    x = 43.3;
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( x(i), 43.3 );

        x(i) = -10.2;
        EXPECT_EQ( x(i), -10.2 );
    }

    // Check default initialization
    LinearAlgebra::Vector<double,2> y;
    EXPECT_EQ( y.stride_0(), 1 );
    EXPECT_EQ( y.extent(0), 2 );

    // Check scalar constructor.
    LinearAlgebra::Vector<double,3> c = 32.3;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( c(i), 32.3 );

    // Check scalar multiplication.
    auto d = 2.0 * c;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( d(i), 64.6 );

    // Check scalar division.
    auto z = d / 2.0;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( z(i), 32.3 );

    // Check cross product.
    LinearAlgebra::Vector<double,3> e0 = { 1.0, 0.0, 0.0 };
    LinearAlgebra::Vector<double,3> e1 = { 0.0, 1.0, 0.0 };
    auto e2 = e0 % e1;
    EXPECT_EQ( e2(0), 0.0 );
    EXPECT_EQ( e2(1), 0.0 );
    EXPECT_EQ( e2(2), 1.0 );

    // Check element product.
    LinearAlgebra::Vector<double,2> f = { 2.0, 1.0 };
    LinearAlgebra::Vector<double,2> g = { 4.0, 2.0 };
    auto h = f & g;
    EXPECT_EQ( h(0), 8.0 );
    EXPECT_EQ( h(1), 2.0 );

    // Check element division.
    h = f | g;
    EXPECT_EQ( h(0), 0.5 );
    EXPECT_EQ( h(1), 0.5 );
}

//---------------------------------------------------------------------------//
void viewTest()
{
    double m[2][3] = { {1.2, -3.5, 5.4},
                       {8.6, 2.6, -0.1} };
    LinearAlgebra::MatrixView<double,2,3> a( &m[0][0], 3, 1 );
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

    double v[6] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };

    LinearAlgebra::VectorView<double,6> x1( &v[0], 1 );
    EXPECT_EQ( x1.stride_0(), 1 );
    EXPECT_EQ( x1.extent(0), 6 );
    for ( int i = 0; i < 6; ++i )
        EXPECT_EQ( x1(i), 1.0 * i );

    LinearAlgebra::VectorView<double,3> x2( &v[0], 2 );
    EXPECT_EQ( x2.stride_0(), 2 );
    EXPECT_EQ( x2.extent(0), 3 );
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( x2(i), 2.0 * i );

    LinearAlgebra::VectorView<double,2> x3( &v[1], 3 );
    EXPECT_EQ( x3.stride_0(), 3 );
    EXPECT_EQ( x3.extent(0), 2 );
    for ( int i = 0; i < 2; ++i )
        EXPECT_EQ( x3(i), 1.0 + 3.0 * i );
}

//---------------------------------------------------------------------------//
void matAddTest()
{
    LinearAlgebra::Matrix<double,1,2> a = { {2.0, 1.0} };
    LinearAlgebra::Matrix<double,1,2> b = { {2.0, 3.0} };

    auto c = a + b;
    EXPECT_EQ( c.extent(0), 1 );
    EXPECT_EQ( c.extent(1), 2 );
    EXPECT_EQ( c(0,0), 4.0 );
    EXPECT_EQ( c(0,1), 4.0 );

    auto d = ~a + ~b;
    EXPECT_EQ( d.extent(0), 2 );
    EXPECT_EQ( d.extent(1), 1 );
    EXPECT_EQ( d(0,0), 4.0 );
    EXPECT_EQ( d(1,0), 4.0 );

    LinearAlgebra::Matrix<double,2,1> e = { {2.0}, {3.0} };

    auto f = ~a + e;
    EXPECT_EQ( f.extent(0), 2 );
    EXPECT_EQ( f.extent(1), 1 );
    EXPECT_EQ( f(0,0), 4.0 );
    EXPECT_EQ( f(1,0), 4.0 );

    auto g = a + ~e;
    EXPECT_EQ( g.extent(0), 1 );
    EXPECT_EQ( g.extent(1), 2 );
    EXPECT_EQ( g(0,0), 4.0 );
    EXPECT_EQ( g(0,1), 4.0 );
}

//---------------------------------------------------------------------------//
void matSubTest()
{
    LinearAlgebra::Matrix<double,1,2> a = { {2.0, 1.0} };
    LinearAlgebra::Matrix<double,1,2> b = { {2.0, 3.0} };

    auto c = a - b;
    EXPECT_EQ( c.extent(0), 1 );
    EXPECT_EQ( c.extent(1), 2 );
    EXPECT_EQ( c(0,0), 0.0 );
    EXPECT_EQ( c(0,1), -2.0 );

    auto d = ~a - ~b;
    EXPECT_EQ( d.extent(0), 2 );
    EXPECT_EQ( d.extent(1), 1 );
    EXPECT_EQ( d(0,0), 0.0 );
    EXPECT_EQ( d(1,0), -2.0 );

    LinearAlgebra::Matrix<double,2,1> e = { {2.0}, {3.0} };

    auto f = ~a - e;
    EXPECT_EQ( f.extent(0), 2 );
    EXPECT_EQ( f.extent(1), 1 );
    EXPECT_EQ( f(0,0), 0.0 );
    EXPECT_EQ( f(1,0), -2.0 );

    auto g = a - ~e;
    EXPECT_EQ( g.extent(0), 1 );
    EXPECT_EQ( g.extent(1), 2 );
    EXPECT_EQ( g(0,0), 0.0 );
    EXPECT_EQ( g(0,1), -2.0 );
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
void vecAddTest()
{
    LinearAlgebra::Vector<double,2> a = { 2.0, 1.0 };
    LinearAlgebra::Vector<double,2> b = { 2.0, 3.0 };

    auto c = a + b;
    EXPECT_EQ( c(0), 4.0 );
    EXPECT_EQ( c(1), 4.0 );
}

//---------------------------------------------------------------------------//
void vecSubTest()
{
    LinearAlgebra::Vector<double,2> a = { 2.0, 1.0 };
    LinearAlgebra::Vector<double,2> b = { 2.0, 3.0 };

    auto c = a - b;
    EXPECT_EQ( c(0), 0.0 );
    EXPECT_EQ( c(1), -2.0 );
}

//---------------------------------------------------------------------------//
void vecVecTest()
{
    LinearAlgebra::Vector<double,2> x = { 1.0, 2.0 };
    LinearAlgebra::Vector<double,2> y = { 2.0, 3.0 };

    auto dot = ~x * y;
    EXPECT_EQ( dot.extent(0), 1 );
    EXPECT_EQ( dot.extent(1), 1 );
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
void expressionTest()
{
    LinearAlgebra::Matrix<double,2,2> a = { {2.0, 1.0}, {2.0, 1.0} };
    LinearAlgebra::Matrix<double,2,2> i = { {1.0, 0.0}, {0.0, 1.0} };
    LinearAlgebra::Vector<double,2> x = { 1.0, 2.0 };

    auto op1 = a + ~a;
    EXPECT_EQ( op1(0,0), 4.0 );
    EXPECT_EQ( op1(0,1), 3.0 );
    EXPECT_EQ( op1(1,0), 3.0 );
    EXPECT_EQ( op1(1,1), 2.0 );

    auto op2 = 0.5*(a + ~a);
    EXPECT_EQ( op2(0,0), 2.0 );
    EXPECT_EQ( op2(0,1), 1.5 );
    EXPECT_EQ( op2(1,0), 1.5 );
    EXPECT_EQ( op2(1,1), 1.0 );

    auto op3 = 0.5*(a + ~a) * i;
    EXPECT_EQ( op3(0,0), 2.0 );
    EXPECT_EQ( op3(0,1), 1.5 );
    EXPECT_EQ( op3(1,0), 1.5 );
    EXPECT_EQ( op3(1,1), 1.0 );

    auto op4 = x * ~x;
    EXPECT_EQ( op4(0,0), 1.0 );
    EXPECT_EQ( op4(0,1), 2.0 );
    EXPECT_EQ( op4(1,0), 2.0 );
    EXPECT_EQ( op4(1,1), 4.0 );

    auto op5 = 0.5 * (a + ~a) * i + (x * ~x);
    EXPECT_EQ( op5(0,0), 3.0 );
    EXPECT_EQ( op5(0,1), 3.5 );
    EXPECT_EQ( op5(1,0), 3.5 );
    EXPECT_EQ( op5(1,1), 5.0 );
}

//---------------------------------------------------------------------------//
template<int N>
void linearSolveTest()
{
    LinearAlgebra::Matrix<double,N,N> A;
    LinearAlgebra::Vector<double,N> x0;

    std::default_random_engine engine( 349305 );
    std::uniform_real_distribution<double> dist( 0.0, 1.0 );
    for ( int i = 0; i < N; ++i )
        x0(i) = dist(engine);
    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
            A(i,j) = dist(engine);

    double eps = 1.0e-12;

    auto b = A * x0;
    auto x1 = A ^ b;
    for ( int i = 0; i < N; ++i )
        EXPECT_NEAR( x0(i), x1(i), eps );

    auto c = ~A * x0;
    auto x2 = ~A ^ c;
    for ( int i = 0; i < N; ++i )
        EXPECT_NEAR( x0(i), x2(i), eps );
}

//---------------------------------------------------------------------------//
template<int N>
void kernelTest()
{
    int size = 10;
    Kokkos::View<double*[N][N],Kokkos::LayoutLeft,TEST_MEMSPACE> view_a( "a", size );
    Kokkos::View<double*[N],Kokkos::LayoutRight,TEST_MEMSPACE> view_x0( "x0", size );
    Kokkos::View<double*[N],Kokkos::LayoutLeft,TEST_MEMSPACE> view_x1( "x1", size );
    Kokkos::View<double*[N],Kokkos::LayoutRight,TEST_MEMSPACE> view_x2( "x2", size );

    Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> pool(3923423);
    Kokkos::fill_random(view_a, pool, 1.0 );
    Kokkos::fill_random(view_x0, pool, 1.0 );

    Kokkos::parallel_for(
        "test_la_kernel",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,size),
        KOKKOS_LAMBDA( const int i ){

            // Get views.
            LinearAlgebra::MatrixView<double,N,N> A_v(
                &view_a(i,0,0), view_a.stride_1(), view_a.stride_2() );
            LinearAlgebra::VectorView<double,N> x0_v(
                &view_x0(i,0), view_x0.stride_1() );
            LinearAlgebra::VectorView<double,N> x1_v(
                &view_x1(i,0), view_x1.stride_1() );
            LinearAlgebra::VectorView<double,N> x2_v(
                &view_x2(i,0), view_x2.stride_1() );

            // Gather.
            typename decltype(A_v)::copy_type A = A_v;
            typename decltype(x0_v)::copy_type x0 = x0_v;
            typename decltype(x1_v)::copy_type x1 = x1_v;
            typename decltype(x2_v)::copy_type x2 = x2_v;

            // Create a composite operator via an expression.
            auto op = 0.75 * (A + 0.5 * ~A);

            // Do work.
            auto b = op * x0;
            x1 = op ^ b;

            auto c = ~op * x0;
            x2 = ~op ^ c;

            // Scatter
            x1_v = x1;
            x2_v = x2;
        });

    auto x0_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view_x0 );
    auto x1_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view_x1 );
    auto x2_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view_x2 );

    double eps = 1.0e-11;
    for ( int i = 0; i < size; ++i )
        for ( int d = 0; d < N; ++d )
        {
            EXPECT_NEAR( x0_host(i,d), x1_host(i,d), eps );
            EXPECT_NEAR( x0_host(i,d), x2_host(i,d), eps );
    }
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

TEST( TEST_CATEGORY, view_test )
{
    viewTest();
}

TEST( TEST_CATEGORY, matadd_test )
{
    matAddTest();
}

TEST( TEST_CATEGORY, matsub_test )
{
    matSubTest();
}

TEST( TEST_CATEGORY, matmat_test )
{
    matMatTest();
}

TEST( TEST_CATEGORY, matVec_test )
{
    matVecTest();
}

TEST( TEST_CATEGORY, vecsub_test )
{
    vecSubTest();
}

TEST( TEST_CATEGORY, vecvec_test )
{
    vecVecTest();
}

TEST( TEST_CATEGORY, vecVec_test )
{
    vecVecTest();
}

TEST( TEST_CATEGORY, expression_test )
{
    expressionTest();
}

TEST( TEST_CATEGORY, linearSolve_test )
{
    linearSolveTest<2>();
    linearSolveTest<2>();
    linearSolveTest<4>();
    linearSolveTest<10>();
    linearSolveTest<20>();
}

TEST( TEST_CATEGORY, kernelTest )
{
    kernelTest<2>();
    kernelTest<3>();
    kernelTest<4>();
    kernelTest<10>();
    kernelTest<20>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
