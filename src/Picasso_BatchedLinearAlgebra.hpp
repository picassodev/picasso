/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <functional>
#include <type_traits>

#include <Picasso_Types.hpp>

namespace Picasso
{
namespace LinearAlgebra
{
//---------------------------------------------------------------------------//
// Overview
//---------------------------------------------------------------------------//
/*
  This file implements kernel-level dense linear algebra operations using a
  combination of expression templates for lazy evaluation and data structures
  to hold intermediates for eager evaluations when necessary.

  The general concept in the implementation for lazy vs. eager evalations is
  this: if an operation will evaluate the same expression multiple times
  (e.g. evaluating the element of a matrix in matrix-matrix multiplication)
  then the expressions are evaluated prior to performing the operation and the
  result of the operation is a copy rather than an expression as a means of
  reducing operations counts. Other operations which do not incur multiple
  evaluations (e.g. matrix-matrix addition) are evaluated lazily and return an
  expression as eager expressions in this case would cause excessive copy
  operations.

  The syntax covers all of the basic operations on vectors and matrices one
  would want to perform. Note that all product operations require the range
  and domain sizes of the vectors/matrices to be compatible. In what follows,
  consider A, B, and C to be matrices, x, y, and z to be vectors, and s to be
  a scalar:

  Scalar assignment: A = s; x = x;

  Copy assignment: A = B; x = y;

  Addition assignment: A += B; x += y;

  Subtraction assignment: A -= B; x -= y;

  Data access: A(i,j) = s; x(i) = s;

  Matrix transpose: ~A (if A is MxN, ~A is NxM)

  Vector transpose: ~x (if x is size N, ~x is a matrix 1xN)

  Matrix-matrix addition/subtraction: C = A + B; C = A - B;

  Vector-vector addition/subtraction: z = x + y; z = x - y;

  Matrix-matrix multiplication: C = A * B;

  Matrix-vector multiplication: y = A * x;

  Vector-matrix multiplication: B = x * A;

  Scalar multiplication: B = s * A; B = A * s; y = s * x; y = x * s;

  Scalar division: B = A / s; y = x / s;

  Dot product: s = ~x * y;

  Inner product: A = x * ~y;

  Cross product: z = x % y; (3-vectors only)

  Element-wise vector multiplication: z = x & y;

  Element-wise vector division: z = x | y;

  Matrix determinants: s = !A;

  LU decomposition: A_lu = LU(A); (returns a copy of the matrix decomposed)

  Matrix inverse: A_inv = inverse(A) (returns a copy of the matrix inverted)

  NxN linear solve (A*x=b): x = A ^ b;

  General asymmetric eigendecomposition


  We can string together multiple expressions to create a more complex
  expression which could have a mixture of eager and lazy evaluation depending
  on the operation type. For example, if A and B are NxN and x is of length N:

  C = 0.5 * (a + ~a) * B + (x * ~x);

  would return a matrix C that is NxN and:

  z = (0.5 * (a + ~a) * B + (x * ~x)) * y;

  would return a vector z of length N and:

  s = !C * ~y  * ((0.5 * (a + ~a) * B + (x * ~x)) * y);

  would return a scalar.
 */

//---------------------------------------------------------------------------//
// Forward declarations.
//---------------------------------------------------------------------------//
template <class T, class Func>
struct QuaternionExpression;
template <class T>
struct Quaternion;
template <class T>
struct QuaternionView;
template <class T, int M, int N, int P, class Func>
struct Tensor3Expression;
template <class T, int M, int N, int P>
struct Tensor3;
template <class T, int M, int N, int P>
struct Tensor3View;
template <class T, int M, int N, int P, int Q>
struct Tensor4;
template <class T, int M, int N, int P, int Q>
struct Tensor4View;
template <class T, int M, int N, int P, int Q, class Func>
struct Tensor4Expression;

//---------------------------------------------------------------------------//
// Type traits.
//---------------------------------------------------------------------------//
// Quaternion
template <class>
struct is_quaternion_impl : public std::false_type
{
};

template <class T, class Func>
struct is_quaternion_impl<QuaternionExpression<T, Func>> : public std::true_type
{
};

template <class T>
struct is_quaternion_impl<Quaternion<T>> : public std::true_type
{
};

template <class T>
struct is_quaternion_impl<QuaternionView<T>> : public std::true_type
{
};

template <class T>
struct is_quaternion
    : public is_quaternion_impl<typename std::remove_cv<T>::type>::type
{
};

// Tensor3
template <class>
struct is_tensor3_impl : public std::false_type
{
};

template <class T, int M, int N, int P, class Func>
struct is_tensor3_impl<Tensor3Expression<T, M, N, P, Func>>
    : public std::true_type
{
};

template <class T, int M, int N, int P>
struct is_tensor3_impl<Tensor3<T, M, N, P>> : public std::true_type
{
};

template <class T, int M, int N, int P>
struct is_tensor3_impl<Tensor3View<T, M, N, P>> : public std::true_type
{
};

template <class T>
struct is_tensor3
    : public is_tensor3_impl<typename std::remove_cv<T>::type>::type
{
};

// Tensor4
template <class>
struct is_tensor4_impl : public std::false_type
{
};

template <class T, int M, int N, int P, int Q, class Func>
struct is_tensor4_impl<Tensor4Expression<T, M, N, P, Q, Func>>
    : public std::true_type
{
};

template <class T, int M, int N, int P, int Q>
struct is_tensor4_impl<Tensor4<T, M, N, P, Q>> : public std::true_type
{
};

template <class T, int M, int N, int P, int Q>
struct is_tensor4_impl<Tensor4View<T, M, N, P, Q>> : public std::true_type
{
};

template <class T>
struct is_tensor4
    : public is_tensor4_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Expression creation functions.
//---------------------------------------------------------------------------//
// Tensor4
template <class T, int M, int N, int P, int Q, class Func>
KOKKOS_INLINE_FUNCTION Tensor4Expression<T, M, N, P, Q, Func>
createTensor4Expression( const Func& f )
{
    return Tensor4Expression<T, M, N, P, Q, Func>( f );
}

// Tensor3
template <class T, int M, int N, int P, class Func>
KOKKOS_INLINE_FUNCTION Tensor3Expression<T, M, N, P, Func>
createTensor3Expression( const Func& f )
{
    return Tensor3Expression<T, M, N, P, Func>( f );
}

// Quaternion
template <class T, class Func>
KOKKOS_INLINE_FUNCTION QuaternionExpression<T, Func>
createQuaternionExpression( const Func& f )
{
    return QuaternionExpression<T, Func>( f );
}

//---------------------------------------------------------------------------//
// Expression containers.
//---------------------------------------------------------------------------//
// Tensor4 expression container.
template <class T, int M, int N, int P, int Q, class Func>
struct Tensor4Expression
{
    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;
    static constexpr int extent_3 = Q;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Tensor4<T, M, N, P, Q>;
    using copy_type = Tensor4<T, M, N, P, Q>;

    Func _f;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor4Expression() = default;

    // Create an expression from a callable object.
    KOKKOS_INLINE_FUNCTION
    Tensor4Expression( const Func& f )
        : _f( f )
    {
    }

    // Extent.
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 3 ? extent_3
                      : ( d == 2 ? extent_2
                                 : ( d == 0 ? extent_0
                                            : ( d == 1 ? extent_1 : 0 ) ) );
    }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j, const int k,
                           const int l ) const
    {
        return _f( i, j, k, l );
    }

    // Get a row as a vector exrpession.
    KOKKOS_INLINE_FUNCTION
    auto vector( const ALL_INDEX_t, const int n, const int p,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, M>(
            [=]( const int i ) { return ( *this )( i, n, p, q ); } );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const int m, const ALL_INDEX_t, const int p,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, N>(
            [=]( const int i ) { return ( *this )( m, i, p, q ); } );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const int m, const int n, const ALL_INDEX_t,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, P>(
            [=]( const int i ) { return ( *this )( m, n, i, q ); } );
    }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const int m, const int n, const int p,
                 const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, Q>(
            [=]( const int i ) { return ( *this )( m, n, p, i ); } );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const ALL_INDEX_t, const ALL_INDEX_t, const int p,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, M, N>(
            [=]( const int i, const int j )
            { return ( *this )( i, j, p, q ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const ALL_INDEX_t, const int n, const ALL_INDEX_t,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, M, P>(
            [=]( const int i, const int j )
            { return ( *this )( i, n, j, q ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const int m, const ALL_INDEX_t, const ALL_INDEX_t,
                 const int q ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, N, P>(
            [=]( const int i, const int j )
            { return ( *this )( m, i, j, q ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const ALL_INDEX_t, const int n, const int p,
                 const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, M, Q>(
            [=]( const int i, const int j )
            { return ( *this )( i, n, p, j ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const int m, const ALL_INDEX_t, const int p,
                 const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, N, Q>(
            [=]( const int i, const int j )
            { return ( *this )( m, i, p, j ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const int m, const int n, const ALL_INDEX_t,
                 const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, P, Q>(
            [=]( const int i, const int j )
            { return ( *this )( m, n, i, j ); } );
    }

    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    auto tensor3( const ALL_INDEX_t, const ALL_INDEX_t, const ALL_INDEX_t,
                  const int b )
    {
        return createTensor3Expression<T, M, N, P>(
            [=]( const int i, const int j, const int k )
            { return ( *this )( i, j, k, b ); } );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    auto tensor3( const ALL_INDEX_t, const ALL_INDEX_t, const int b,
                  const ALL_INDEX_t )
    {
        return createTensor3Expression<T, M, N, Q>(
            [=]( const int i, const int j, const int k )
            { return ( *this )( i, j, b, k ); } );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    auto tensor3( const ALL_INDEX_t, const int b, const ALL_INDEX_t,
                  const ALL_INDEX_t )
    {
        return createTensor3Expression<T, M, P, Q>(
            [=]( const int i, const int j, const int k )
            { return ( *this )( i, b, j, k ); } );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    auto tensor3( const int b, const ALL_INDEX_t, const ALL_INDEX_t,
                  const ALL_INDEX_t )
    {
        return createTensor3Expression<T, N, P, Q>(
            [=]( const int i, const int j, const int k )
            { return ( *this )( b, i, j, k ); } );
    }
};

// Tensor3 expression container.
template <class T, int M, int N, int P, class Func>
struct Tensor3Expression
{
    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Tensor3<T, M, N, P>;
    using copy_type = Tensor3<T, M, N, P>;

    Func _f;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor3Expression() = default;

    // Create an expression from a callable object.
    KOKKOS_INLINE_FUNCTION
    Tensor3Expression( const Func& f )
        : _f( f )
    {
    }

    // Extent.
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 2 ? extent_2
                      : ( d == 0 ? extent_0 : ( d == 1 ? extent_1 : 0 ) );
    }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j, const int k ) const
    {
        return _f( i, j, k );
    }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const ALL_INDEX_t, const int n, const int p ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, M>(
            [=]( const int i ) { return ( *this )( i, n, p ); } );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const int m, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, N>(
            [=]( const int i ) { return ( *this )( m, i, p ); } );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    auto vector( const int m, const int n, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createVectorExpression<T, P>(
            [=]( const int i ) { return ( *this )( m, n, i ); } );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const ALL_INDEX_t, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, M, N>(
            [=]( const int i, const int j ) { return ( *this )( i, j, p ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const ALL_INDEX_t, const int n, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, M, P>(
            [=]( const int i, const int j ) { return ( *this )( i, n, j ); } );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    auto matrix( const int m, const ALL_INDEX_t, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::createMatrixExpression<T, N, P>(
            [=]( const int i, const int j ) { return ( *this )( m, i, j ); } );
    }
};

//---------------------------------------------------------------------------//
// Quaternion expression container.
template <class T, class Func>
struct QuaternionExpression
{
    static constexpr int extent_0 = 4;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Quaternion<T>;
    using copy_type = Quaternion<T>;

    Func _f;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    QuaternionExpression() = default;

    // Create an expression from a callable object.
    KOKKOS_INLINE_FUNCTION
    QuaternionExpression( const Func& f )
        : _f( f )
    {
    }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int ) const { return extent_0; }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i ) const { return _f( i ); }

    // Evaluate the expression at an index. 2D version for vectors treated as
    // matrices.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, int ) const { return _f( i ); }
};

//---------------------------------------------------------------------------//
// Tensor3
//---------------------------------------------------------------------------//
// Dense 3-dimensional tensor.
template <class T, int M, int N, int P>
struct Tensor3
{
    T _d[M][N][P];

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = Tensor3View<T, M, N, P>;
    using copy_type = Tensor3<T, M, N, P>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor3() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor3( const std::initializer_list<
             std::initializer_list<std::initializer_list<T>>>
                 data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        for ( const auto& slice : data )
        {
            j = 0;
            for ( const auto& row : slice )
            {
                k = 0;
                for ( const auto& value : row )
                {
                    _d[i][j][k] = value;
                    ++k;
                }
                ++j;
            }
            ++i;
        }
    }

    // Deep copy constructor. Triggers expression evaluation.
    template <
        class Expression,
        typename std::enable_if<is_tensor3<Expression>::value, int>::type = 0>
    KOKKOS_INLINE_FUNCTION Tensor3( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = e( i, j, k );
            }
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor3( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = value;
            }
    }

    // Assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value, Tensor3&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = e( i, j, k );
            }
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value, Tensor3&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) += e( i, j, k );
            }
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value, Tensor3&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) -= e( i, j, k );
            }
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    Tensor3& operator=( const std::initializer_list<
                        std::initializer_list<std::initializer_list<T>>>
                            data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        for ( const auto& slice : data )
        {
            j = 0;
            for ( const auto& row : slice )
            {
                k = 0;
                for ( const auto& value : row )
                {
                    _d[i][j][k] = value;
                    ++k;
                }
                ++j;
            }
            ++i;
        }

        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Tensor3& operator=( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = value;
            }
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return N * P; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return P; }

    KOKKOS_INLINE_FUNCTION
    int stride_2() const { return 1; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const
    {
        return ( 0 == d ) ? N * P : ( 1 == d ? P : 1 );
    }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 2 ? extent_2
                      : ( d == 0 ? extent_0 : ( d == 1 ? extent_1 : 0 ) );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j, const int k ) const
    {
        return _d[i][j][k];
    }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j, const int k )
    {
        return _d[i][j][k];
    }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( &_d[0][0][0] ); }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, M>
    vector( const ALL_INDEX_t, const int n, const int p ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, M>(
            const_cast<T*>( &_d[0][n][p] ), N * P );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, N>
    vector( const int m, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, N>(
            const_cast<T*>( &_d[m][0][p] ), P );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, P> vector( const int m, const int n,
                                                    const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, P>(
            const_cast<T*>( &_d[m][n][0] ), 1 );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, N>
    matrix( const ALL_INDEX_t, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, N>(
            const_cast<T*>( &_d[0][0][p] ), N * P, P );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, P>
    matrix( const ALL_INDEX_t, const int n, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, P>(
            const_cast<T*>( &_d[0][n][0] ), N * P, 1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, P>
    matrix( const int m, const ALL_INDEX_t, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, P>(
            const_cast<T*>( &_d[m][0][0] ), P, 1 );
    }
};

//---------------------------------------------------------------------------//
// View for wrapping Tensor3 data.
//
// NOTE: Data in this view may be non-contiguous.
template <class T, int M, int N, int P>
struct Tensor3View
{
    T* _d;
    int _stride[3];

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = Tensor3View<T, M, N, P>;
    using copy_type = Tensor3<T, M, N, P>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor3View() = default;

    // Tensor3View constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor3View( const Tensor3<T, M, N, P>& t )
        : _d( t.data() )
    {
        _stride[0] = t.stride_0();
        _stride[1] = t.stride_1();
        _stride[2] = t.stride_2();
    }

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor3View( T* data, const int stride_0, const int stride_1,
                 const int stride_2 )
        : _d( data )
    {
        _stride[0] = stride_0;
        _stride[1] = stride_1;
        _stride[2] = stride_2;
    }

    // Assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value,
                                Tensor3View&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = e( i, j, k );
            }
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value,
                                Tensor3View&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) += e( i, j, k );
            }
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor3<Expression>::value,
                                Tensor3View&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) -= e( i, j, k );
            }
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    Tensor3View& operator=( const std::initializer_list<
                            std::initializer_list<std::initializer_list<T>>>
                                data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        for ( const auto& slice : data )
        {
            j = 0;
            for ( const auto& row : slice )
            {
                k = 0;
                for ( const auto& value : row )
                {
                    ( *this )( i, j, k ) = value;
                    ++k;
                }
                ++j;
            }
            ++i;
        }

        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Tensor3View& operator=( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                    ( *this )( i, j, k ) = value;
            }
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return _stride[0]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return _stride[1]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_2() const { return _stride[2]; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const { return _stride[d]; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 2 ? extent_2
                      : ( d == 0 ? extent_0 : ( d == 1 ? extent_1 : 0 ) );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j, const int k ) const
    {
        return _d[_stride[0] * i + _stride[1] * j + _stride[2] * k];
    }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j, const int k )
    {
        return _d[_stride[0] * i + _stride[1] * j + _stride[2] * k];
    }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, M>
    vector( const ALL_INDEX_t, const int n, const int p ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, M>(
            const_cast<T*>( &_d[_stride[1] * n + _stride[2] * p] ), N * P );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, N>
    vector( const int m, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, N>(
            const_cast<T*>( &_d[_stride[0] * m + _stride[2] * p] ), P );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, P> vector( const int m, const int n,
                                                    const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, P>(
            const_cast<T*>( &_d[_stride[0] * m + _stride[1] * n] ), 1 );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, N>
    matrix( const ALL_INDEX_t, const ALL_INDEX_t, const int p ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, N>(
            const_cast<T*>( &_d[_stride[2] * p] ), N * P, P );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, P>
    matrix( const ALL_INDEX_t, const int n, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, P>(
            const_cast<T*>( &_d[_stride[1] * n] ), N * P, 1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, P>
    matrix( const int m, const ALL_INDEX_t, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, P>(
            const_cast<T*>( &_d[_stride[0] * m] ), P, 1 );
    }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( _d ); }
};

//---------------------------------------------------------------------------//
// Tensor4
//---------------------------------------------------------------------------//
// Dense rank-4 tensor.
template <class T, int M, int N, int P, int Q>
struct Tensor4
{
    T _d[M][N][P][Q];

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;
    static constexpr int extent_3 = Q;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = Tensor4View<T, M, N, P, Q>;
    using copy_type = Tensor4<T, M, N, P, Q>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor4() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor4( const std::initializer_list<std::initializer_list<
                 std::initializer_list<std::initializer_list<T>>>>
                 data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        int l = 0;

        for ( const auto& block : data )
        {
            j = 0;
            for ( const auto& slice : block )
            {
                k = 0;
                for ( const auto& row : slice )
                {
                    l = 0;
                    for ( const auto& value : row )
                    {
                        _d[i][j][k][l] = value;
                        ++l;
                    }
                    ++k;
                }
                ++j;
            }
            ++i;
        }
    }

    // Deep copy constructor. Triggers expression evaluation.
    template <
        class Expression,
        typename std::enable_if<is_tensor4<Expression>::value, int>::type = 0>
    KOKKOS_INLINE_FUNCTION Tensor4( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = e( i, j, k, l );
                }
            }
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor4( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = value;
                }
            }
    }

    // Assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value, Tensor4&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = e( i, j, k, l );
                }
            }
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value, Tensor4&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) += e( i, j, k, l );
                }
            }
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value, Tensor4&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) -= e( i, j, k, l );
                }
            }
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    Tensor4& operator=( const std::initializer_list<std::initializer_list<
                            std::initializer_list<std::initializer_list<T>>>>
                            data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        int l = 0;
        for ( const auto& block : data )
        {
            j = 0;
            for ( const auto& slice : block )
            {
                k = 0;
                for ( const auto& row : slice )
                {
                    l = 0;
                    for ( const auto& value : row )
                    {
                        _d[i][j][k][l] = value;
                        ++l;
                    }
                    ++k;
                }
                ++j;
            }
            ++i;
        }

        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Tensor4& operator=( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = value;
                }
            }
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return N * P * Q; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return P * Q; }

    KOKKOS_INLINE_FUNCTION
    int stride_2() const { return Q; }

    KOKKOS_INLINE_FUNCTION
    int stride_3() const { return 1; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const
    {
        return 2 == d ? Q : ( ( 0 == d ) ? N * P * Q : ( 1 == d ? P * Q : 1 ) );
    }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 3 ? extent_3
                      : ( d == 2 ? extent_2
                                 : ( d == 0 ? extent_0
                                            : ( d == 1 ? extent_1 : 0 ) ) );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j, const int k,
                                const int l ) const
    {
        return _d[i][j][k][l];
    }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j, const int k, const int l )
    {
        return _d[i][j][k][l];
    }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( &_d[0][0][0][0] ); }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, M>
    vector( const ALL_INDEX_t, const int n, const int p, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, M>(
            const_cast<T*>( &_d[0][n][p][q] ), N * P * Q );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, N>
    vector( const int m, const ALL_INDEX_t, const int p, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, N>(
            const_cast<T*>( &_d[m][0][p][q] ), P * Q );
    } // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, P>
    vector( const int m, const int n, const ALL_INDEX_t, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, P>(
            const_cast<T*>( &_d[m][n][0][q] ), Q );
    } // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, Q>
    vector( const int m, const int n, const int p, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, Q>(
            const_cast<T*>( &_d[m][n][p][0] ), 1 );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, N> matrix( const ALL_INDEX_t,
                                                       const ALL_INDEX_t,
                                                       const int p,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, N>(
            const_cast<T*>( &_d[0][0][p][q] ), N * P * Q, P * Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, P> matrix( const ALL_INDEX_t,
                                                       const int n,
                                                       const ALL_INDEX_t,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, P>(
            const_cast<T*>( &_d[0][n][0][q] ), N * P * Q, Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, P> matrix( const int m,
                                                       const ALL_INDEX_t,
                                                       const ALL_INDEX_t,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, P>(
            const_cast<T*>( &_d[m][0][0][q] ), P * Q, Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, Q> matrix( const ALL_INDEX_t,
                                                       const int n, const int p,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, Q>(
            const_cast<T*>( &_d[0][n][p][0] ), N * P * Q, 1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, Q> matrix( const int m,
                                                       const ALL_INDEX_t,
                                                       const int p,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, Q>(
            const_cast<T*>( &_d[m][0][p][0] ), P * Q, 1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, P, Q> matrix( const int m, const int n,
                                                       const ALL_INDEX_t,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, P, Q>(
            const_cast<T*>( &_d[m][n][0][0] ), Q, 1 );
    }

    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, N, P> tensor3( const ALL_INDEX_t, const ALL_INDEX_t,
                                     const ALL_INDEX_t, const int b )
    {
        return Tensor3View<T, M, N, P>( const_cast<T*>( &_d[0][0][0][b] ),
                                        N * P * Q, P * Q, Q );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, N, Q> tensor3( const ALL_INDEX_t, const ALL_INDEX_t,
                                     const int b, const ALL_INDEX_t )
    {
        return Tensor3View<T, M, N, Q>( const_cast<T*>( &_d[0][0][b][0] ),
                                        N * P * Q, P * Q, 1 );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, P, Q> tensor3( const ALL_INDEX_t, const int b,
                                     const ALL_INDEX_t, const ALL_INDEX_t )
    {
        return Tensor3View<T, M, P, Q>( const_cast<T*>( &_d[0][b][0][0] ),
                                        N * P * Q, Q, 1 );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, N, P, Q> tensor3( const int b, const ALL_INDEX_t,
                                     const ALL_INDEX_t, const ALL_INDEX_t )
    {
        return Tensor3View<T, N, P, Q>( const_cast<T*>( &_d[b][0][0][0] ),
                                        P * Q, Q, 1 );
    }
};

//---------------------------------------------------------------------------//
// View for wrapping Tensor4 data.
//
// NOTE: Data in this view may be non-contiguous.
template <class T, int M, int N, int P, int Q>
struct Tensor4View
{
    T* _d;
    int _stride[4];

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;
    static constexpr int extent_2 = P;
    static constexpr int extent_3 = Q;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = Tensor4View<T, M, N, P, Q>;
    using copy_type = Tensor4<T, M, N, P, Q>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Tensor4View() = default;

    // Tensor3View constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor4View( const Tensor4<T, M, N, P, Q>& t )
        : _d( t.data() )
    {
        _stride[0] = t.stride_0();
        _stride[1] = t.stride_1();
        _stride[2] = t.stride_2();
        _stride[3] = t.stride_3();
    }

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Tensor4View( T* data, const int stride_0, const int stride_1,
                 const int stride_2, const int stride_3 )
        : _d( data )
    {
        _stride[0] = stride_0;
        _stride[1] = stride_1;
        _stride[2] = stride_2;
        _stride[3] = stride_3;
    }

    // Assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value,
                                Tensor4View&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = e( i, j, k, l );
                }
            }
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value,
                                Tensor4View&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) += e( i, j, k, l );
                }
            }
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_tensor4<Expression>::value,
                                Tensor4View&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );
        static_assert( Expression::extent_2 == extent_2, "Extents must match" );
        static_assert( Expression::extent_3 == extent_3, "Extents must match" );

        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) -= e( i, j, k, l );
                }
            }
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    Tensor4View&
    operator=( const std::initializer_list<std::initializer_list<
                   std::initializer_list<std::initializer_list<T>>>>
                   data )
    {
        int i = 0;
        int j = 0;
        int k = 0;
        int l = 0;
        for ( const auto& block : data )
        {
            j = 0;
            for ( const auto& slice : block )
            {
                k = 0;
                for ( const auto& row : slice )
                {
                    l = 0;
                    for ( const auto& value : row )
                    {
                        ( *this )( i, j, k, l ) = value;
                        ++l;
                    }
                    ++k;
                }
                ++j;
            }
            ++i;
        }

        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Tensor4View& operator=( const T value )
    {
        for ( int i = 0; i < M; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
            {
                for ( int k = 0; k < P; ++k )
                {
                    for ( int l = 0; l < Q; ++l )
                        ( *this )( i, j, k, l ) = value;
                }
            }
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return _stride[0]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return _stride[1]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_2() const { return _stride[2]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_3() const { return _stride[3]; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const { return _stride[d]; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 3 ? extent_3
                      : ( d == 2 ? extent_2
                                 : ( d == 0 ? extent_0
                                            : ( d == 1 ? extent_1 : 0 ) ) );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j, const int k,
                                const int l ) const
    {
        return _d[_stride[0] * i + _stride[1] * j + _stride[2] * k +
                  _stride[3] * l];
    }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j, const int k, const int l )
    {
        return _d[_stride[0] * i + _stride[1] * j + _stride[2] * k +
                  _stride[3] * l];
    }

    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, M>
    vector( const ALL_INDEX_t, const int n, const int p, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, M>(
            const_cast<T*>(
                &_d[_stride[1] * n + _stride[2] * p + _stride[3] * q] ),
            N * P * Q );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, N>
    vector( const int m, const ALL_INDEX_t, const int p, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, N>(
            const_cast<T*>(
                &_d[_stride[0] * m + _stride[2] * p + _stride[3] * q] ),
            P * Q );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, P>
    vector( const int m, const int n, const ALL_INDEX_t, const int q ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, P>(
            const_cast<T*>(
                &_d[_stride[0] * m + _stride[1] * n + _stride[3] * q] ),
            Q );
    }
    // Get a row as a vector view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, Q>
    vector( const int m, const int n, const int p, const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::VectorView<T, Q>(
            const_cast<T*>(
                &_d[_stride[0] * m + _stride[1] * n + _stride[2] * p] ),
            1 );
    }

    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, N> matrix( const ALL_INDEX_t,
                                                       const ALL_INDEX_t,
                                                       const int p,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, N>(
            const_cast<T*>( &_d[_stride[2] * p + _stride[3] * q] ), N * P * Q,
            P * Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, P> matrix( const ALL_INDEX_t,
                                                       const int n,
                                                       const ALL_INDEX_t,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, P>(
            const_cast<T*>( &_d[_stride[1] * n + _stride[3] * q] ), N * P * Q,
            Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, P> matrix( const int m,
                                                       const ALL_INDEX_t,
                                                       const ALL_INDEX_t,
                                                       const int q ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, P>(
            const_cast<T*>( &_d[_stride[0] * m + _stride[3] * q] ), P * Q, Q );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, M, Q> matrix( const ALL_INDEX_t,
                                                       const int n, const int p,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, M, Q>(
            const_cast<T*>( &_d[_stride[1] * n + _stride[2] * p] ), N * P * Q,
            1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, N, Q> matrix( const int m,
                                                       const ALL_INDEX_t,
                                                       const int p,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, N, Q>(
            const_cast<T*>( &_d[_stride[0] * m + _stride[2] * p] ), P * Q, 1 );
    }
    // Get a matrix as a matrix view.
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::MatrixView<T, P, Q> matrix( const int m, const int n,
                                                       const ALL_INDEX_t,
                                                       const ALL_INDEX_t ) const
    {
        return Cabana::LinearAlgebra::MatrixView<T, P, Q>(
            const_cast<T*>( &_d[_stride[0] * m + _stride[1] * n] ), Q, 1 );
    }

    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, N, P> tensor3( const ALL_INDEX_t, const ALL_INDEX_t,
                                     const ALL_INDEX_t, const int b )
    {
        return Tensor3View<T, M, N, P>( const_cast<T*>( &_d[_stride[3] * b] ),
                                        N * P * Q, P * Q, Q );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, N, Q> tensor3( const ALL_INDEX_t, const ALL_INDEX_t,
                                     const int b, const ALL_INDEX_t )
    {
        return Tensor3View<T, M, N, Q>( const_cast<T*>( &_d[_stride[2] * b] ),
                                        N * P * Q, P * Q, 1 );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, M, P, Q> tensor3( const ALL_INDEX_t, const int b,
                                     const ALL_INDEX_t, const ALL_INDEX_t )
    {
        return Tensor3View<T, M, P, Q>( const_cast<T*>( &_d[_stride[1] * b] ),
                                        N * P * Q, Q, 1 );
    }
    // Get a tensor3 as a Tensor3 view.
    KOKKOS_INLINE_FUNCTION
    Tensor3View<T, N, P, Q> tensor3( const int b, const ALL_INDEX_t,
                                     const ALL_INDEX_t, const ALL_INDEX_t )
    {
        return Tensor3View<T, N, P, Q>( const_cast<T*>( &_d[_stride[0] * b] ),
                                        P * Q, Q, 1 );
    }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( _d ); }
};

//---------------------------------------------------------------------------//
// Quaternion
//---------------------------------------------------------------------------//
// Quaternion.
template <class T>
struct Quaternion
{
    T _d[4];

    static constexpr int extent_0 = 4;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = QuaternionView<T>;
    using copy_type = Quaternion<T>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Quaternion() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Quaternion( const std::initializer_list<T> data )
    {
        int i = 0;
        for ( const auto& value : data )
        {
            _d[i] = value;
            ++i;
        }
    }

    // Deep copy constructor. Triggers expression evaluation.
    template <class Expression,
              typename std::enable_if<is_quaternion<Expression>::value,
                                      int>::type = 0>
    KOKKOS_INLINE_FUNCTION Quaternion( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = e( i );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Quaternion( const T value )
    {
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = value;
    }

    // Scalar + vector constructor.
    KOKKOS_INLINE_FUNCTION
    Quaternion( const T value,
                const Cabana::LinearAlgebra::VectorView<T, 3> vec )
    {
        ( *this )( 0 ) = value;
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 1; i < extent_0; ++i )
            ( *this )( i ) = vec( i - 1 );
    }

    // Deep copy assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                Quaternion&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = e( i );
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                Quaternion&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) += e( i );
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                Quaternion&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) -= e( i );
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    Quaternion& operator=( const std::initializer_list<T> data )
    {
        int i = 0;
        for ( const auto& value : data )
        {
            _d[i] = value;
            ++i;
        }
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Quaternion& operator=( const T value )
    {
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = value;
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return 1; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return 0; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const { return ( 0 == d ) ? 1 : 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 0 ? extent_0 : ( d == 1 ? 1 : 0 );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i ) const { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i ) { return _d[i]; }

    // Access an individual element. 2D version for vectors treated as matrices.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int ) const { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int ) { return _d[i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( &_d[0] ); }

    // Get the scalar part
    KOKKOS_INLINE_FUNCTION
    const_reference scalar() const { return _d[0]; }

    KOKKOS_INLINE_FUNCTION
    reference scalar() { return _d[0]; }

    // Get the vector part
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, 3> vector() const
    {
        return Cabana::LinearAlgebra::VectorView<T, 3>(
            const_cast<T*>( &_d[1] ), 1 );
    }

    // Get the vector part
    KOKKOS_INLINE_FUNCTION
    Cabana::LinearAlgebra::VectorView<T, 3> vector()
    {
        return Cabana::LinearAlgebra::VectorView<T, 3>(
            const_cast<T*>( &_d[1] ), 1 );
    }
};

//---------------------------------------------------------------------------//
// View for wrapping quaternion data.
//
// NOTE: Data in this view may be non-contiguous.
template <class T>
struct QuaternionView
{
    T* _d;
    int _stride;

    static constexpr int extent_0 = 4;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    using eval_type = QuaternionView<T>;
    using copy_type = Quaternion<T>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    QuaternionView() = default;

    // Vector constructor.
    KOKKOS_INLINE_FUNCTION
    QuaternionView( const Quaternion<T>& q )
        : _d( q.data() )
        , _stride( q.stride_0() )
    {
    }

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    QuaternionView( T* data, const int stride )
        : _d( data )
        , _stride( stride )
    {
    }

    // Assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                QuaternionView&>::type
        operator=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = e( i );
        return *this;
    }

    // Addition assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                QuaternionView&>::type
        operator+=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) += e( i );
        return *this;
    }

    // Subtraction assignment operator. Triggers expression evaluation.
    template <class Expression>
    KOKKOS_INLINE_FUNCTION
        typename std::enable_if<is_quaternion<Expression>::value,
                                QuaternionView&>::type
        operator-=( const Expression& e )
    {
        static_assert( Expression::extent_0 == extent_0, "Extents must match" );
        static_assert( Expression::extent_1 == extent_1, "Extents must match" );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) -= e( i );
        return *this;
    }

    // Initializer list assignment operator.
    KOKKOS_INLINE_FUNCTION
    QuaternionView& operator=( const std::initializer_list<T> data )
    {
        int i = 0;
        for ( const auto& value : data )
        {
            ( *this )( i ) = value;
            ++i;
        }
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    QuaternionView& operator=( const T value )
    {
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int i = 0; i < extent_0; ++i )
            ( *this )( i ) = value;
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const { return _stride; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const { return 0; }

    KOKKOS_INLINE_FUNCTION
    int stride( const int d ) const { return ( 0 == d ) ? _stride : 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    constexpr int extent( const int d ) const
    {
        return d == 0 ? extent_0 : ( d == 1 ? 1 : 0 );
    }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i ) const { return _d[_stride * i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i ) { return _d[_stride * i]; }

    // Access an individual element. 2D version for vectors treated as matrices.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int ) const
    {
        return _d[_stride * i];
    }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int ) { return _d[_stride * i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const { return const_cast<pointer>( _d ); }
};

//---------------------------------------------------------------------------//
// Tensor4-tensor4 deep copy.
//---------------------------------------------------------------------------//
template <
    class Tensor4A, class ExpressionB,
    typename std::enable_if_t<
        is_tensor4<Tensor4A>::value && is_tensor4<ExpressionB>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void deepCopy( Tensor4A& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename Tensor4A::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( Tensor4A::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( Tensor4A::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    static_assert( Tensor4A::extent_2 == ExpressionB::extent_2,
                   "extent_2 must match" );
    static_assert( Tensor4A::extent_3 == ExpressionB::extent_3,
                   "extent_3 must match" );
    for ( int i = 0; i < Tensor4A::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < Tensor4A::extent_1; ++j )
            for ( int k = 0; k < Tensor4A::extent_2; ++k )
                for ( int l = 0; l < Tensor4A::extent_3; ++l )
                    a( i, j, k, l ) = b( i, j, k, l );
}

//---------------------------------------------------------------------------//
// Tensor3-tensor3 deep copy.
//---------------------------------------------------------------------------//
template <
    class Tensor3A, class ExpressionB,
    typename std::enable_if_t<
        is_tensor3<Tensor3A>::value && is_tensor3<ExpressionB>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void deepCopy( Tensor3A& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename Tensor3A::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( Tensor3A::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( Tensor3A::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    static_assert( Tensor3A::extent_2 == ExpressionB::extent_2,
                   "extent_2 must match" );
    for ( int i = 0; i < Tensor3A::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < Tensor3A::extent_1; ++j )
            for ( int k = 0; k < Tensor3A::extent_2; ++k )
                a( i, j, k ) = b( i, j, k );
}

//---------------------------------------------------------------------------//
// Matrix-matrix deep copy.
//---------------------------------------------------------------------------//
template <class MatrixA, class ExpressionB,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<MatrixA>::value &&
                  Cabana::LinearAlgebra::is_matrix<ExpressionB>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION void deepCopy( MatrixA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename MatrixA::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( MatrixA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( MatrixA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    for ( int i = 0; i < MatrixA::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < MatrixA::extent_1; ++j )
            a( i, j ) = b( i, j );
}

//---------------------------------------------------------------------------//
// Vector-vector deep copy.
//---------------------------------------------------------------------------//
template <class VectorX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<VectorX>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION void deepCopy( VectorX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename VectorX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( VectorX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < VectorX::extent_0; ++i )
        x( i ) = y( i );
}

//---------------------------------------------------------------------------//
// Quaternion-quaternion deep copy.
//---------------------------------------------------------------------------//
template <class QuaternionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<QuaternionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION void deepCopy( QuaternionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename QuaternionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( QuaternionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < QuaternionX::extent_0; ++i )
        x( i ) = y( i );
}

//---------------------------------------------------------------------------//
// Transpose.
//---------------------------------------------------------------------------//
// Quaternion conjugate
template <class Expression,
          typename std::enable_if_t<is_quaternion<Expression>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto operator~( const Expression& e )
{
    typename Expression::eval_type x_eval = e;

    return Quaternion<typename Expression::value_type>{
        x_eval( 0 ), -x_eval( 1 ), -x_eval( 2 ), -x_eval( 3 ) };
}

//---------------------------------------------------------------------------//
// Matrix-matrix addition.
//---------------------------------------------------------------------------//
template <class ExpressionA, class ExpressionB,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
                  Cabana::LinearAlgebra::is_matrix<ExpressionB>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator+( const ExpressionA& a,
                                       const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    return Cabana::LinearAlgebra::createMatrixExpression<
        typename ExpressionA::value_type, ExpressionA::extent_0,
        ExpressionA::extent_1>( [=]( const int i, const int j )
                                { return a( i, j ) + b( i, j ); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix subtraction.
//---------------------------------------------------------------------------//
template <class ExpressionA, class ExpressionB,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
                  Cabana::LinearAlgebra::is_matrix<ExpressionB>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator-( const ExpressionA& a,
                                       const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionA::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must_match" );
    return Cabana::LinearAlgebra::createMatrixExpression<
        typename ExpressionA::value_type, ExpressionA::extent_0,
        ExpressionA::extent_1>( [=]( const int i, const int j )
                                { return a( i, j ) - b( i, j ); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication.
//---------------------------------------------------------------------------//
template <class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        Cabana::LinearAlgebra::is_matrix<ExpressionB>::value,
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionA::extent_0, ExpressionB::extent_1>>
operator*( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_0,
                   "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionB::eval_type b_eval = b;
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionA::extent_0, ExpressionB::extent_1>
        c = static_cast<typename ExpressionA::value_type>( 0 );

    for ( int i = 0; i < ExpressionA::extent_0; ++i )
        for ( int j = 0; j < ExpressionB::extent_1; ++j )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int k = 0; k < ExpressionA::extent_1; ++k )
                c( i, j ) += a_eval( i, k ) * b_eval( k, j );

    return c;
}

//---------------------------------------------------------------------------//
// Matrix-vector multiplication
//---------------------------------------------------------------------------//
template <class ExpressionA, class ExpressionX>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        Cabana::LinearAlgebra::is_vector<ExpressionX>::value,
    Cabana::LinearAlgebra::Vector<typename ExpressionA::value_type,
                                  ExpressionA::extent_0>>
operator*( const ExpressionA& a, const ExpressionX& x )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionX::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_1 == ExpressionX::extent_0,
                   "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionX::eval_type x_eval = x;
    Cabana::LinearAlgebra::Vector<typename ExpressionA::value_type,
                                  ExpressionA::extent_0>
        y = static_cast<typename ExpressionA::value_type>( 0 );

    for ( int i = 0; i < ExpressionA::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionA::extent_1; ++j )
            y( i ) += a_eval( i, j ) * x_eval( j );

    return y;
}

//---------------------------------------------------------------------------//
// Vector-matrix multiplication. (i.e. vector-vector transpose multiplication)
//---------------------------------------------------------------------------//
template <class ExpressionA, class ExpressionX>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        Cabana::LinearAlgebra::is_vector<ExpressionX>::value,
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionX::extent_0, ExpressionA::extent_1>>
operator*( const ExpressionX& x, const ExpressionA& a )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionX::value_type>::value,
                   "value_type must match" );
    static_assert( 1 == ExpressionA::extent_0, "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionX::eval_type x_eval = x;
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionX::extent_0, ExpressionA::extent_1>
        y;

    for ( int i = 0; i < ExpressionX::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionA::extent_1; ++j )
            y( i, j ) = x_eval( i ) * a_eval( 0, j );

    return y;
}

//---------------------------------------------------------------------------//
// Vector-vector addition.
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator+( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return Cabana::LinearAlgebra::createVectorExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) + y( i ); } );
}

//---------------------------------------------------------------------------//
// Vector-vector subtraction.
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator-( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return Cabana::LinearAlgebra::createVectorExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) - y( i ); } );
}

//---------------------------------------------------------------------------//
// Quaternion-quaternion addition.
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION auto operator+( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return createQuaternionExpression<typename ExpressionX::value_type,
                                      ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) + y( i ); } );
}

//---------------------------------------------------------------------------//
// Quaternion-quaternion subtraction.
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION auto operator-( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return createQuaternionExpression<typename ExpressionX::value_type,
                                      ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) - y( i ); } );
}

//---------------------------------------------------------------------------//
// Quaternion-quaternion multiplication
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION auto operator&( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent_0 must match" );

    typename ExpressionX::eval_type x_eval = x;
    typename ExpressionY::eval_type y_eval = y;

    // Hamilton product of two quaternions
    return Quaternion<typename ExpressionX::value_type>{
        x_eval( 0 ) * y_eval( 0 ) - x_eval( 1 ) * y_eval( 1 ) -
            x_eval( 2 ) * y_eval( 2 ) - x_eval( 3 ) * y_eval( 3 ),
        x_eval( 0 ) * y_eval( 1 ) + x_eval( 1 ) * y_eval( 0 ) +
            x_eval( 2 ) * y_eval( 3 ) - x_eval( 3 ) * y_eval( 2 ),
        x_eval( 0 ) * y_eval( 2 ) - x_eval( 1 ) * y_eval( 3 ) +
            x_eval( 2 ) * y_eval( 0 ) + x_eval( 3 ) * y_eval( 1 ),
        x_eval( 0 ) * y_eval( 3 ) + x_eval( 1 ) * y_eval( 2 ) -
            x_eval( 2 ) * y_eval( 1 ) + x_eval( 3 ) * y_eval( 0 ) };
}

//---------------------------------------------------------------------------//
// Quaternion-matrix conjugation
//---------------------------------------------------------------------------//
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionX>::value &&
                  is_quaternion<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator&( const ExpressionX& X,
                                       const ExpressionY& q )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == 3 && ExpressionX::extent_1 == 3,
                   "matrix must be 3x3" );

    typename ExpressionX::eval_type X_eval = X;
    typename ExpressionY::eval_type q_eval = q;

    ExpressionX X_res;

    for ( int n = 0; n < 3; n++ )
    {
        LinearAlgebra::Quaternion<double> p = { 0.0, X_eval.row( n ) };
        auto p_rot = ( q_eval & p ) & ~q_eval;
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int d = 0; d < 3; d++ )
            X_res.row( n )( d ) = p_rot.vector()( d );
    }

    return X_res;
}

template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION auto operator&=( const ExpressionX& x,
                                        const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent_0 must match" );

    typename ExpressionX::eval_type x_eval = x;
    typename ExpressionY::eval_type y_eval = y;

    return Quaternion<typename ExpressionX::value_type>{
        x_eval( 0 ) * y_eval( 0 ) - x_eval( 1 ) * y_eval( 1 ) -
            x_eval( 2 ) * y_eval( 2 ) - x_eval( 3 ) * y_eval( 3 ),
        x_eval( 0 ) * y_eval( 1 ) + x_eval( 1 ) * y_eval( 0 ) +
            x_eval( 2 ) * y_eval( 3 ) - x_eval( 3 ) * y_eval( 2 ),
        x_eval( 0 ) * y_eval( 2 ) - x_eval( 1 ) * y_eval( 3 ) +
            x_eval( 2 ) * y_eval( 0 ) + x_eval( 3 ) * y_eval( 1 ),
        x_eval( 0 ) * y_eval( 3 ) + x_eval( 1 ) * y_eval( 2 ) -
            x_eval( 2 ) * y_eval( 1 ) + x_eval( 3 ) * y_eval( 0 ) };
}

//---------------------------------------------------------------------------//
// Quaternion-quaternion division
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value &&
                                        is_quaternion<ExpressionY>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION auto operator|( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );

    typename ExpressionX::eval_type x_eval = x;
    typename ExpressionY::eval_type y_eval = y;

    auto y_norm_2 = y_eval( 0 ) * y_eval( 0 ) + y_eval( 1 ) * y_eval( 1 ) +
                    y_eval( 2 ) * y_eval( 2 ) + y_eval( 3 ) * y_eval( 3 );
    auto y_inv = ~y_eval / y_norm_2;

    return x_eval & y_inv;
}

//---------------------------------------------------------------------------//
// Vector products.
//---------------------------------------------------------------------------//
// Cross product
template <class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_vector<ExpressionX>::value &&
        Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
    Cabana::LinearAlgebra::Vector<typename ExpressionX::value_type, 3>>
operator%( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == 3,
                   "cross product is for 3-vectors" );
    static_assert( ExpressionY::extent_0 == 3,
                   "cross product is for 3-vectors" );
    typename ExpressionX::eval_type x_eval = x;
    typename ExpressionY::eval_type y_eval = y;
    return Cabana::LinearAlgebra::Vector<typename ExpressionX::value_type, 3>{
        x_eval( 1 ) * y_eval( 2 ) - x_eval( 2 ) * y_eval( 1 ),
        x_eval( 2 ) * y_eval( 0 ) - x_eval( 0 ) * y_eval( 2 ),
        x_eval( 0 ) * y_eval( 1 ) - x_eval( 1 ) * y_eval( 0 ) };
}

//---------------------------------------------------------------------------//
// Element-wise multiplication.
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator&( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent_0 must match" );
    return Cabana::LinearAlgebra::createVectorExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) * y( i ); } );
}

//---------------------------------------------------------------------------//
// Element-wise division.
template <class ExpressionX, class ExpressionY,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionY>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto operator|( const ExpressionX& x,
                                       const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                                typename ExpressionY::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return Cabana::LinearAlgebra::createVectorExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0>(
        [=]( const int i ) { return x( i ) / y( i ); } );
}

//---------------------------------------------------------------------------//
// Scalar multiplication.
//---------------------------------------------------------------------------//
// Tensor4.
template <class ExpressionA,
          typename std::enable_if_t<is_tensor4<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const typename ExpressionA::value_type& s, const ExpressionA& a )
{
    return createTensor4Expression<
        typename ExpressionA::value_type, ExpressionA::extent_0,
        ExpressionA::extent_1, ExpressionA::extent_2, ExpressionA::extent_3>(
        [=]( const int i, const int j, const int k, const int l )
        { return s * a( i, j, k, l ); } );
}

template <class ExpressionA,
          typename std::enable_if_t<is_tensor4<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    return s * a;
}

// Tensor3.
template <class ExpressionA,
          typename std::enable_if_t<is_tensor3<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const typename ExpressionA::value_type& s, const ExpressionA& a )
{
    return createTensor3Expression<typename ExpressionA::value_type,
                                   ExpressionA::extent_0, ExpressionA::extent_1,
                                   ExpressionA::extent_2>(
        [=]( const int i, const int j, const int k )
        { return s * a( i, j, k ); } );
}

template <class ExpressionA,
          typename std::enable_if_t<is_tensor3<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    return s * a;
}

// Matrix.
template <class ExpressionA,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const typename ExpressionA::value_type& s, const ExpressionA& a )
{
    return Cabana::LinearAlgebra::createMatrixExpression<
        typename ExpressionA::value_type, ExpressionA::extent_0,
        ExpressionA::extent_1>( [=]( const int i, const int j )
                                { return s * a( i, j ); } );
}

template <class ExpressionA,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    return s * a;
}

//---------------------------------------------------------------------------//
// Vector.
template <class ExpressionX,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const typename ExpressionX::value_type& s, const ExpressionX& x )
{
    return Cabana::LinearAlgebra::createVectorExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0>(
        [=]( const int i ) { return s * x( i ); } );
}

template <class ExpressionX,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const ExpressionX& x, const typename ExpressionX::value_type& s )
{
    return s * x;
}

//---------------------------------------------------------------------------//
// Quaternion.
template <class ExpressionX,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const typename ExpressionX::value_type& s, const ExpressionX& x )
{
    return createQuaternionExpression<typename ExpressionX::value_type>(
        [=]( const int i ) { return s * x( i ); } );
}

template <class ExpressionX,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator*( const ExpressionX& x, const typename ExpressionX::value_type& s )
{
    return s * x;
}

//---------------------------------------------------------------------------//
// Scalar division.
//---------------------------------------------------------------------------//
// Tensor4.
template <class ExpressionA,
          typename std::enable_if_t<is_tensor4<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator/( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    auto s_inv = static_cast<typename ExpressionA::value_type>( 1 ) / s;
    return s_inv * a;
}

// Tensor3.
template <class ExpressionA,
          typename std::enable_if_t<is_tensor3<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator/( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    auto s_inv = static_cast<typename ExpressionA::value_type>( 1 ) / s;
    return s_inv * a;
}

// Matrix.
template <class ExpressionA,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_matrix<ExpressionA>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator/( const ExpressionA& a, const typename ExpressionA::value_type& s )
{
    auto s_inv = static_cast<typename ExpressionA::value_type>( 1 ) / s;
    return s_inv * a;
}

//---------------------------------------------------------------------------//
// Vector.
template <class ExpressionX,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator/( const ExpressionX& x, const typename ExpressionX::value_type& s )
{
    auto s_inv = static_cast<typename ExpressionX::value_type>( 1 ) / s;
    return s_inv * x;
}

//---------------------------------------------------------------------------//
// Quaternion.
template <class ExpressionX,
          typename std::enable_if_t<is_quaternion<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto
operator/( const ExpressionX& x, const typename ExpressionX::value_type& s )
{
    auto s_inv = static_cast<typename ExpressionX::value_type>( 1 ) / s;
    return s_inv * x;
}

//---------------------------------------------------------------------------//
// Matrix determinants.
//---------------------------------------------------------------------------//
// 2x2 specialization
template <class Expression>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<Expression>::value &&
        Expression::extent_0 == 2 && Expression::extent_1 == 2,
    typename Expression::value_type>
operator!( const Expression& a )
{
    return a( 0, 0 ) * a( 1, 1 ) - a( 0, 1 ) * a( 1, 0 );
}

//---------------------------------------------------------------------------//
// 3x3 specialization
template <class Expression>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<Expression>::value &&
        Expression::extent_0 == 3 && Expression::extent_1 == 3,
    typename Expression::value_type>
operator!( const Expression& a )
{
    return a( 0, 0 ) * a( 1, 1 ) * a( 2, 2 ) +
           a( 0, 1 ) * a( 1, 2 ) * a( 2, 0 ) +
           a( 0, 2 ) * a( 1, 0 ) * a( 2, 1 ) -
           a( 0, 2 ) * a( 1, 1 ) * a( 2, 0 ) -
           a( 0, 1 ) * a( 1, 0 ) * a( 2, 2 ) -
           a( 0, 0 ) * a( 1, 2 ) * a( 2, 1 );
}

//---------------------------------------------------------------------------//
// LU decomposition.
//---------------------------------------------------------------------------//
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value,
    typename ExpressionA::copy_type>
LU( const ExpressionA& a )
{
    using value_type = typename ExpressionA::value_type;
    constexpr int m = ExpressionA::extent_0;
    constexpr int n = ExpressionA::extent_1;
    constexpr int k = ( m < n ? m : n );

    typename ExpressionA::copy_type lu = a;

    for ( int p = 0; p < k; ++p )
    {
        const int iend = m - p;
        const int jend = n - p;

        const value_type diag = lu( p, p );

        for ( int i = 1; i < iend; ++i )
        {
            lu( i + p, p ) /= diag;
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 1; j < jend; ++j )
                lu( i + p, j + p ) -= lu( i + p, p ) * lu( p, j + p );
        }
    }

    return lu;
}

//---------------------------------------------------------------------------//
// Matrix trace
//---------------------------------------------------------------------------//
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value,
    typename ExpressionA::value_type>
trace( const ExpressionA& a )
{
    static_assert( ExpressionA::extent_1 == ExpressionA::extent_0,
                   "matrix must be square" );

    typename ExpressionA::value_type trace = 0.0;
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < ExpressionA::extent_0; ++i )
        trace += a( i, i );
    return trace;
}

//---------------------------------------------------------------------------//
// Identity
//---------------------------------------------------------------------------//
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value, void>
identity( ExpressionA& a )
{
    static_assert( ExpressionA::extent_1 == ExpressionA::extent_0,
                   "matrix must be square" );
    a = static_cast<typename ExpressionA::value_type>( 0 );
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < ExpressionA::extent_0; ++i )
        a( i, i ) = static_cast<typename ExpressionA::value_type>( 1 );
}

//---------------------------------------------------------------------------//
// Levi-Civita permutation tensor
//---------------------------------------------------------------------------//
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if_t<is_tensor3<ExpressionA>::value, void>
    permutation( ExpressionA& a )
{
    static_assert( ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3 &&
                       ExpressionA::extent_2 == 3,
                   "tensor3 must be 3x3x3" );
    a = static_cast<typename ExpressionA::value_type>( 0 );

    a = { { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, -1.0 }, { 0.0, 1.0, 0.0 } },
          { { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, 0.0, 0.0 } },
          { { 0.0, -1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } };
}

//---------------------------------------------------------------------------//
// Diagonal matrix.
//---------------------------------------------------------------------------//
template <class ExpressionX,
          typename std::enable_if_t<
              Cabana::LinearAlgebra::is_vector<ExpressionX>::value, int> = 0>
KOKKOS_INLINE_FUNCTION auto diagonal( const ExpressionX& x )
{
    return Cabana::LinearAlgebra::createMatrixExpression<
        typename ExpressionX::value_type, ExpressionX::extent_0,
        ExpressionX::extent_0>(
        [=]( const int i, const int j )
        {
            return ( i == j )
                       ? x( i )
                       : static_cast<typename ExpressionX::value_type>( 0 );
        } );
}

//---------------------------------------------------------------------------//
//  Contraction.
//  Tensor3 and a Vector
//---------------------------------------------------------------------------//

template <class ExpressionT, class ExpressionV,
          typename std::enable_if_t<
              is_tensor3<ExpressionT>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionV>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto contract( const ExpressionT& t,
                                      const ExpressionV& v,
                                      std::integral_constant<std::size_t, 0> )
{
    static_assert( ExpressionT::extent_0 == ExpressionV::extent_0,
                   "Inner extents must match" );

    typename ExpressionT::eval_type t_eval = t;
    typename ExpressionV::eval_type v_eval = v;
    Cabana::LinearAlgebra::Matrix<typename ExpressionT::value_type,
                                  ExpressionT::extent_1, ExpressionT::extent_2>
        res = static_cast<typename ExpressionT::value_type>( 0 );

    for ( int i = 0; i < ExpressionT::extent_1; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionT::extent_2; ++j )
            for ( int k = 0; k < ExpressionV::extent_0; ++k )
                res( i, j ) += t_eval( k, i, j ) * v_eval( k );

    return res;
}

template <class ExpressionT, class ExpressionV,
          typename std::enable_if_t<
              is_tensor3<ExpressionT>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionV>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto contract( const ExpressionT& t,
                                      const ExpressionV& v,
                                      std::integral_constant<std::size_t, 1> )
{
    static_assert( ExpressionT::extent_1 == ExpressionV::extent_0,
                   "Inner extents must match" );

    typename ExpressionT::eval_type t_eval = t;
    typename ExpressionV::eval_type v_eval = v;
    Cabana::LinearAlgebra::Matrix<typename ExpressionT::value_type,
                                  ExpressionT::extent_0, ExpressionT::extent_2>
        res = static_cast<typename ExpressionT::value_type>( 0 );

    for ( int i = 0; i < ExpressionT::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionT::extent_2; ++j )
            for ( int k = 0; k < ExpressionV::extent_0; ++k )
                res( i, j ) += t_eval( i, k, j ) * v_eval( k );

    return res;
}

template <class ExpressionT, class ExpressionV,
          typename std::enable_if_t<
              is_tensor3<ExpressionT>::value &&
                  Cabana::LinearAlgebra::is_vector<ExpressionV>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto contract( const ExpressionT& t,
                                      const ExpressionV& v,
                                      std::integral_constant<std::size_t, 2> )
{
    static_assert( ExpressionT::extent_2 == ExpressionV::extent_0,
                   "Inner extents must match" );

    typename ExpressionT::eval_type t_eval = t;
    typename ExpressionV::eval_type v_eval = v;
    Cabana::LinearAlgebra::Matrix<typename ExpressionT::value_type,
                                  ExpressionT::extent_0, ExpressionT::extent_1>
        res = static_cast<typename ExpressionT::value_type>( 0 );

    for ( int i = 0; i < ExpressionT::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionT::extent_1; ++j )
            for ( int k = 0; k < ExpressionV::extent_0; ++k )
                res( i, j ) += t_eval( i, j, k ) * v_eval( k );

    return res;
}

template <class ExpressionT, class ExpressionM,
          typename std::enable_if_t<
              is_tensor4<ExpressionT>::value &&
                  Cabana::LinearAlgebra::is_matrix<ExpressionM>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION auto contract( const ExpressionT& t,
                                      const ExpressionM& m )
{
    static_assert( ExpressionT::extent_2 == ExpressionM::extent_0,
                   "Inner extents must match" );
    static_assert( ExpressionT::extent_3 == ExpressionM::extent_1,
                   "Inner extents must match " );

    typename ExpressionT::eval_type t_eval = t;
    typename ExpressionM::eval_type m_eval = m;
    Cabana::LinearAlgebra::Matrix<typename ExpressionT::value_type,
                                  ExpressionT::extent_0, ExpressionT::extent_1>
        res = static_cast<typename ExpressionT::value_type>( 0 );

    for ( int i = 0; i < ExpressionT::extent_0; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < ExpressionT::extent_1; ++j )
            for ( int k = 0; k < ExpressionM::extent_0; ++k )
                for ( int l = 0; l < ExpressionM::extent_1; ++l )
                    res( i, j ) += t_eval( i, j, k, l ) * m_eval( k, l );

    return res;
}

//---------------------------------------------------------------------------//
// Matrix inverse.
//---------------------------------------------------------------------------//
// 2x2 specialization with determinant given.
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2,
    typename ExpressionA::copy_type>
inverse( const ExpressionA& a, const typename ExpressionA::value_type& a_det )
{
    typename ExpressionA::eval_type a_eval = a;

    auto a_det_inv = static_cast<typename ExpressionA::value_type>( 1 ) / a_det;

    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type, 2, 2> a_inv;

    a_inv( 0, 0 ) = a_eval( 1, 1 ) * a_det_inv;
    a_inv( 0, 1 ) = -a_eval( 0, 1 ) * a_det_inv;
    a_inv( 1, 0 ) = -a_eval( 1, 0 ) * a_det_inv;
    a_inv( 1, 1 ) = a_eval( 0, 0 ) * a_det_inv;

    return a_inv;
}

//---------------------------------------------------------------------------//
// 2x2 specialization.
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2,
    typename ExpressionA::copy_type>
inverse( const ExpressionA& a )
{
    typename ExpressionA::eval_type a_eval = a;
    return inverse( a_eval, !a_eval );
}

//---------------------------------------------------------------------------//
// 3x3 specialization with determinant given.
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    typename ExpressionA::copy_type>
inverse( const ExpressionA& a, const typename ExpressionA::value_type& a_det )
{
    typename ExpressionA::eval_type a_eval = a;

    auto a_det_inv = static_cast<typename ExpressionA::value_type>( 1 ) / a_det;

    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type, 3, 3> a_inv;

    a_inv( 0, 0 ) =
        ( a_eval( 1, 1 ) * a_eval( 2, 2 ) - a_eval( 1, 2 ) * a_eval( 2, 1 ) ) *
        a_det_inv;
    a_inv( 0, 1 ) =
        ( a_eval( 0, 2 ) * a_eval( 2, 1 ) - a_eval( 0, 1 ) * a_eval( 2, 2 ) ) *
        a_det_inv;
    a_inv( 0, 2 ) =
        ( a_eval( 0, 1 ) * a_eval( 1, 2 ) - a_eval( 0, 2 ) * a_eval( 1, 1 ) ) *
        a_det_inv;

    a_inv( 1, 0 ) =
        ( a_eval( 1, 2 ) * a_eval( 2, 0 ) - a_eval( 1, 0 ) * a_eval( 2, 2 ) ) *
        a_det_inv;
    a_inv( 1, 1 ) =
        ( a_eval( 0, 0 ) * a_eval( 2, 2 ) - a_eval( 0, 2 ) * a_eval( 2, 0 ) ) *
        a_det_inv;
    a_inv( 1, 2 ) =
        ( a_eval( 0, 2 ) * a_eval( 1, 0 ) - a_eval( 0, 0 ) * a_eval( 1, 2 ) ) *
        a_det_inv;

    a_inv( 2, 0 ) =
        ( a_eval( 1, 0 ) * a_eval( 2, 1 ) - a_eval( 1, 1 ) * a_eval( 2, 0 ) ) *
        a_det_inv;
    a_inv( 2, 1 ) =
        ( a_eval( 0, 1 ) * a_eval( 2, 0 ) - a_eval( 0, 0 ) * a_eval( 2, 1 ) ) *
        a_det_inv;
    a_inv( 2, 2 ) =
        ( a_eval( 0, 0 ) * a_eval( 1, 1 ) - a_eval( 0, 1 ) * a_eval( 1, 0 ) ) *
        a_det_inv;

    return a_inv;
}

//---------------------------------------------------------------------------//
// 3x3 specialization.
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    typename ExpressionA::copy_type>
inverse( const ExpressionA& a )
{
    typename ExpressionA::eval_type a_eval = a;
    return inverse( a_eval, !a_eval );
}

//---------------------------------------------------------------------------//
// General case.
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        !( ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2 ) &&
        !( ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3 ),
    typename ExpressionA::copy_type>
inverse( const ExpressionA& a )
{
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionA::extent_0, ExpressionA::extent1>
        ident;
    identity( ident );
    return a ^ ident;
}

//---------------------------------------------------------------------------//
// Matrix exponential
// Adapted from D. Gebremedhin and C. Weatherford
// https://arxiv.org/abs/1606.08395
//---------------------------------------------------------------------------//
template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value,
    typename ExpressionA::copy_type>
exponential( const ExpressionA& a )
{
    static_assert( ExpressionA::extent_1 == ExpressionA::extent_0,
                   "matrix must be square" );
    Cabana::LinearAlgebra::Matrix<typename ExpressionA::value_type,
                                  ExpressionA::extent_0, ExpressionA::extent_1>
        ident;
    identity( ident );

    auto a2 = a * a;
    auto a3 = a2 * a;
    auto a4 = a3 * a;

    double alpha = 4.955887515892002289e-14;
    Kokkos::Array<Kokkos::Array<double, 5>, 4> c;
    c = {
        Kokkos::Array<double, 5>{ 3599.994262347704951, 862.0738730089864644,
                                  -14.86233950714664427, -4.881331340410683266,
                                  1.0 },
        Kokkos::Array<double, 5>{ 1693.461215815646064, 430.8068649851425321,
                                  77.58934041908401266, 7.763092503482958289,
                                  1.0 },
        Kokkos::Array<double, 5>{ 1478.920917621023984, 387.7896702475912482,
                                  98.78409444643527097, 9.794888991082968084,
                                  1.0 },
        Kokkos::Array<double, 5>{ 2237.981769593417334, 545.9089563171489062,
                                  37.31797993128430013, 3.323349845844756893,
                                  1.0 },
    };

    auto ai1 = c[0][0] * ident + c[0][1] * a + c[0][2] * a2 + c[0][3] * a3 +
               c[0][4] * a4;
    auto ai2 = c[1][0] * ident + c[1][1] * a + c[1][2] * a2 + c[1][3] * a3 +
               c[1][4] * a4;
    auto ai3 = c[2][0] * ident + c[2][1] * a + c[2][2] * a2 + c[2][3] * a3 +
               c[2][4] * a4;
    auto ai4 = c[3][0] * ident + c[3][1] * a + c[3][2] * a2 + c[3][3] * a3 +
               c[3][4] * a4;

    return alpha * ( ai1 * ai2 * ai3 * ai4 );
}

//---------------------------------------------------------------------------//
// Linear solve.
//---------------------------------------------------------------------------//
// 2x2 specialization. Single and multiple RHS supported.
template <class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ( Cabana::LinearAlgebra::is_matrix<ExpressionB>::value ||
          Cabana::LinearAlgebra::is_vector<ExpressionB>::value ) &&
        ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2,
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    return inverse( a ) * b;
}

//---------------------------------------------------------------------------//
// 3x3 specialization. Single and multiple RHS supported.
template <class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ( Cabana::LinearAlgebra::is_matrix<ExpressionB>::value ||
          Cabana::LinearAlgebra::is_vector<ExpressionB>::value ) &&
        ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    return inverse( a ) * b;
}

//---------------------------------------------------------------------------//
// General case. Single and multiple RHS supported.
template <class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ( Cabana::LinearAlgebra::is_matrix<ExpressionB>::value ||
          Cabana::LinearAlgebra::is_vector<ExpressionB>::value ) &&
        !( ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2 ) &&
        !( ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3 ),
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                                typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_0,
                   "Inner extent must match" );
    static_assert( ExpressionA::extent_1 == ExpressionA::extent_0,
                   "matrix must be square" );

    using value_type = typename ExpressionA::value_type;
    constexpr int m = ExpressionB::extent_0;
    constexpr int n = ExpressionB::extent_1;

    // Compute LU decomposition.
    auto a_lu = LU( a );

    // Create RHS/LHS
    typename ExpressionB::copy_type x = b;

    // Solve Ly = b for y where y = Ux
    for ( int p = 0; p < m; ++p )
    {
        const int iend = m - p;
        const int jend = n;

        for ( int i = 1; i < iend; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
            for ( int j = 0; j < jend; ++j )
                x( i + p, j ) -= a_lu( i + p, p ) * x( p, j );
    }

    // Solve Ux = y for x.
    for ( int p = ( m - 1 ); p >= 0; --p )
    {
        const int iend = p;
        const int jend = n;

        const value_type diag = a_lu( p, p );

#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < n; ++j )
            x( p, j ) /= diag;

        if ( p > 0 )
        {
            for ( int i = 0; i < iend; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
                for ( int j = 0; j < jend; ++j )
                    x( i, j ) -= a_lu( i, p ) * x( p, j );
        }
    }

    return x;
}

template <typename ValueType>
KOKKOS_INLINE_FUNCTION void condSwap( bool b, ValueType& lv, ValueType& rv )
{
    auto tmpv = lv;
    lv = b ? rv : lv;
    rv = b ? tmpv : rv;
}

template <typename ValueType>
KOKKOS_INLINE_FUNCTION void condNegSwap( bool b, ValueType& lv, ValueType& rv )
{
    auto tmpv = -1.0 * lv;
    lv = b ? rv : lv;
    rv = b ? tmpv : rv;
}

template <>
KOKKOS_INLINE_FUNCTION void
condNegSwap<Cabana::LinearAlgebra::VectorView<double, 3>>(
    bool b, Cabana::LinearAlgebra::VectorView<double, 3>& lv,
    Cabana::LinearAlgebra::VectorView<double, 3>& rv )
{
    using vector_type =
        typename Cabana::LinearAlgebra::VectorView<double, 3>::copy_type;
    vector_type tmpv;

    if ( b )
    {
        for ( int d = 0; d < 3; d++ )
        {
            tmpv( d ) = -1.0 * lv( d );
            lv( d ) = rv( d );
            rv( d ) = tmpv( d );
        }
    }
}

KOKKOS_INLINE_FUNCTION
Quaternion<double> givensQuaternion( double a11, double a12, double a22,
                                     Kokkos::Array<std::size_t, 2> ij )
{
    const double gamma = 3.0 + 2.0 * Kokkos::sqrt( 2.0 );
    const double pi = 4.0 * Kokkos::atan( 1.0 );

    const double cs = Kokkos::cos( pi / 8.0 );
    const double ss = Kokkos::sin( pi / 8.0 );

    double ch = 2.0 * ( a11 - a22 );
    double sh = a12;

    bool b = ( gamma * sh * sh ) < ( ch * ch );

    double ome = 1.0 / Kokkos::sqrt( ch * ch + sh * sh );
    ch = b ? ome * ch : cs;
    sh = b ? ome * sh : ss;

    Quaternion<double> q = 0.0;

    // The ordering of the quaternion is different
    // according to the element indices
    if ( ij[0] == 0 && ij[1] == 1 )
    {
        q = { ch, 0.0, 0.0, sh };
    }
    else if ( ij[0] == 0 && ij[1] == 2 )
    {
        q = { ch, 0.0, -sh, 0.0 };
    }
    else if ( ij[0] == 1 && ij[1] == 2 )
    {
        q = { ch, sh, 0.0, 0.0 };
    }

    return q;
}

KOKKOS_INLINE_FUNCTION
Quaternion<double> givensQR( double a1, double a2, double tol,
                             Kokkos::Array<std::size_t, 2> ij )
{
    double rho = Kokkos::sqrt( a1 * a1 + a2 * a2 );

    double sh = rho > tol ? a2 : 0.0;
    double ch = Kokkos::fabs( a1 ) + Kokkos::max( rho, tol );

    condSwap( a1 < 0, sh, ch );

    double ome = 1.0 / Kokkos::sqrt( ch * ch + sh * sh );

    ch = ome * ch;
    sh = ome * sh;

    Quaternion<double> q = 0.0;

    // The ordering of the quaternion is different
    // according to the element indices
    if ( ij[0] == 2 && ij[1] == 0 )
    {
        q = { ch, 0.0, -sh, 0.0 };
    }
    else if ( ij[0] == 1 && ij[1] == 0 )
    {
        q = { ch, 0.0, 0.0, sh };
    }
    else if ( ij[0] == 2 && ij[1] == 1 )
    {
        q = { ch, sh, 0.0, 0.0 };
    }

    return q;
}

template <class ExpressionA>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    Quaternion<typename ExpressionA::value_type>>
cyclicJacobi( const ExpressionA& A, const int max_iter )
{
    // Form the symmetric positive-definite matrix from A
    auto S = ~A * A;

    // Indices for cyclic Jacobi
    Kokkos::Array<Kokkos::Array<std::size_t, 2>, 3> diag_inds;
    diag_inds = { Kokkos::Array<std::size_t, 2>{ 0, 1 },
                  Kokkos::Array<std::size_t, 2>{ 0, 2 },
                  Kokkos::Array<std::size_t, 2>{ 1, 2 } };

    // Get the next pq pair in the sequence
    auto pq = diag_inds[0];

    // Construct the 2x2 submatrix from the (pq)-entries of the 3x3 matrix
    double s_pp = S( pq[0], pq[0] );
    double s_pq = S( pq[0], pq[1] );
    double s_qq = S( pq[1], pq[1] );

    Quaternion<double> q_total = givensQuaternion( s_pp, s_pq, s_qq, pq );

    // Convert to rotation matrix for conjugation
    Cabana::LinearAlgebra::Matrix<double, 3, 3> Q_1{ q_total };

    S = ~Q_1 * S * Q_1;

    for ( int i = 1; i < max_iter; ++i )
    {
        // Get the next pq pair in the sequence
        pq = diag_inds[i % 3];

        // Construct the 2x2 submatrix from the (pq)-entries of the 3x3 matrix
        s_pp = S( pq[0], pq[0] );
        s_pq = S( pq[0], pq[1] );
        s_qq = S( pq[1], pq[1] );

        // Compute the approximate Given's quaternion
        auto q = givensQuaternion( s_pp, s_pq, s_qq, pq );

        // Convert to rotation matrix for conjugation
        Cabana::LinearAlgebra::Matrix<double, 3, 3> Q{ q };

        S = ~Q * S * Q; // Update to (k+1)
        q_total = q_total & q;
    }

    return q_total;
}

template <class MatrixType>
KOKKOS_INLINE_FUNCTION void sortSingularValues( MatrixType& B, MatrixType& V )
{
    // Array of singular values
    Kokkos::Array<double, 3> rho = { 0.0 };

    for ( int d = 0; d < 3; d++ )
    {
        auto b = B.column( d );
        rho[d] = b( 0 ) * b( 0 ) + b( 1 ) * b( 1 ) + b( 2 ) * b( 2 );
    }

    auto b0 = B.column( 0 );
    auto b1 = B.column( 1 );
    auto b2 = B.column( 2 );

    auto v0 = V.column( 0 );
    auto v1 = V.column( 1 );
    auto v2 = V.column( 2 );

    // Perform column swaps based on the ordering of the singular values
    condNegSwap( rho[0] < rho[1], b0, b1 );
    condNegSwap( rho[0] < rho[1], v0, v1 );

    condSwap( rho[0] < rho[1], rho[0], rho[1] );

    condNegSwap( rho[0] < rho[2], b0, b2 );
    condNegSwap( rho[0] < rho[2], v0, v2 );

    condSwap( rho[0] < rho[2], rho[0], rho[2] );

    condNegSwap( rho[1] < rho[2], b1, b2 );
    condNegSwap( rho[1] < rho[2], v1, v2 );
}

//---------------------------------------------------------------------------//
// Matrix Singular Value Decomposition
// Implementation based on McAdams, Selle, et al.,
// Technical Report Univ. Wisconsin, 2011.
//
// Computes the singular value decomposition of the matrix A
// A = U * Σ * V^T
//
//---------------------------------------------------------------------------//

template <class ExpressionA, class EigenU, class Diagonal, class EigenV>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<
    Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
        Cabana::LinearAlgebra::is_matrix<EigenU>::value &&
        Cabana::LinearAlgebra::is_matrix<Diagonal>::value &&
        Cabana::LinearAlgebra::is_matrix<EigenV>::value,
    void>
svd( const ExpressionA& A, EigenU& U, Diagonal& D, EigenV& V )
{
    constexpr int num_iter = 15;
    constexpr double tol = 1e-14;

    // Perform modified cyclic Jacobi iterations to calculate V
    auto q = cyclicJacobi( A, num_iter );

    // Normalize the quaternion and convert to the orthogonal matrix V
    double q_mag = Kokkos::sqrt( q( 0 ) * q( 0 ) + q( 1 ) * q( 1 ) +
                                 q( 2 ) * q( 2 ) + q( 3 ) * q( 3 ) );

    Cabana::LinearAlgebra::Matrix<double, 3, 3> V_rot{ q / q_mag };

    Cabana::LinearAlgebra::Matrix<double, 3, 3> Q = 0.0;

    // Perform a QR factorization of the matrix B = AV
    auto B = A * V_rot;

    sortSingularValues( B, V_rot );

    auto q21 = givensQR( B( 0, 0 ), B( 1, 0 ), tol, { 1, 0 } );
    auto Q1 = static_cast<Cabana::LinearAlgebra::Matrix<double, 3, 3>>( q21 );

    auto B1 = ~Q1 * B;

    auto q31 = givensQR( B1( 0, 0 ), B1( 2, 0 ), tol, { 2, 0 } );
    auto Q2 = static_cast<Cabana::LinearAlgebra::Matrix<double, 3, 3>>( q31 );

    auto B2 = ~Q2 * B1;

    auto q32 = givensQR( B2( 1, 1 ), B2( 2, 1 ), tol, { 2, 1 } );
    auto Q3 = static_cast<Cabana::LinearAlgebra::Matrix<double, 3, 3>>( q32 );

    auto B3 = ~Q3 * B2;

    Q = static_cast<Cabana::LinearAlgebra::Matrix<double, 3, 3>>( q21 & q31 &
                                                                  q32 );

    U = Q;
    D = B3;
    V = V_rot;
}

//---------------------------------------------------------------------------//
// Eigendecomposition
//---------------------------------------------------------------------------//
// template <class ExpressionA, class Eigenvalues, class Eigenvectors>
// KOKKOS_INLINE_FUNCTION
//     typename
//     std::enable_if_t<Cabana::LinearAlgebra::is_matrix<ExpressionA>::value &&
//                               Cabana::LinearAlgebra::Cabana::LinearAlgebra::is_vector<Eigenvalues>::value
//                               &&
//                               Cabana::LinearAlgebra::is_matrix<Eigenvectors>::value,
//                               typename ExpressionA::copy_type>
//     eigendecomposition( const ExpressionA& a,
//                         Eigenvalues& e_real,
//                         Eigenvalues& e_imag,
//                         Eigenvectors& u_left,
//                         Eigenvectors& u_right )
// {
//     static_assert( ExpressionA::extent_0 == ExpressionA::extent_1,
//                    "Matrix must be square" );
//     static_assert( std::is_same_v<typename Eigenvalues::value_type,
//                    typename ExpressionA::value_type>,
//                    "Value type must match" );
//     static_assert( std::is_same_v<typename Eigenvectors::value_type,
//                    typename ExpressionA::value_type>,
//                    "Value type must match" );
//     static_assert( Eigenvalues::extent_0 == ExpressionA::extent_0,
//                    "Dimensions must match" );
//     static_assert( Eigenvectors::extent_0 == ExpressionA::extent_0,
//                    "Dimensions must match" );
//     static_assert( Eigenvectors::extent_1 == ExpressionA::extent_0,
//                    "Dimensions must match" );
// }

//---------------------------------------------------------------------------//

} // end namespace LinearAlgebra

//---------------------------------------------------------------------------//
// Type aliases.
//---------------------------------------------------------------------------//

template <class T>
using Mat2 = Cabana::LinearAlgebra::Matrix<T, 2, 2>;
template <class T>
using MatView2 = Cabana::LinearAlgebra::MatrixView<T, 2, 2>;
template <class T>
using Vec2 = Cabana::LinearAlgebra::Vector<T, 2>;
template <class T>
using VecView2 = Cabana::LinearAlgebra::VectorView<T, 2>;

template <class T>
using Mat3 = Cabana::LinearAlgebra::Matrix<T, 3, 3>;
template <class T>
using MatView3 = Cabana::LinearAlgebra::MatrixView<T, 3, 3>;
template <class T>
using Vec3 = Cabana::LinearAlgebra::Vector<T, 3>;
template <class T>
using VecView3 = Cabana::LinearAlgebra::VectorView<T, 3>;

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
