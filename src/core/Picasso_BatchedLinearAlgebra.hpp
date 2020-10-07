#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <KokkosBatched_Set_Decl.hpp>
#include <KokkosBatched_Set_Impl.hpp>

#include <KokkosBatched_Scale_Decl.hpp>
#include <KokkosBatched_Scale_Impl.hpp>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>

#include <KokkosBatched_SolveLU_Decl.hpp>

#include <type_traits>
#include <cmath>
#include <functional>

namespace Picasso
{
namespace LinearAlgebra
{
//---------------------------------------------------------------------------//
// Overview
//---------------------------------------------------------------------------//
/*
  This file implements kernel-level dense linear algebra operations using a
  combination of expression templates for lazy evaluation and KokkosKernels
  for eager evaluations when necessary.

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

  Data access: A(i,j) = s; x(i) = s;

  LU decomposition: A_lu = A.LU(); (returns a copy of the matrix decomposed)

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

  Inner product: A = x * y~;

  Cross product: z = x % y; (3-vectors only)

  Element-wise vector multiplication: z = x & y;

  Element-wise vector division: z = x | y;

  Matrix determinants: s = !A;

  NxN linear solve (A*x=b): x = A ^ b;


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
template<class T, int M, int N, class Func> struct MatrixExpression;
template<class T, int M, int N> struct Matrix;
template<class T, int M, int N> struct MatrixView;
template<class T, int N, class Func> struct VectorExpression;
template<class T, int N> struct Vector;
template<class T, int N> struct VectorView;

//---------------------------------------------------------------------------//
// Type traits.
//---------------------------------------------------------------------------//
// Matrix
template<class>
struct is_matrix_impl : public std::false_type
{};

template<class T, int M, int N, class Func>
struct is_matrix_impl<MatrixExpression<T,M,N,Func>> : public std::true_type
{};

template<class T, int M, int N>
struct is_matrix_impl<Matrix<T,M,N>> : public std::true_type
{};

template<class T, int M, int N>
struct is_matrix_impl<MatrixView<T,M,N>> : public std::true_type
{};

template<class T>
struct is_matrix : public is_matrix_impl<typename std::remove_cv<T>::type>::type
{};

// Vector
template<class>
struct is_vector_impl : public std::false_type
{};

template<class T, int N, class Func>
struct is_vector_impl<VectorExpression<T,N,Func>> : public std::true_type
{};

template<class T, int N>
struct is_vector_impl<Vector<T,N>> : public std::true_type
{};

template<class T, int N>
struct is_vector_impl<VectorView<T,N>> : public std::true_type
{};

template<class T>
struct is_vector : public is_vector_impl<typename std::remove_cv<T>::type>::type
{};

//---------------------------------------------------------------------------//
// Expression containers.
//---------------------------------------------------------------------------//
// Matrix expression container.
template<class T, int M, int N, class Func>
struct MatrixExpression
{
    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Matrix<T,M,N>;
    using copy_type = Matrix<T,M,N>;

    Func _f;
    int _extent[2] = {M,N};

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    MatrixExpression() = default;

    // Create an expression from a callable object.
    KOKKOS_INLINE_FUNCTION
    MatrixExpression( const Func& f )
        : _f( f )
    {}

    // Extent.
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j ) const
    { return _f(i,j); }

    // LU decomposition.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,M,N> LU() const
    {
        Matrix<T,M,N> lu = *this;
        KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke(
            lu );
        return lu;
    }
};

// Creation function.
template<class T, int M, int N, class Func>
KOKKOS_INLINE_FUNCTION
MatrixExpression<T,M,N,Func> createMatrixExpression( const Func& f )
{
    return MatrixExpression<T,M,N,Func>(f);
}

//---------------------------------------------------------------------------//
// Vector expression container.
template<class T, int N, class Func>
struct VectorExpression
{
    static constexpr int extent_0 = N;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Vector<T,N>;
    using copy_type = Vector<T,N>;

    Func _f;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    VectorExpression() = default;

    // Create an expression from a callable object.
    KOKKOS_INLINE_FUNCTION
    VectorExpression( const Func& f )
        : _f( f )
    {}

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int ) const
    { return N; }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i ) const
    { return _f(i); }
};

// Creation function.
template<class T, int N, class Func>
KOKKOS_INLINE_FUNCTION
VectorExpression<T,N,Func> createVectorExpression( const Func& f )
{
    return VectorExpression<T,N,Func>(f);
}

//---------------------------------------------------------------------------//
// Matrix
//---------------------------------------------------------------------------//
// Dense matrix with a KokkosKernels compatible data interface.
template<class T, int M, int N>
struct Matrix
{
    T _d[M][N];
    int _extent[2] = {M,N};

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = MatrixView<T,M,N>;
    using copy_type = Matrix<T,M,N>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Matrix() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const std::initializer_list<std::initializer_list<T>> data )
    {
        int i = 0;
        int j = 0;
        for ( const auto& row : data )
        {
            j = 0;
            for ( const auto& value : row )
            {
                _d[i][j] = value;
                ++j;
            }
            ++i;
        }
    }

    // Deep copy constructor. Triggers expression evaluation.
    template<class Expression,
             typename std::enable_if<is_matrix<Expression>::value,int>::type = 0>
    KOKKOS_INLINE_FUNCTION
    Matrix( const Expression& e )
    {
        for ( int i = 0; i < M; ++i )
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
                (*this)(i,j) = e(i,j);
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Assignment operator. Triggers expression evaluation.
    template<class Expression>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_matrix<Expression>::value,Matrix&>::type
    operator=( const Expression& e )
    {
        for ( int i = 0; i < M; ++i )
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
                (*this)(i,j) = e(i,j);
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return N; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 1; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j ) const
    { return _d[i][j]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[i][j]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0][0]); }

    // LU decomposition.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,M,N> LU() const
    {
        Matrix<T,M,N> lu = *this;
        KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke( lu );
        return lu;
    }
};

//---------------------------------------------------------------------------//
// Scalar overload.
template<class T>
struct Matrix<T,1,1>
{
    T _d;

    static constexpr int extent_0 = 1;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = MatrixView<T,1,1>;
    using copy_type = Matrix<T,1,1>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Matrix() = default;

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const T value )
        : _d( value )
    {}

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const T value )
    {
        _d = value;
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return 1; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 1; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int ) const
    { return 1; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int, const int ) const
    { return _d; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int, const int )
    { return _d; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d); }

    // Scalar conversion operator.
    KOKKOS_INLINE_FUNCTION
    operator value_type() const
    { return _d; }
};

//---------------------------------------------------------------------------//
// View for wrapping matrix data with a Kokkos-kernels compatible data
// interface.
//
// NOTE: Data in this view may be non-contiguous.
template<class T, int M, int N>
struct MatrixView
{
    T* _d;
    int _stride[2];
    int _extent[2] = {M,N};

    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = MatrixView<T,M,N>;
    using copy_type = Matrix<T,M,N>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    MatrixView() = default;

    // Matrix constructor.
    KOKKOS_INLINE_FUNCTION
    MatrixView( const Matrix<T,M,N>& m )
        : _d( m.data() )
    {
        _stride[0] = m.stride_0();
        _stride[1] = m.stride_1();
    }

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    MatrixView( T* data, const int stride_0, const int stride_1 )
        : _d( data )
    {
        _stride[0] = stride_0;
        _stride[1] = stride_1;
    }

    // Assignment operator. Triggers expression evaluation.
    template<class Expression>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_matrix<Expression>::value,MatrixView&>::type
    operator=( const Expression& e )
    {
        for ( int i = 0; i < M; ++i )
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
                (*this)(i,j) = e(i,j);
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    MatrixView& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return _stride[0]; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return _stride[1]; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j ) const
    { return _d[_stride[0]*i + _stride[1]*j]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[_stride[0]*i + _stride[1]*j]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }
};

//---------------------------------------------------------------------------//
// Vector
//---------------------------------------------------------------------------//
// Dense vector with a KokkosKernels compatible data interface.
template<class T, int N>
struct Vector
{
    T _d[N];
    int _extent[2] = {N,1};

    static constexpr int extent_0 = N;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = VectorView<T,N>;
    using copy_type = Vector<T,N>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Vector() = default;

    // Initializer list constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const std::initializer_list<T> data )
    {
        int i = 0;
        for ( const auto& value : data )
        {
            _d[i] = value;
            ++i;
        }
    }

    // Deep copy constructor. Triggers expression evaluation.
    template<class Expression,
             typename std::enable_if<is_vector<Expression>::value,int>::type = 0>
    KOKKOS_INLINE_FUNCTION
    Vector( const Expression& e )
    {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int i = 0; i < N; ++i )
                (*this)(i) = e(i);
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator. Triggers expression evaluation.
    template<class Expression>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_vector<Expression>::value,Vector&>::type
    operator=( const Expression& e )
    {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int i = 0; i < N; ++i )
                (*this)(i) = e(i);
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return 1; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i ) const
    { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }
};

//---------------------------------------------------------------------------//
// Scalar overload.
template<class T>
struct Vector<T,1>
{
    T _d;

    static constexpr int extent_0 = 1;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = VectorView<T,1>;
    using copy_type = Vector<T,1>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Vector() = default;

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const T value )
        : _d( value )
    {}

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const T value )
    {
        _d = value;
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return 1; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 1; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int ) const
    { return 1; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int ) const
    { return _d; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int )
    { return _d; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d); }

    // Scalar conversion operator.
    KOKKOS_INLINE_FUNCTION
    operator value_type() const
    { return _d; }
};

//---------------------------------------------------------------------------//
// View for wrapping vector data with matrix/vector objects. Kokkos-kernels
// compatible data interface.
//
// NOTE: Data in this view may be non-contiguous.
template<class T, int N>
struct VectorView
{
    T* _d;
    int _stride;
    int _extent[2] = {N,1};

    static constexpr int extent_0 = N;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using reference = T&;

    using eval_type = VectorView<T,N>;
    using copy_type = Vector<T,N>;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    VectorView() = default;

    // Vector construtor.
    KOKKOS_INLINE_FUNCTION
    VectorView( const Vector<T,N>& v )
        : _d( v.data() )
        , _stride( v.stride_0() )
    {}

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    VectorView( T* data, const int stride )
        : _d( data )
        , _stride( stride )
    {}

    // Assignment operator. Triggers expression evaluation.
    template<class Expression>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_vector<Expression>::value,VectorView&>::type
    operator=( const Expression& e )
    {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for ( int i = 0; i < N; ++i )
            (*this)(i) = e(i);
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    VectorView& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return _stride; }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return 0; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i ) const
    { return _d[_stride*i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[_stride*i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }
};

//---------------------------------------------------------------------------//
// Transpose.
//---------------------------------------------------------------------------//
// Matrix operator.
template<class Expression,
         typename std::enable_if_t<is_matrix<Expression>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator~( const Expression& e )
{
    return createMatrixExpression<
        typename Expression::value_type,
        Expression::extent_1,
        Expression::extent_0>(
            [=]( const int i, const int j ){ return e(j,i); } );
}

//---------------------------------------------------------------------------//
// Vector operator.
template<class Expression,
         typename std::enable_if_t<is_vector<Expression>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator~( const Expression& e )
{
    return createMatrixExpression<
        typename Expression::value_type,1,Expression::extent_0>(
            [=]( const int, const int j ){ return e(j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix addition.
//---------------------------------------------------------------------------//
template<class ExpressionA,
         class ExpressionB,
         typename std::enable_if_t<is_matrix<ExpressionA>::value &&
                                   is_matrix<ExpressionB>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator+( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    return createMatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){ return a(i,j) + b(i,j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix subtraction.
//---------------------------------------------------------------------------//
template<class ExpressionA,
         class ExpressionB,
         typename std::enable_if_t<is_matrix<ExpressionA>::value &&
                                   is_matrix<ExpressionB>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator-( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionA::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must_match" );
    return createMatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){ return a(i,j) - b(i,j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication.
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value && is_matrix<ExpressionB>::value,
    Matrix<typename ExpressionA::value_type,
           ExpressionA::extent_0,ExpressionB::extent_1>>
operator*( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_0,
                   "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionB::eval_type b_eval = b;
    Matrix<typename ExpressionA::value_type,
           ExpressionA::extent_0,
           ExpressionB::extent_1> c =
        Kokkos::ArithTraits<typename ExpressionA::value_type>::zero();
    KokkosBatched::SerialGemm<
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::Gemm::Unblocked>::invoke(
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            a_eval,
            b_eval,
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            c );
    return c;
}

//---------------------------------------------------------------------------//
// Matrix-vector multiplication
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionX>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value && is_vector<ExpressionX>::value,
    Vector<typename ExpressionA::value_type,ExpressionA::extent_0>>
operator*( const ExpressionA& a, const ExpressionX& x )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionX::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_1 == ExpressionX::extent_0,
                   "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionX::eval_type x_eval = x;
    Vector<typename ExpressionA::value_type,ExpressionA::extent_0> y =
        Kokkos::ArithTraits<typename ExpressionA::value_type>::zero();
    KokkosBatched::SerialGemv<
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::Gemv::Unblocked>::invoke(
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            a_eval,
            x_eval,
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            y );
    return y;
}

//---------------------------------------------------------------------------//
// Vector-matrix multiplication
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionX>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value && is_vector<ExpressionX>::value,
    Matrix<typename ExpressionA::value_type,
           ExpressionX::extent_0,
           ExpressionA::extent_1>>
operator*( const ExpressionX& x, const ExpressionA& a )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionX::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionX::extent_1 == ExpressionA::extent_0,
                   "inner extent must match" );

    typename ExpressionA::eval_type a_eval = a;
    typename ExpressionX::eval_type x_eval = x;
    Matrix<typename ExpressionA::value_type,
           ExpressionX::extent_0,
           ExpressionA::extent_1> y =
        Kokkos::ArithTraits<typename ExpressionA::value_type>::zero();
    KokkosBatched::SerialGemm<
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::Gemm::Unblocked>::invoke(
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            x_eval,
            a_eval,
            Kokkos::ArithTraits<typename ExpressionA::value_type>::one(),
            y );
    return y;
}

//---------------------------------------------------------------------------//
// Vector-vector addition.
//---------------------------------------------------------------------------//
template<class ExpressionX,
         class ExpressionY,
         typename std::enable_if_t<is_vector<ExpressionX>::value &&
                                   is_vector<ExpressionY>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator+( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return createVectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) + y(i); } );
}

//---------------------------------------------------------------------------//
// Vector-vector subtraction.
//---------------------------------------------------------------------------//
template<class ExpressionX,
         class ExpressionY,
         typename std::enable_if_t<is_vector<ExpressionX>::value &&
                                   is_vector<ExpressionY>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator-( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return createVectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) - y(i); } );
}

//---------------------------------------------------------------------------//
// Vector products.
//---------------------------------------------------------------------------//
// Cross product
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    Vector<typename ExpressionX::value_type,3>>
operator%( const ExpressionX& x,
           const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == 3,
                   "cross product is for 3-vectors" );
    static_assert( ExpressionY::extent_0 == 3,
                   "cross product is for 3-vectors" );
    typename ExpressionX::eval_type x_eval = x;
    typename ExpressionY::eval_type y_eval = y;
    return Vector<typename ExpressionX::value_type,3>{
        x_eval(1)*y_eval(2) - x_eval(2)*y_eval(1),
            x_eval(2)*y_eval(0) - x_eval(0)*y_eval(2),
            x_eval(0)*y_eval(1) - x_eval(1)*y_eval(0) };
}

//---------------------------------------------------------------------------//
// Element-wise multiplication.
template<class ExpressionX,
         class ExpressionY,
         typename std::enable_if_t<is_vector<ExpressionX>::value &&
                                   is_vector<ExpressionY>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator&( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent_0 must match" );
    return createVectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) * y(i); } );
}

//---------------------------------------------------------------------------//
// Element-wise division.
template<class ExpressionX,
         class ExpressionY,
         typename std::enable_if_t<is_vector<ExpressionX>::value &&
                                   is_vector<ExpressionY>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator|( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return createVectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) / y(i); } );
}

//---------------------------------------------------------------------------//
// Scalar multiplication.
//---------------------------------------------------------------------------//
// Matrix.
template<class ExpressionA,
         typename std::enable_if_t<is_matrix<ExpressionA>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator*( const typename ExpressionA::value_type s, const ExpressionA& a )
{
    return createMatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){ return s * a(i,j); } );
}

template<class ExpressionA,
         typename std::enable_if_t<is_matrix<ExpressionA>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator*( const ExpressionA& a, const typename ExpressionA::value_type s )
{
    return s * a;
}

//---------------------------------------------------------------------------//
// Vector.
template<class ExpressionX,
         typename std::enable_if_t<is_vector<ExpressionX>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator*( const typename ExpressionX::value_type s, const ExpressionX& x )
{
    return createVectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return s * x(i); } );
}

template<class ExpressionX,
         typename std::enable_if_t<is_vector<ExpressionX>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator*( const ExpressionX& x, const typename ExpressionX::value_type s )
{
    return s * x;
}

//---------------------------------------------------------------------------//
// Scalar division.
//---------------------------------------------------------------------------//
// Matrix.
template<class ExpressionA,
         typename std::enable_if_t<is_matrix<ExpressionA>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator/( const ExpressionA& a, const typename ExpressionA::value_type s )
{
    auto s_inv =
        Kokkos::ArithTraits<typename ExpressionA::value_type>::one() / s;
    return s_inv * a;
}

//---------------------------------------------------------------------------//
// Vector.
template<class ExpressionX,
         typename std::enable_if_t<is_vector<ExpressionX>::value,int> = 0>
KOKKOS_INLINE_FUNCTION
auto
operator/( const ExpressionX& x, const typename ExpressionX::value_type s )
{
    auto s_inv =
        Kokkos::ArithTraits<typename ExpressionX::value_type>::one() / s;
    return s_inv * x;
}

//---------------------------------------------------------------------------//
// Matrix determinants.
//---------------------------------------------------------------------------//
// 2x2 specialization
template<class Expression>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<is_matrix<Expression>::value &&
                          Expression::extent_0 == 2 &&
                          Expression::extent_1 == 2,
                          typename Expression::value_type>
operator!( const Expression& a )
{
    return a(0,0) * a(1,1) - a(0,1) * a(1,0);
}

//---------------------------------------------------------------------------//
// 3x3 specialization
template<class Expression>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<is_matrix<Expression>::value &&
                          Expression::extent_0 == 3 &&
                          Expression::extent_1 == 3,
                          typename Expression::value_type>
operator!( const Expression& a )
{
    return
        a(0,0) * a(1,1) * a(2,2) +
        a(0,1) * a(1,2) * a(2,0) +
        a(0,2) * a(1,0) * a(2,1) -
        a(0,2) * a(1,1) * a(2,0) -
        a(0,1) * a(1,0) * a(2,2) -
        a(0,0) * a(1,2) * a(2,1);
}

//---------------------------------------------------------------------------//
// Linear solve.
//---------------------------------------------------------------------------//
// 2x2 specialization. Single and multiple RHS supported.
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2,
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionB::extent_0 == 2,
                   "extent_0 must be 2" );

    typename ExpressionA::eval_type a_eval = a;

    auto a_det_inv = 1.0 / !a_eval;

    Matrix<typename ExpressionA::value_type,2,2> a_inv;

    a_inv(0,0) = a_eval(1,1) * a_det_inv;
    a_inv(0,1) = -a_eval(0,1) * a_det_inv;
    a_inv(1,0) = -a_eval(1,0) * a_det_inv;
    a_inv(1,1) = a_eval(0,0) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// 3x3 specialization. Single and multiple RHS supported.
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionB::extent_0 == 3,
                   "extent_0 must be 3" );

    typename ExpressionA::eval_type a_eval = a;

    auto a_det_inv = 1.0 / !a_eval;

    Matrix<typename ExpressionA::value_type,3,3> a_inv;

    a_inv(0,0) = (a_eval(1,1)*a_eval(2,2) - a_eval(1,2)*a_eval(2,1)) * a_det_inv;
    a_inv(0,1) = (a_eval(0,2)*a_eval(2,1) - a_eval(0,1)*a_eval(2,2)) * a_det_inv;
    a_inv(0,2) = (a_eval(0,1)*a_eval(1,2) - a_eval(0,2)*a_eval(1,1)) * a_det_inv;

    a_inv(1,0) = (a_eval(1,2)*a_eval(2,0) - a_eval(1,0)*a_eval(2,2)) * a_det_inv;
    a_inv(1,1) = (a_eval(0,0)*a_eval(2,2) - a_eval(0,2)*a_eval(2,0)) * a_det_inv;
    a_inv(1,2) = (a_eval(0,2)*a_eval(1,0) - a_eval(0,0)*a_eval(1,2)) * a_det_inv;

    a_inv(2,0) = (a_eval(1,0)*a_eval(2,1) - a_eval(1,1)*a_eval(2,0)) * a_det_inv;
    a_inv(2,1) = (a_eval(0,1)*a_eval(2,0) - a_eval(0,0)*a_eval(2,1)) * a_det_inv;
    a_inv(2,2) = (a_eval(0,0)*a_eval(1,1) - a_eval(0,1)*a_eval(1,0)) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// General case. Single and multiple RHS supported.
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if_t<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    !(ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2) &&
    !(ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3),
    typename ExpressionB::copy_type>
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_0,
                   "Inner extent must match" );
    auto a_lu = a.LU();
    typename ExpressionB::copy_type x = b;
    KokkosBatched::SerialSolveLU<
        KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::SolveLU::Unblocked>::invoke( a_lu, x );
    return x;
}

//---------------------------------------------------------------------------//

} // end namespace LinearAlgebra

//---------------------------------------------------------------------------//
// Type aliases.
//---------------------------------------------------------------------------//

template<class T>
using Mat2 = LinearAlgebra::Matrix<T,2,2>;
template<class T>
using MatView2 = LinearAlgebra::MatrixView<T,2,2>;
template<class T>
using Vec2 = LinearAlgebra::Vector<T,2>;
template<class T>
using VecView2 = LinearAlgebra::VectorView<T,2>;

template<class T>
using Mat3 = LinearAlgebra::Matrix<T,3,3>;
template<class T>
using MatView3 = LinearAlgebra::MatrixView<T,3,3>;
template<class T>
using Vec3 = LinearAlgebra::Vector<T,3>;
template<class T>
using VecView3 = LinearAlgebra::VectorView<T,3>;

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
