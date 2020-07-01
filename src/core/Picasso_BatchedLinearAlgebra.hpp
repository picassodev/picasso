#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>

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
// Forward declarations.
//---------------------------------------------------------------------------//
template<class T, int M, int N> struct MatrixExpression;
template<class T, int M, int N> struct Matrix;
template<class T, int M, int N> struct MatrixView;
template<class T, int N> struct VectorExpression;
template<class T, int N> struct Vector;
template<class T, int N> struct VectorView;

//---------------------------------------------------------------------------//
// Type traits.
//---------------------------------------------------------------------------//
// Matrix
template<class>
struct is_matrix_impl : public std::false_type
{};

template<class T, int M, int N>
struct is_matrix_impl<MatrixExpression<T,M,N>> : public std::true_type
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

template<class T, int N>
struct is_vector_impl<VectorExpression<T,N>> : public std::true_type
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
// Expressions containers.
//---------------------------------------------------------------------------//
// Matrix expression container.
template<class T, int M, int N>
struct MatrixExpression
{
    static constexpr int extent_0 = M;
    static constexpr int extent_1 = N;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Matrix<T,M,N>;
    using copy_type = Matrix<T,M,N>;

    std::function<value_type(int,int)> _f;
    int _extent[2] = {M,N};

    // Create an expression from a callable object.
    template<class Func>
    KOKKOS_INLINE_FUNCTION
    MatrixExpression( const Func& f )
        : _f( f )
    {}

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Evaluate the expression at an index.
    KOKKOS_INLINE_FUNCTION
    value_type operator()( const int i, const int j ) const
    { return _f(i,j); }

    // Copy/eval conversion operator. Triggers an expression evaluation.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    {
        eval_type eval;
        for ( int i = 0; i < M; ++i )
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
                eval(i,j) = _f(i,j);
        return eval;
    }

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

//---------------------------------------------------------------------------//
// Vector expression container.
template<class T, int N>
struct VectorExpression
{
    static constexpr int extent_0 = N;
    static constexpr int extent_1 = 1;

    using value_type = T;
    using non_const_value_type = typename std::remove_cv<T>::type;

    using eval_type = Vector<T,N>;
    using copy_type = Vector<T,N>;

    std::function<value_type(int)> _f;

     // Create an expression from a callable object.
    template<class Func>
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

    // Copy/eval conversion operator. Triggers an expression evaluation.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    {
        eval_type eval;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int i = 0; i < N; ++i )
                eval(i) = _f(i);
        return eval;
    }
};

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

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
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

    // Conversion operator. Evaluation is a shallow copy.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    { return eval_type( this->data(), N, 1 ); }

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

    // Conversion operator. Evaluation is a shallow copy.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    { return eval_type( this->data(), 1, 1 ); }

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

    // Deep copy conversion operator.
    KOKKOS_INLINE_FUNCTION
    operator copy_type() const
    {
        copy_type copy;
        for ( int i = 0; i < M; ++i )
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
            for ( int j = 0; j < N; ++j )
                copy(i,j) = (*this)(i,j);
        return copy;
    }

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

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const Vector& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const Vector& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
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

    // Conversion operator. Evaluation is a shallow copy.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    { return eval_type( this->data(), 1 ); }
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

    // Conversion operator. Evaluation is a shallow copy.
    KOKKOS_INLINE_FUNCTION
    operator eval_type() const
    { return eval_type( this->data(), 1 ); }

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

    // Deep copy conversion operator.
    KOKKOS_INLINE_FUNCTION
    operator copy_type() const
    {
        copy_type copy;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for ( int i = 0; i < N; ++i )
            copy(i) = (*this)(i);
        return copy;
    }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }
};

//---------------------------------------------------------------------------//
// Transpose.
//---------------------------------------------------------------------------//
// Matrix operator.
template<class Expression>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<Expression>::value,
    MatrixExpression<typename Expression::value_type,
                     Expression::extent_1,
                     Expression::extent_0>>::type
operator~( const Expression& e )
{
    return MatrixExpression<
        typename Expression::value_type,
        Expression::extent_1,
        Expression::extent_0>(
            [=]( const int i, const int j ){ return e(j,i); } );
}

//---------------------------------------------------------------------------//
// Vector operator.
template<class Expression>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<Expression>::value,
    MatrixExpression<typename Expression::value_type,1,Expression::extent_0>
    >::type
operator~( const Expression& e )
{
    return MatrixExpression<
        typename Expression::value_type,1,Expression::extent_0>(
            [=]( const int, const int j ){ return e(j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix addition.
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value && is_matrix<ExpressionB>::value,
    MatrixExpression<typename ExpressionA::value_type,
                     ExpressionA::extent_0,
                     ExpressionA::extent_1>>::type
operator+( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must match" );
    return MatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){ return a(i,j) + b(i,j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix subtraction.
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value && is_matrix<ExpressionB>::value,
    MatrixExpression<typename ExpressionA::value_type,
                     ExpressionA::extent_0,
                     ExpressionB::extent_1>>::type
operator-( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionA::value_type>::value,
                   "value_type must match" );
    static_assert( ExpressionA::extent_0 == ExpressionB::extent_0,
                   "extent_0 must match" );
    static_assert( ExpressionA::extent_1 == ExpressionB::extent_1,
                   "extent_1 must_match" );
    return MatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){ return a(i,j) - b(i,j); } );
}

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication.
//---------------------------------------------------------------------------//
template<class ExpressionA, class ExpressionB>
typename std::enable_if<
    is_matrix<ExpressionA>::value && is_matrix<ExpressionB>::value,
    Matrix<typename ExpressionA::value_type,
           ExpressionA::extent_0,ExpressionB::extent_1>>::type
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
typename std::enable_if<
    is_matrix<ExpressionA>::value && is_vector<ExpressionX>::value,
    Vector<typename ExpressionA::value_type,ExpressionA::extent_0>>::type
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
typename std::enable_if<
    is_matrix<ExpressionA>::value && is_vector<ExpressionX>::value,
    Matrix<typename ExpressionA::value_type,
           ExpressionX::extent_0,
           ExpressionA::extent_1>>::type
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
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator+( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return VectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) + y(i); } );
}

//---------------------------------------------------------------------------//
// Vector-vector subtraction.
//---------------------------------------------------------------------------//
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator-( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return VectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) - y(i); } );
}

//---------------------------------------------------------------------------//
// Vector products.
//---------------------------------------------------------------------------//
// Cross product
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    Vector<typename ExpressionX::value_type,3>>::type
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
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator&( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent_0 must match" );
    return VectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) * y(i); } );
}

//---------------------------------------------------------------------------//
// Element-wise division.
template<class ExpressionX, class ExpressionY>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value && is_vector<ExpressionY>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator|( const ExpressionX& x, const ExpressionY& y )
{
    static_assert( std::is_same<typename ExpressionX::value_type,
                   typename ExpressionY::value_type>::value,
                   "value_type must match");
    static_assert( ExpressionX::extent_0 == ExpressionY::extent_0,
                   "extent must match" );
    return VectorExpression<
        typename ExpressionX::value_type,ExpressionX::extent_0>(
            [=]( const int i ){ return x(i) / y(i); } );
}

//---------------------------------------------------------------------------//
// Scalar multiplication.
//---------------------------------------------------------------------------//
// Matrix.
template<class ExpressionA>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value,
    MatrixExpression<typename ExpressionA::value_type,
                     ExpressionA::extent_0,
                     ExpressionA::extent_1>>::type
operator*( const typename ExpressionA::value_type s, const ExpressionA& a )
{
    return MatrixExpression<
        typename ExpressionA::value_type,
        ExpressionA::extent_0,
        ExpressionA::extent_1>(
            [=]( const int i, const int j ){
                return s * a(i,j);
            } );
}

template<class ExpressionA>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value,
    MatrixExpression<typename ExpressionA::value_type,
                     ExpressionA::extent_0,
                     ExpressionA::extent_1>>::type
operator*( const ExpressionA& a, const typename ExpressionA::value_type s )
{
    return s * a;
}

//---------------------------------------------------------------------------//
// Vector.
template<class ExpressionX>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator*( const typename ExpressionX::value_type s, const ExpressionX& x )
{
    return VectorExpression<
        typename ExpressionX::value_type,
        ExpressionX::extent_0>(
            [=]( const int i ){
                return s * x(i);
            } );
}

template<class ExpressionX>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
operator*( const ExpressionX& x, const typename ExpressionX::value_type s )
{
    return s * x;
}

//---------------------------------------------------------------------------//
// Scalar division.
//---------------------------------------------------------------------------//
// Matrix.
template<class ExpressionA>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value,
    MatrixExpression<typename ExpressionA::value_type,
                     ExpressionA::extent_0,
                     ExpressionA::extent_1>>::type
operator/( const ExpressionA& a, const typename ExpressionA::value_type s )
{
    auto s_inv =
        Kokkos::ArithTraits<typename ExpressionA::value_type>::one() / s;
    return s_inv * a;
}

//---------------------------------------------------------------------------//
// Vector.
template<class ExpressionX>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_vector<ExpressionX>::value,
    VectorExpression<typename ExpressionX::value_type,
                     ExpressionX::extent_0>>::type
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
typename std::enable_if<
    is_matrix<Expression>::value &&
    Expression::extent_0 == 2 &&
    Expression::extent_1 == 2,
    typename Expression::value_type>::type
operator!( const Expression& a )
{
    return a(0,0) * a(1,1) - a(0,1) * a(1,0);
}

//---------------------------------------------------------------------------//
// 3x3 specialization
template<class Expression>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<Expression>::value &&
    Expression::extent_0 == 3 &&
    Expression::extent_1 == 3,
    typename Expression::value_type>::type
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
typename std::enable_if<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2,
    typename ExpressionB::copy_type>::type
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionB::extent_0 == 2,
                   "extent_0 must be 2" );

    auto a_det_inv = 1.0 / !a;

    Matrix<typename ExpressionA::value_type,2,2> a_inv;

    a_inv(0,0) = a(1,1) * a_det_inv;
    a_inv(0,1) = -a(0,1) * a_det_inv;
    a_inv(1,0) = -a(1,0) * a_det_inv;
    a_inv(1,1) = a(0,0) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// 3x3 specialization. Single and multiple RHS supported.
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3,
    typename ExpressionB::copy_type>::type
operator^( const ExpressionA& a, const ExpressionB& b )
{
    static_assert( std::is_same<typename ExpressionA::value_type,
                   typename ExpressionB::value_type>::value,
                   "value_type must be the same" );
    static_assert( ExpressionB::extent_0 == 3,
                   "extent_0 must be 3" );

    auto a_det_inv = 1.0 / !a;

    Matrix<typename ExpressionA::value_type,3,3> a_inv;

    a_inv(0,0) = (a(1,1)*a(2,2) - a(1,2)*a(2,1)) * a_det_inv;
    a_inv(0,1) = (a(0,2)*a(2,1) - a(0,1)*a(2,2)) * a_det_inv;
    a_inv(0,2) = (a(0,1)*a(1,2) - a(0,2)*a(1,1)) * a_det_inv;

    a_inv(1,0) = (a(1,2)*a(2,0) - a(1,0)*a(2,2)) * a_det_inv;
    a_inv(1,1) = (a(0,0)*a(2,2) - a(0,2)*a(2,0)) * a_det_inv;
    a_inv(1,2) = (a(0,2)*a(1,0) - a(0,0)*a(1,2)) * a_det_inv;

    a_inv(2,0) = (a(1,0)*a(2,1) - a(1,1)*a(2,0)) * a_det_inv;
    a_inv(2,1) = (a(0,1)*a(2,0) - a(0,0)*a(2,1)) * a_det_inv;
    a_inv(2,2) = (a(0,0)*a(1,1) - a(0,1)*a(1,0)) * a_det_inv;

    return a_inv * b;
}

//---------------------------------------------------------------------------//
// General case. Single and multiple RHS supported.
template<class ExpressionA, class ExpressionB>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    is_matrix<ExpressionA>::value &&
    (is_matrix<ExpressionB>::value || is_vector<ExpressionB>::value) &&
    !(ExpressionA::extent_0 == 2 && ExpressionA::extent_1 == 2) &&
    !(ExpressionA::extent_0 == 3 && ExpressionA::extent_1 == 3),
    typename ExpressionB::copy_type>::type
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
