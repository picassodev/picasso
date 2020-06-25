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

namespace Picasso
{
namespace LinearAlgebra
{
//---------------------------------------------------------------------------//
// Transpose tags.
struct NoTranspose
{
    using type = KokkosBatched::Trans::NoTranspose;
};

struct Transpose
{
    using type = KokkosBatched::Trans::Transpose;
};

//---------------------------------------------------------------------------//
// Tags for Copy vs. View semantics.
struct Copy
{};

struct View
{};

//---------------------------------------------------------------------------//
// Matrix
//---------------------------------------------------------------------------//
// Dense matrix in row-major order with a KokkosKernels compatible data
// interface.
template<class T,
         int M,
         int N,
         class TransposeType = NoTranspose,
         class Memory = Copy>
struct Matrix;

//---------------------------------------------------------------------------//
// No transpose. Copy semantics.
template<class T, int M, int N>
struct Matrix<T,M,N,NoTranspose,Copy>
{
    T _d[M][N];
    int _extent[2] = {M,N};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

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
    template<class Memory>
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,M,N,NoTranspose,Memory>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
    }

    // Deep copy transpose constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,N,M,Transpose,View>& rhs )
    {
        KokkosBatched::SerialCopy<Transpose::type>::invoke(
            rhs, *this );
    }

    // Scalar constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
    }

    // Deep copy assignment operator.
    template<class Memory>
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,NoTranspose,Memory>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
        return *this;
    }

    // Deep copy transpose assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,Transpose,View>& rhs )
    {
        KokkosBatched::SerialCopy<Transpose::type>::invoke(
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

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,M,N,Transpose,View> operator~()
    {
        return Matrix<T,M,N,Transpose,View>( this->data(), N, 1 );
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
    const_reference operator()( const int i, const int j ) const
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
    Matrix<T,N,M,NoTranspose,Copy> LU() const
    {
        Matrix<T,N,M,NoTranspose,Copy> lu = *this;
        KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke( lu );
        return lu;
    }
};

//---------------------------------------------------------------------------//
// No transpose. View semantics.
template<class T, int M, int N>
struct Matrix<T,M,N,NoTranspose,View>
{
    T *_d;
    int _extent[2] = {M,N};
    int _stride[2];

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Matrix() = default;

    // View constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( T* data, const int stride_0, const int stride_1 )
        : _d ( data )
    {
        _stride[0] = stride_0;
        _stride[1] = stride_1;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,M,N,Transpose,View> operator~()
    {
        return Matrix<T,M,N,Transpose,View>( this->data(), _stride[0], _stride[1] );
    }

    // Strides.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return _stride[0]; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return _stride[1]; }

    // Extent
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j ) const
    { return _d[i*_stride[0] + j*_stride[1]]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[i*_stride[0] + j*_stride[1]]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }

    // LU decomposition.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,N,M,NoTranspose,Copy> LU() const
    {
        Matrix<T,N,M,NoTranspose,Copy> lu = *this;
        KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke( lu );
        return lu;
    }
};

//---------------------------------------------------------------------------//
// Transpose. This class is essentially a shallow-copy placeholder to enable
// transpose matrix operations without copies in Kokkos-kernels operations as
// well as other implementation details. Tranpose always has view semantics.
template<class T, int M, int N>
struct Matrix<T,M,N,Transpose,View>
{
    T* _d;
    int _extent[2] = {M,N};
    int _stride[2];

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // View constructor.
    Matrix( T* data, const int stride_0, const int stride_1 )
        : _d ( data )
    {
        _stride[0] = stride_0;
        _stride[1] = stride_1;
    }

    // Strides. Written in terms of the original non-transpose matrix for
    // compatibility with Kokkos-kernels operations.
    KOKKOS_INLINE_FUNCTION
    int stride_0() const
    { return _stride[0]; }

    KOKKOS_INLINE_FUNCTION
    int stride_1() const
    { return _stride[1]; }

    // Extent. Written in terms of the original non-transpose matrix for
    // compatibility with Kokkos-kernels operations.
    KOKKOS_INLINE_FUNCTION
    int extent( const int d ) const
    { return _extent[d]; }

    // Access an individual element. Access is designed as if this was
    // actually holding the transposed data to facilitate implementation
    // details.
    KOKKOS_INLINE_FUNCTION
    const_reference operator()( const int i, const int j ) const
    { return _d[i*_stride[1] + j*_stride[0]]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[i*_stride[1] + j*_stride[0]]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }

    // LU decomposition. This is the decomposition of the transposed
    // operator.
    KOKKOS_INLINE_FUNCTION
    Matrix<T,N,M,NoTranspose,Copy> LU() const
    {
        Matrix<T,N,M,NoTranspose,Copy> lu = *this;
        KokkosBatched::SerialLU<KokkosBatched::Algo::LU::Unblocked>::invoke( lu );
        return lu;
    }
};

//---------------------------------------------------------------------------//
// Vector
//---------------------------------------------------------------------------//
// Dense vector with a KokkosKernels compatible data interface.
template<class T,
         int N,
         class TransposeType = NoTranspose,
         class Memory = Copy>
struct Vector;

//---------------------------------------------------------------------------//
// No transpose. Copy semantics.
template<class T, int N>
struct Vector<T,N,NoTranspose,Copy>
{
    T _d[N];
    int _extent[2] = {N,1};

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

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
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke( rhs, *this );
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
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke( rhs, *this );
        return *this;
    }

    // Scalar value assignment.
    KOKKOS_INLINE_FUNCTION
    Vector& operator=( const T value )
    {
        KokkosBatched::SerialSet::invoke( value, *this );
        return *this;
    }

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Vector<T,N,Transpose,View> operator~()
    {
        return Vector<T,N,Transpose,View>( this->data(), 1 );
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
    const_reference operator()( const int i ) const
    { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }

    // Euclidean norm.
    T norm2() const
    {
        T n2 = Kokkos::ArithTraits<T>::zero();
        for ( int i = 0; i < N; ++i )
            n2 += _d[i]*_d[i];
        return sqrt(n2);
    }
};
//---------------------------------------------------------------------------//
// No transpose. View semantics.
template<class T, int N>
struct Vector<T,N,NoTranspose,View>
{
    T* _d;
    int _extent[2] = {N,1};
    int _stride;

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Vector() = default;

    // View constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( T* data, const int stride )
        : _d( data )
        , _stride( stride )
    {}

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Vector<T,N,Transpose,View> operator~()
    {
        return Vector<T,N,Transpose,View>( this->data() );
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
    const_reference operator()( const int i ) const
    { return _d[i*_stride]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[i*_stride]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }

    // Euclidean norm.
    T norm2() const
    {
        T n2 = Kokkos::ArithTraits<T>::zero();
        for ( int i = 0; i < N; ++i )
            n2 += _d[i*_stride]*_d[i*_stride];
        return sqrt(n2);
    }
};

//---------------------------------------------------------------------------//
// Transpose. This class is essentially a shallow-copy placeholder to enable
// transpose vector operations without copies. Transpose always has view
// semantics.
template<class T, int N>
struct Vector<T,N,Transpose,View>
{
    T* _d;
    int _extent[2] = {N,1};
    int _stride;

    using value_type = T;
    using non_const_value_type = typename std::remove_const<T>::type;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( T* data, const int stride )
        : _d(data)
        , _stride( stride )
    {}

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

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }
};

//---------------------------------------------------------------------------//
// Matrix-matrix addition.
//---------------------------------------------------------------------------//
// No transpose case.
template<class T, int M, int N, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator+( const Matrix<T,M,N,NoTranspose,MemoryA>& a,
           const Matrix<T,M,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) + b(i,j);
    return c;
}

//---------------------------------------------------------------------------//
// No transpose - transpose case.
template<class T, int M, int N, class MemoryA>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator+( const Matrix<T,M,N,NoTranspose,MemoryA>& a,
           const Matrix<T,N,M,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) + b(i,j);
    return c;
}

//---------------------------------------------------------------------------//
// Transpose - no transpose case.
template<class T, int M, int N, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator+( const Matrix<T,N,M,Transpose,View>& a,
           const Matrix<T,M,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) + b(i,j);
    return c;
}

//---------------------------------------------------------------------------//
// Transpose - transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator+( const Matrix<T,N,M,Transpose,View>& a,
           const Matrix<T,N,M,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) + b(i,j);
    return c;
}

//---------------------------------------------------------------------------//
// Matrix-matrix subtraction.
//---------------------------------------------------------------------------//
// No transpose case.
template<class T, int M, int N, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator-( const Matrix<T,M,N,NoTranspose,MemoryA>& a,
           const Matrix<T,M,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) - b(i,j);
    return c;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-//
// No transpose - transpose case.
template<class T, int M, int N, class MemoryA>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator-( const Matrix<T,M,N,NoTranspose,MemoryA>& a,
           const Matrix<T,N,M,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) - b(i,j);
    return c;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-//
// Transpose - no transpose case.
template<class T, int M, int N, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator-( const Matrix<T,N,M,Transpose,View>& a,
           const Matrix<T,M,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) - b(i,j);
    return c;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-//
// Transpose - transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator-( const Matrix<T,N,M,Transpose,View>& a,
           const Matrix<T,N,M,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < N; ++j )
            c(i,j) = a(i,j) - b(i,j);
    return c;
}

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication
//---------------------------------------------------------------------------//
// No transpose case.
template<class T, int M, int N, int K, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator*( const Matrix<T,M,K,NoTranspose,MemoryA>& a,
           const Matrix<T,K,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator*( const Matrix<T,K,M,Transpose,View>& a,
           const Matrix<T,N,K,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// NoTranspose-Transpose case.
template<class T, int M, int N, int K, class MemoryA>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator*( const Matrix<T,M,K,NoTranspose,MemoryA>& a,
           const Matrix<T,N,K,Transpose,View>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose-NoTranspose case.
template<class T, int M, int N, int K, class MemoryB>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator*( const Matrix<T,K,M,Transpose,View>& a,
           const Matrix<T,K,N,NoTranspose,MemoryB>& b )
{
    Matrix<T,M,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Matrix-vector multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N, class MemoryA, class MemoryX>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose,Copy>
operator*( const Matrix<T,M,N,NoTranspose,MemoryA>& a,
           const Vector<T,N,NoTranspose,MemoryX>& x )
{
    Vector<T,M,NoTranspose,Copy> y;
    KokkosBatched::SerialGemv<NoTranspose::type,
                              KokkosBatched::Algo::Gemv::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, x, Kokkos::ArithTraits<T>::one(), y );
    return y;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N, class MemoryX>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose,Copy>
operator*( const Matrix<T,N,M,Transpose,View>& a,
           const Vector<T,N,NoTranspose,MemoryX>& x )
{
    Vector<T,M,NoTranspose,Copy> y;
    KokkosBatched::SerialGemv<Transpose::type,
                              KokkosBatched::Algo::Gemv::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, x, Kokkos::ArithTraits<T>::one(), y );
    return y;
}

//---------------------------------------------------------------------------//
// Vector-matrix multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N, class MemoryA>
KOKKOS_INLINE_FUNCTION
Matrix<T,1,N,NoTranspose,Copy>
operator*( const Vector<T,M,Transpose,View>& x,
           const Matrix<T,M,N,NoTranspose,MemoryA>& a )
{
    Matrix<T,1,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              NoTranspose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, a, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,1,M,NoTranspose,Copy>
operator*( const Vector<T,N,Transpose,View>& x,
           const Matrix<T,M,N,Transpose,View>& a )
{
    Matrix<T,1,M,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, a, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Vector-vector addition.
//---------------------------------------------------------------------------//
template<class T, int N, class MemoryX, class MemoryY>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator+( const Vector<T,N,NoTranspose,MemoryX>& x,
           const Vector<T,N,NoTranspose,MemoryY>& y )
{
    Vector<T,N,NoTranspose,Copy> z;
    for ( int i = 0; i < N; ++i )
        z(i) = x(i) + y(i);
    return z;
}

//---------------------------------------------------------------------------//
// Vector-vector subtraction.
//---------------------------------------------------------------------------//
template<class T, int N, class MemoryX, class MemoryY>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator-( const Vector<T,N,NoTranspose,MemoryX>& x,
           const Vector<T,N,NoTranspose,MemoryY>& y )
{
    Vector<T,N,NoTranspose,Copy> z;
    for ( int i = 0; i < N; ++i )
        z(i) = x(i) - y(i);
    return z;
}

//---------------------------------------------------------------------------//
// Vector products.
//---------------------------------------------------------------------------//
// Dot product.
template<class T, int N, class MemoryY>
KOKKOS_INLINE_FUNCTION
T
operator*( const Vector<T,N,Transpose,View>& x,
           const Vector<T,N,NoTranspose,MemoryY>& y )
{
    auto v = Kokkos::ArithTraits<T>::zero();
    KokkosBatched::InnerMultipleDotProduct<1> dp( 0, 1, 1, 1 );
    dp.serial_invoke(  Kokkos::ArithTraits<T>::one(),
                       x.data(), y.data(), N, &v );
    return v;
}

//---------------------------------------------------------------------------//
// Inner product.
template<class T, int N, class MemoryX>
KOKKOS_INLINE_FUNCTION
Matrix<T,N,N,NoTranspose,Copy>
operator*( const Vector<T,N,NoTranspose,MemoryX>& x,
           const Vector<T,N,Transpose,View>& y )
{
    Matrix<T,N,N,NoTranspose,Copy> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, y, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Cross product
template<class T, class MemoryX, class MemoryY>
KOKKOS_INLINE_FUNCTION
Vector<T,3,NoTranspose,Copy>
operator%( const Vector<T,3,NoTranspose,MemoryX>& x,
           const Vector<T,3,NoTranspose,MemoryY>& y )
{
    Vector<T,3,NoTranspose,Copy> z = { x(1)*y(2) - x(2)*y(1),
                                       x(2)*y(0) - x(0)*y(2),
                                       x(0)*y(1) - x(1)*y(0) };
    return z;
}

//---------------------------------------------------------------------------//
// Element-wise multiplication.
template<class T, int N, class MemoryX, class MemoryY>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator&( const Vector<T,N,NoTranspose,MemoryX>& x,
           const Vector<T,N,NoTranspose,MemoryY>& y )
{
    Vector<T,N,NoTranspose,Copy> z;
    for ( int i = 0; i < N; ++i )
        z(i) = x(i) * y(i);
    return z;
}

//---------------------------------------------------------------------------//
// Element-wise division.
template<class T, int N, class MemoryX, class MemoryY>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator|( const Vector<T,N,NoTranspose,MemoryX>& x,
           const Vector<T,N,NoTranspose,MemoryY>& y )
{
    Vector<T,N,NoTranspose,Copy> z;
    for ( int i = 0; i < N; ++i )
        z(i) = x(i) / y(i);
    return z;
}

//---------------------------------------------------------------------------//
// Scalar multiplication.
//---------------------------------------------------------------------------//
// Matrix. No Transpose.
template<class T, int M, int N, class MemoryA>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose,Copy>
operator*( const T v, const Matrix<T,M,N,NoTranspose,MemoryA>& a )
{
    Matrix<T,M,N,NoTranspose,Copy> b = a;
    KokkosBatched::SerialScale::invoke( v, b );
    return b;
}

//---------------------------------------------------------------------------//
// Vector. No transpose.
template<class T, int N, class MemoryX>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator*( const T v, const Vector<T,N,NoTranspose,MemoryX>& x )
{
    Vector<T,N,NoTranspose,Copy> y = x;
    KokkosBatched::SerialScale::invoke( v, y );
    return y;
}

//---------------------------------------------------------------------------//
// Matrix determinants.
//---------------------------------------------------------------------------//
// 2x2 specialization
template<class T, class Trans, class Memory>
KOKKOS_INLINE_FUNCTION
T
operator!( const Matrix<T,2,2,Trans,Memory>& a )
{
    return a(0,0) * a(1,1) - a(0,1) * a(1,0);
}

//---------------------------------------------------------------------------//
// 3x3 specialization
template<class T, class Trans, class Memory>
KOKKOS_INLINE_FUNCTION
T
operator!( const Matrix<T,3,3,Trans,Memory>& a )
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
// General case.
template<class T, int N, class TransA, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Vector<T,N,NoTranspose,Copy>
operator^( const Matrix<T,N,N,TransA,MemoryA>& a,
           const Vector<T,N,NoTranspose,MemoryB>& b )
{
    auto a_lu = a.LU();
    auto x = b;
    KokkosBatched::SerialSolveLU<
        NoTranspose::type,
        KokkosBatched::Algo::SolveLU::Unblocked>::invoke( a_lu, x );
    return x;
}

//---------------------------------------------------------------------------//
// 2x2 specialization.
template<class T, class TransA, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Vector<T,2,NoTranspose,Copy>
operator^( const Matrix<T,2,2,TransA,MemoryA>& a,
           const Vector<T,2,NoTranspose,MemoryB>& b )
{
    auto a_det_inv = 1.0 / !a;

    Matrix<T,2,2,NoTranspose,Copy> a_inv;

    a_inv(0,0) = a(1,1) * a_det_inv;
    a_inv(0,1) = -a(0,1) * a_det_inv;
    a_inv(1,0) = -a(1,0) * a_det_inv;
    a_inv(1,1) = a(0,0) * a_det_inv;

    return a_inv * b;
}
//---------------------------------------------------------------------------//
// 3x3 specialization.
template<class T, class TransA, class MemoryA, class MemoryB>
KOKKOS_INLINE_FUNCTION
Vector<T,3,NoTranspose,Copy>
operator^( const Matrix<T,3,3,TransA,MemoryA>& a,
           const Vector<T,3,NoTranspose,MemoryB>& b )
{
    auto a_det_inv = 1.0 / !a;

    Matrix<T,3,3,NoTranspose> a_inv;

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

} // end namespace LinearAlgebra

//---------------------------------------------------------------------------//
// Type aliases.
//---------------------------------------------------------------------------//

template<class T>
using Vec2 =
    LinearAlgebra::Vector<T,2,LinearAlgebra::NoTranspose,LinearAlgebra::Copy>;

template<class T>
using Mat2 =
    LinearAlgebra::Matrix<T,2,2,LinearAlgebra::NoTranspose,LinearAlgebra::Copy>;

template<class T>
using Vec3 =
    LinearAlgebra::Vector<T,3,LinearAlgebra::NoTranspose,LinearAlgebra::Copy>;

template<class T>
using Mat3 =
    LinearAlgebra::Matrix<T,3,3,LinearAlgebra::NoTranspose,LinearAlgebra::Copy>;

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
