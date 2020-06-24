#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <Kokkos_ArithTraits.hpp>

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>

#include <KokkosBatched_Set_Decl.hpp>
#include <KokkosBatched_Set_Impl.hpp>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>

#include <type_traits>

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
// Matrix
//---------------------------------------------------------------------------//
// Dense matrix in row-major order with a KokkosKernels compatible data
// interface.
template<class T, int M, int N, class TransposeType = NoTranspose>
struct Matrix;

// No transpose
template<class T, int M, int N>
struct Matrix<T,M,N,NoTranspose>
{
    T _d[M][N];
    int _extent[2] = {M,N};

    using value_type = T;
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
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,M,N,NoTranspose>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
    }

    // Deep copy transpose constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix<T,N,M,Transpose>& rhs )
    {
        KokkosBatched::SerialCopy<Transpose::type>::invoke(
            rhs, *this );
    }

    // Deep copy assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,NoTranspose>& rhs )
    {
        KokkosBatched::SerialCopy<NoTranspose::type>::invoke(
            rhs, *this );
        return *this;
    }

    // Deep copy transpose assignment operator.
    KOKKOS_INLINE_FUNCTION
    Matrix& operator=( const Matrix<T,M,N,Transpose>& rhs )
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
    Matrix<T,M,N,Transpose> operator~()
    {
        return Matrix<T,M,N,Transpose>( this->data() );
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
};

// Transpose. This class is essentially a shallow-copy placeholder to enable
// tranpose matrix operations without copies.
template<class T, int M, int N>
struct Matrix<T,M,N,Transpose>
{
    T* _d;
    int _extent[2] = {M,N};

    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( T* data )
        : _d( data )
    {}

    // Deep copy constructor.
    KOKKOS_INLINE_FUNCTION
    Matrix( const Matrix& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
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

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(_d); }
};

//---------------------------------------------------------------------------//
// Vector
//---------------------------------------------------------------------------//
// Dense vector with a KokkosKernels compatible data interface.
template<class T, int N, class TransposeType = NoTranspose>
struct Vector;

// No tranpose
template<class T, int N>
struct Vector<T,N,NoTranspose>
{
    T _d[N];
    int _extent[2] = {N,1};

    using value_type = T;
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
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
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

    // Transpose operator.
    KOKKOS_INLINE_FUNCTION
    Vector<T,N,Transpose> operator~()
    {
        return Vector<T,N,Transpose>( this->data() );
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
};

// Transpose. This class is essentially a shallow-copy placeholder to enable
// tranpose vector operations without copies.
template<class T, int N>
struct Vector<T,N,Transpose>
{
    T* _d;
    int _extent[2] = {N,1};

    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = typename std::add_const<T>::type&;

    // Pointer constructor.
    KOKKOS_INLINE_FUNCTION
    Vector( T* data )
        : _d(data)
    {}

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

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    pointer data() const
    { return const_cast<pointer>(&_d[0]); }
};

//---------------------------------------------------------------------------//
// Matrix-matrix multiplication
//---------------------------------------------------------------------------//
// NoTranspose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,M,K,NoTranspose>& a, const Matrix<T,K,N,NoTranspose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
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
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,K,M,Transpose>& a, const Matrix<T,N,K,Transpose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<Transpose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// NoTranspose-Transpose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,M,K,NoTranspose>& a, const Matrix<T,N,K,Transpose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, b, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//
// Transpose-NoTranspose case.
template<class T, int M, int N, int K>
KOKKOS_INLINE_FUNCTION
Matrix<T,M,N,NoTranspose>
operator*( const Matrix<T,K,M,Transpose>& a, const Matrix<T,K,N,NoTranspose>& b )
{
    Matrix<T,M,N,NoTranspose> c;
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
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose>
operator*( const Matrix<T,M,N,NoTranspose>& a, const Vector<T,N,NoTranspose>& x )
{
    Vector<T,M,NoTranspose> y;
    KokkosBatched::SerialGemv<NoTranspose::type,
                              KokkosBatched::Algo::Gemv::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  a, x, Kokkos::ArithTraits<T>::one(), y );
    return y;
}

//---------------------------------------------------------------------------//
// Transpose case.
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Vector<T,M,NoTranspose>
operator*( const Matrix<T,N,M,Transpose>& a, const Vector<T,N,NoTranspose>& x )
{
    Vector<T,M,NoTranspose> y;
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
template<class T, int M, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,1,N,NoTranspose>
operator*( const Vector<T,M,Transpose>& x, const Matrix<T,M,N,NoTranspose>& a )
{
    Matrix<T,1,N,NoTranspose> c;
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
Matrix<T,1,M,NoTranspose>
operator*( const Vector<T,N,Transpose>& x, const Matrix<T,M,N,Transpose>& a )
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
// Vector-vector multiplication.
//---------------------------------------------------------------------------//
// Dot product.
template<class T, int N>
KOKKOS_INLINE_FUNCTION
T operator*( const Vector<T,N,Transpose>& x, const Vector<T,N,NoTranspose>& y )
{
    auto v = Kokkos::ArithTraits<T>::zero();
    KokkosBatched::InnerMultipleDotProduct<1> dp( 0, 1, 1, 1 );
    dp.serial_invoke(  Kokkos::ArithTraits<T>::one(),
                       x.data(), y.data(), N, &v );
    return v;
}

//---------------------------------------------------------------------------//
// Inner product.
template<class T, int N>
KOKKOS_INLINE_FUNCTION
Matrix<T,N,N,NoTranspose>
operator*( const Vector<T,N,NoTranspose>& x, const Vector<T,N,Transpose>& y )
{
    Matrix<T,N,N,NoTranspose> c;
    KokkosBatched::SerialGemm<NoTranspose::type,
                              Transpose::type,
                              KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                                  Kokkos::ArithTraits<T>::one(),
                                  x, y, Kokkos::ArithTraits<T>::one(), c );
    return c;
}

//---------------------------------------------------------------------------//

} // end namespace LinearAlgebra
} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
