#ifndef PICASSO_BATCHEDLINEARALGEBRA_HPP
#define PICASSO_BATCHEDLINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBatched_Copy_Impl.hpp>

#include <type_traits>

namespace Picasso
{
namespace LinearAlgebra
{
//---------------------------------------------------------------------------//
// Dense matrix in row-major order with a KokkosKernels compatible data
// interface.
template<class T, int M, int N>
struct Matrix
{
    T _d[M][N];
    int _extent[2] = {M,N};

    using value_type = T;
    using pointer = T*;
    using const_pointer = typename std::add_const<T>::type*;
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
    Matrix( const Matrix& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Deep copy assignment operator.
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
        for ( int i = 0; i < M; ++i )
            for ( int j = 0; j < N; ++j )
                _d[i][j] = value;
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
    const_reference operator()( const int i, const int j ) const
    { return _d[i][j]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i, const int j )
    { return _d[i][j]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    const_pointer data() const
    { return &_d[0][0]; }

    KOKKOS_INLINE_FUNCTION
    pointer data()
    { return &_d[0][0]; }
};

//---------------------------------------------------------------------------//
// Dense vector with a KokkosKernels compatible data interface.
template<class T, int N>
class Vector
{
  private:

    T _d[N];
    int _extent[2] = {N,1};

  public:

    using value_type = T;
    using pointer = T*;
    using const_pointer = typename std::add_const<T>::type*;
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
    Vector( const Vector& rhs )
    {
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose>::invoke(
            rhs, *this );
    }

    // Deep copy assignment operator.
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
        for ( int i = 0; i < N; ++i )
            _d[i] = value;
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
    const_reference operator()( const int i ) const
    { return _d[i]; }

    KOKKOS_INLINE_FUNCTION
    reference operator()( const int i )
    { return _d[i]; }

    // Get the raw data.
    KOKKOS_INLINE_FUNCTION
    const_pointer data() const
    { return &_d[0]; }

    KOKKOS_INLINE_FUNCTION
    pointer data()
    { return &_d[0]; }
};
//---------------------------------------------------------------------------//

} // end namespace LinearAlgebra
} // end namespace Picasso

#endif // end PICASSO_BATCHEDLINEARALGEBRA_HPP
