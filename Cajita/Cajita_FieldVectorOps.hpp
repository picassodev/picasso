#ifndef CAJITA_GRIDFIELDVECTOROPS_HPP
#define CAJITA_GRIDFIELDVECTOROPS_HPP

#include <Cajita_Field.hpp>
#include <Cajita_GridExecPolicy.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_MpiTraits.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <type_traits>
#include <cmath>

namespace Cajita
{
namespace FieldVectorOps
{
//---------------------------------------------------------------------------//
// Calculate the infinity-norm of a vector. (Scalar fields only)
template<class FieldType>
typename FieldType::value_type normInf( const FieldType& grid_field )
{
    using value_type = typename FieldType::value_type;
    value_type max;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::Max<value_type> reducer( max );
    Kokkos::parallel_reduce(
        "FieldVectorOps::normInf",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            grid_field.block(), grid_field.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { if ( fabs(data(i,j,k,0)) > result ) result = fabs(data(i,j,k,0)); },
        reducer );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &max, 1, MpiTraits<value_type>::type(),
                   MPI_MAX, grid_field.comm() );

    return max;
}

//---------------------------------------------------------------------------//
// Calculate the 1-norm of a vector. (Scalar fields only)
template<class FieldType>
typename FieldType::value_type norm1( const FieldType& grid_field )
{
    using value_type = typename FieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::parallel_reduce(
        "FieldVectorOps::norm1",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            grid_field.block(), grid_field.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += fabs(data(i,j,k,0)); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, grid_field.comm() );

    return sum;
}

//---------------------------------------------------------------------------//
// Calculate the 2-norm of a vector. (Scalar fields only)
template<class FieldType>
typename FieldType::value_type norm2( const FieldType& grid_field )
{
    using value_type = typename FieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::parallel_reduce(
        "FieldVectorOps::norm2",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            grid_field.block(), grid_field.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += data(i,j,k,0) * data(i,j,k,0); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, grid_field.comm() );

    return std::sqrt(sum);
}

//---------------------------------------------------------------------------//
// Calculate the dot product of two vectors. (Scalar fields only)
template<class FieldType>
typename FieldType::value_type dot( const FieldType& vec_a,
                                        const FieldType& vec_b )
{
    using value_type = typename FieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    Kokkos::parallel_reduce(
        "FieldVectorOps::dot",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            vec_a.block(), vec_a.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += data_a(i,j,k,0) * data_b(i,j,k,0); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, vec_a.comm() );

    return sum;
}

//---------------------------------------------------------------------------//
// Assign a value to a vector.
template<class FieldType>
void assign( const typename FieldType::value_type value,
             FieldType& grid_field )
{
    Kokkos::deep_copy( grid_field.data(), value );
}

//---------------------------------------------------------------------------//
// Scale a vector by a scalar.
template<class FieldType>
void scale( const typename FieldType::value_type alpha,
            FieldType& grid_field )
{
    auto data = grid_field.data();
    int e0 = data.extent(3);
    Kokkos::parallel_for(
        "FieldVectorOps::scale",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            grid_field.block(), grid_field.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data(i,j,k,n0) *= alpha;
        } );
}

//---------------------------------------------------------------------------//
// Vector update: C = alpha * A + beta * B
template<class FieldType>
void update( const typename FieldType::value_type alpha,
             const FieldType& vec_a,
             const typename FieldType::value_type beta,
             const FieldType& vec_b,
             FieldType& vec_c )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    int e0 = data_a.extent(3);
    Kokkos::parallel_for(
        "FieldVectorOps::update",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            vec_a.block(), vec_a.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data_c(i,j,k,n0) =
                    alpha * data_a(i,j,k,n0) + beta * data_b(i,j,k,n0);
        } );
}
//---------------------------------------------------------------------------//
// Vector update: D = alpha * A + beta * B + gamma * C
template<class FieldType>
void update( const typename FieldType::value_type alpha,
             const FieldType& vec_a,
             const typename FieldType::value_type beta,
             const FieldType& vec_b,
             const typename FieldType::value_type gamma,
             const FieldType& vec_c,
             FieldType& vec_d )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    auto data_d = vec_d.data();
    int e0 = data_a.extent(3);
    Kokkos::parallel_for(
        "FieldVectorOps::update",
        GridExecution::createLocalEntityPolicy<typename FieldType::execution_space>(
            vec_a.block(), vec_a.location() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data_d(i,j,k,n0) = alpha * data_a(i,j,k,n0) +
                                   beta * data_b(i,j,k,n0) +
                                   gamma * data_c(i,j,k,n0);
        } );
}

//---------------------------------------------------------------------------//

} // end namespace FieldVectorOps
} // end namespace Cajita

#endif // end CAJITA_GRIDFIELDVECTOROPS_HPP
