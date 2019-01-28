#ifndef HARLOW_GRIDFIELDVECTOROPS_HPP
#define HARLOW_GRIDFIELDVECTOROPS_HPP

#include <Harlow_GridField.hpp>
#include <Harlow_GridExecPolicy.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_MpiTraits.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <type_traits>
#include <cmath>

namespace Harlow
{
namespace GridVectorOp
{
//---------------------------------------------------------------------------//
// Given a grid field get a local execution policy.
template<class GridFieldType>
Kokkos::MDRangePolicy<typename GridFieldType::execution_space,
                      Kokkos::Rank<3> >
createVectorOpExecPolicy( const GridFieldType& grid_field )
{
    using execution_space = typename GridFieldType::execution_space;

    if ( FieldLocation::Node == grid_field.location() )
        return createLocalNodeExecPolicy<execution_space>(
            grid_field.block() );
    else if ( FieldLocation::Cell == grid_field.location() )
        return createLocalCellExecPolicy<execution_space>(
            grid_field.block() );
    else if ( FieldLocation::Face == grid_field.location() )
        return createLocalFaceExecPolicy<execution_space>(
            grid_field.block() );
    else
        throw std::invalid_argument("Undefined field location");
}

//---------------------------------------------------------------------------//
// Calculate the infinity-norm of a vector. (Scalar fields only)
template<class GridFieldType>
typename GridFieldType::value_type normInf( const GridFieldType& grid_field )
{
    using value_type = typename GridFieldType::value_type;
    value_type max;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::Max<value_type> reducer( max );
    Kokkos::parallel_reduce(
        "GridVectorOp::normInf",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { if ( fabs(data(i,j,k)) > result ) result = fabs(data(i,j,k)); },
        reducer );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &max, 1, MpiTraits<value_type>::type(),
                   MPI_MAX, grid_field.comm() );

    return max;
}

//---------------------------------------------------------------------------//
// Calculate the 1-norm of a vector. (Scalar fields only)
template<class GridFieldType>
typename GridFieldType::value_type norm1( const GridFieldType& grid_field )
{
    using value_type = typename GridFieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::parallel_reduce(
        "GridVectorOp::norm1",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += fabs(data(i,j,k)); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, grid_field.comm() );

    return sum;
}

//---------------------------------------------------------------------------//
// Calculate the 2-norm of a vector. (Scalar fields only)
template<class GridFieldType>
typename GridFieldType::value_type norm2( const GridFieldType& grid_field )
{
    using value_type = typename GridFieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data = grid_field.data();
    Kokkos::parallel_reduce(
        "GridVectorOp::norm2",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += data(i,j,k) * data(i,j,k); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, grid_field.comm() );

    return std::sqrt(sum);
}

//---------------------------------------------------------------------------//
// Calculate the dot product of two vectors. (Scalar fields only)
template<class GridFieldType>
typename GridFieldType::value_type dot( const GridFieldType& vec_a,
                                        const GridFieldType& vec_b )
{
    using value_type = typename GridFieldType::value_type;
    value_type sum;

    // Local reduction.
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    Kokkos::parallel_reduce(
        "GridVectorOp::dot",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, value_type& result )
        { result += data_a(i,j,k) * data_b(i,j,k); },
        sum );

    // Global reduction.
    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, MpiTraits<value_type>::type(),
                   MPI_SUM, vec_a.comm() );

    return sum;
}

//---------------------------------------------------------------------------//
// Assign a value to a vector.
template<class GridFieldType>
void assign( const typename GridFieldType::value_type value,
             GridFieldType& grid_field )
{
    Kokkos::deep_copy( grid_field.data(), value );
}

//---------------------------------------------------------------------------//
// Scale a vector by a scalar.

// Rank - 0
template<class GridFieldType>
void scale(
    const typename GridFieldType::value_type alpha,
    GridFieldType& grid_field,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==0),int*>::type = 0 )
{
    auto data = grid_field.data();
    Kokkos::parallel_for(
        "GridVectorOp::scale",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        { data(i,j,k) *= alpha; } );
}

// Rank - 1
template<class GridFieldType>
void scale(
    const typename GridFieldType::value_type alpha,
    GridFieldType& grid_field,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==1),int*>::type = 0 )
{
    auto data = grid_field.data();
    auto e0 = data.extent(3);
    Kokkos::parallel_for(
        "GridVectorOp::scale",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data(i,j,k,n0) *= alpha;
        } );
}

// Rank - 2
template<class GridFieldType>
void scale(
    const typename GridFieldType::value_type alpha,
    GridFieldType& grid_field,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==2),int*>::type = 0 )
{
    auto data = grid_field.data();
    int e0 = data.extent(3);
    int e1 = data.extent(4);
    Kokkos::parallel_for(
        "GridVectorOp::scale",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    data(i,j,k,n0,n1) *= alpha;
        } );
}

// Rank - 3
template<class GridFieldType>
void scale(
    const typename GridFieldType::value_type alpha,
    GridFieldType& grid_field,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==3),int*>::type = 0 )
{
    auto data = grid_field.data();
    int e0 = data.extent(3);
    int e1 = data.extent(4);
    int e2 = data.extent(4);
    Kokkos::parallel_for(
        "GridVectorOp::scale",
        createVectorOpExecPolicy( grid_field ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    for ( int n2 = 0; n2 < e2; ++n2 )
                        data(i,j,k,n0,n1,n2) *= alpha;
        } );
}

//---------------------------------------------------------------------------//
// Vector update: C = alpha * A + beta * B

// Rank - 0
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    GridFieldType& vec_c,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==0),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        { data_c(i,j,k) = alpha * data_a(i,j,k) + beta * data_b(i,j,k); } );
}

// Rank - 1
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    GridFieldType& vec_c,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==1),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    int e0 = data_a.extent(3);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data_c(i,j,k,n0) =
                    alpha * data_a(i,j,k,n0) + beta * data_b(i,j,k,n0);
        } );
}

// Rank - 2
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    GridFieldType& vec_c,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==2),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    int e0 = data_a.extent(3);
    int e1 = data_a.extent(4);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    data_c(i,j,k,n0,n1) =
                        alpha * data_a(i,j,k,n0,n1) + beta * data_b(i,j,k,n0,n1);
        } );
}

// Rank - 3
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    GridFieldType& vec_c,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==3),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    int e0 = data_a.extent(3);
    int e1 = data_a.extent(4);
    int e2 = data_a.extent(5);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    for ( int n2 = 0; n2 < e2; ++n2 )
                        data_c(i,j,k,n0,n1,n2) =
                            alpha * data_a(i,j,k,n0,n1,n2) +
                            beta * data_b(i,j,k,n0,n1,n2);
        } );
}

//---------------------------------------------------------------------------//
// Vector update: D = alpha * A + beta * B + gamma * C

// Rank - 0
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    const typename GridFieldType::value_type gamma,
    const GridFieldType& vec_c,
    GridFieldType& vec_d,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==0),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    auto data_d = vec_d.data();
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            data_d(i,j,k) = alpha * data_a(i,j,k) +
                            beta * data_b(i,j,k) +
                            gamma * data_c(i,j,k);
        } );
}

// Rank - 1
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    const typename GridFieldType::value_type gamma,
    const GridFieldType& vec_c,
    GridFieldType& vec_d,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==1),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    auto data_d = vec_d.data();
    int e0 = data_a.extent(3);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                data_d(i,j,k,n0) = alpha * data_a(i,j,k,n0) +
                                   beta * data_b(i,j,k,n0) +
                                   gamma * data_c(i,j,k,n0);
        } );
}

// Rank - 2
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    const typename GridFieldType::value_type gamma,
    const GridFieldType& vec_c,
    GridFieldType& vec_d,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==2),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    auto data_d = vec_d.data();
    int e0 = data_a.extent(3);
    int e1 = data_a.extent(4);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    data_d(i,j,k,n0,n1) = alpha * data_a(i,j,k,n0,n1) +
                                          beta * data_b(i,j,k,n0,n1) +
                                          gamma * data_c(i,j,k,n0,n1);
        } );
}

// Rank - 3
template<class GridFieldType>
void update(
    const typename GridFieldType::value_type alpha,
    const GridFieldType& vec_a,
    const typename GridFieldType::value_type beta,
    const GridFieldType& vec_b,
    const typename GridFieldType::value_type gamma,
    const GridFieldType& vec_c,
    GridFieldType& vec_d,
    typename std::enable_if<
    (std::rank<typename GridFieldType::data_type>::value==3),int*>::type = 0 )
{
    auto data_a = vec_a.data();
    auto data_b = vec_b.data();
    auto data_c = vec_c.data();
    auto data_d = vec_d.data();
    int e0 = data_a.extent(3);
    int e1 = data_a.extent(4);
    int e2 = data_a.extent(5);
    Kokkos::parallel_for(
        "GridVectorOp::update",
        createVectorOpExecPolicy( vec_a ),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {
            for ( int n0 = 0; n0 < e0; ++n0 )
                for ( int n1 = 0; n1 < e1; ++n1 )
                    for ( int n2 = 0; n2 < e2; ++n2 )
                        data_d(i,j,k,n0,n1,n2) = alpha * data_a(i,j,k,n0,n1,n2) +
                                                 beta * data_b(i,j,k,n0,n1,n2) +
                                                 gamma * data_c(i,j,k,n0,n1,n2);
        } );
}

//---------------------------------------------------------------------------//

} // end namespace GridVectorOp
} // end namespace Harlow

#endif // end HARLOW_GRIDFIELDVECTOROPS_HPP
