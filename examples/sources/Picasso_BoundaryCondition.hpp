#ifndef PICASSO_BOUNDARYCONDITION_HPP
#define PICASSO_BOUNDARYCONDITION_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

namespace Picasso
{

struct BoundaryCondition
{
    Kokkos::Array<long int, 6> bc_index_space;
    Kokkos::Array<bool, 6 > on_boundary;

    // Free slip boundary condition
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void apply( ViewType view, const int i, const int j,
                                       const int k ) const
    {
        // -x face
        if ( i == bc_index_space[0] && on_boundary[0] )
             view( i, j, k, 0 ) = 0.0;
        // +x face
        if ( i == bc_index_space[3] && on_boundary[3] )
             view( i, j, k, 0 ) = 0.0;
        // -y face
        if ( j == bc_index_space[1] && on_boundary[1] )
             view( i, j, k, 1 ) = 0.0;
        // +y face
        if ( j == bc_index_space[4] && on_boundary[4] )
             view( i, j, k, 1 ) = 0.0;
        // -z face
        if ( k == bc_index_space[2] && on_boundary[2] )
             view( i, j, k, 2 ) = 0.0;
        // +z face
        if ( k == bc_index_space[5] && on_boundary[5] )
             view( i, j, k, 2 ) = 0.0;
    }
};

struct Properties
{
    double gamma;
    double bulk_modulus;
    Kokkos::Array<double, 3> gravity;

    KOKKOS_INLINE_FUNCTION
    Properties( const double _gamma, const double _bulk_modulus,
                const Kokkos::Array<double, 3> _gravity )
        : gamma( _gamma )
        , bulk_modulus( _bulk_modulus )
        , gravity( _gravity )
    {
    }
};

} // end namespace Picasso

#endif // end PICASSO_BOUNDARYCONDITION_HPP
