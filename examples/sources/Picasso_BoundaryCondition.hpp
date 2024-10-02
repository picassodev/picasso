#ifndef PICASSO_BOUNDARYCONDITION_HPP
#define PICASSO_BOUNDARYCONDITION_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

namespace Picasso
{

template <class LocalGridType>
struct BoundaryCondition
{
    LocalGridType local_grid;

    // Free slip boundary condition
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void apply( ViewType view, const int i, const int j,
                                       const int k ) const
    {

        auto index_space = local_grid.indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );

        auto global_grid = local_grid.globalGrid();

        // -x face
        if ( i == index_space.min( Cabana::Grid::Dim::I ) &&
             global_grid.onLowBoundary( Cabana::Grid::Dim::I ) )
             view( i, j, k, 0 ) = 0.0;
        // +x face
        if ( i == index_space.min( Cabana::Grid::Dim::I ) &&
             global_grid.onHighBoundary( Cabana::Grid::Dim::I ) )
             view( i, j, k, 0 ) = 0.0;
        // -y face
        if ( j == index_space.min( Cabana::Grid::Dim::J ) &&
             global_grid.onLowBoundary( Cabana::Grid::Dim::J ) )
             view( i, j, k, 1 ) = 0.0;
        // +y face
        if ( j == index_space.min( Cabana::Grid::Dim::J ) &&
             global_grid.onHighBoundary( Cabana::Grid::Dim::J ) )
             view( i, j, k, 1 ) = 0.0;
        // -z face
        if ( k == index_space.min( Cabana::Grid::Dim::K ) &&
             global_grid.onLowBoundary( Cabana::Grid::Dim::K ) )
             view( i, j, k, 2 ) = 0.0;
        // +z face
        if ( k == index_space.min( Cabana::Grid::Dim::K ) &&
             global_grid.onHighBoundary( Cabana::Grid::Dim::K ) )
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
