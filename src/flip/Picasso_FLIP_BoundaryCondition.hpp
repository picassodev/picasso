#ifndef PICASSO_FLIP_BOUNDARYCONDITION_HPP
#define PICASSO_FLIP_BOUNDARYCONDITION_HPP

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

namespace Picasso
{
namespace FLIP
{
//---------------------------------------------------------------------------//
// No-slip on domain box boundaries.
template<class Mesh>
struct DomainNoSlipBoundary
{
    Kokkos::Array<long,3> _global_min;
    Kokkos::Array<long,3> _global_max;
    Cajita::IndexConversion::L2G<typename Mesh::cajita_mesh,Cajita::Node> _l2g;

    DomainNoSlipBoundary( const Mesh& mesh )
        : _l2g( *(mesh.localGrid()) )
    {
        const auto& global_grid = mesh.localGrid()->globalGrid();
        auto halo_min = mesh.minimumHaloWidth();
        for ( int d = 0; d < 3; ++d )
        {
            if ( global_grid.isPeriodic(d) )
            {
                _global_min[d] = -1;
                _global_max[d] = global_grid.globalNumEntity( Cajita::Node(), d );
            }
            else
            {
                _global_min[d] = halo_min + 1;
                _global_max[d] = global_grid.globalNumEntity( Cajita::Node(), d ) -
                                 halo_min - 1;
            }
        }
    }

    template<class View>
    KOKKOS_INLINE_FUNCTION
    void operator()( const View& v, const int i, const int j, const int k ) const
    {
        // Compute global index.
        int gi, gj, gk;
        _l2g( i, j, k, gi, gj, gk );

        // If outside the domain and not periodic apply no-slip.
        if ( _global_min[0] > gi ||
             _global_max[0] <= gi ||
             _global_min[1] > gj ||
             _global_max[1] <= gj ||
             _global_min[2] > gk ||
             _global_max[2] <= gk )
        {
            for ( int d = 0; d < 3; ++d )
                v(i,j,k,d) = 0.0;
        }
    }
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_BOUNDARYCONDITION_HPP
