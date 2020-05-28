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
struct DomainNoSlipBoundary
{
    Kokkos::Array<long,3> _global_offset;
    Kokkos::Array<long,3> _global_min;
    Kokkos::Array<long,3> _global_max;

    DomainNoSlipBoundary() {};

    template<class Mesh>
    DomainNoSlipBoundary( const Mesh& mesh )
    {
        auto ghost_space = mesh.localGrid()->indexSpace(
            Cajita::Ghost(),Cajita::Node(),Cajita::Global());
        for ( int d = 0; d < 3; ++d )
            _global_offset[d] = ghost_space.min(d);

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
        long i_global = _global_offset[0] + i;
        long j_global = _global_offset[1] + j;
        long k_global = _global_offset[2] + k;

        if ( _global_min[0] > i_global ||
             _global_max[0] <= i_global ||
             _global_min[1] > j_global ||
             _global_max[1] <= j_global ||
             _global_min[2] > k_global ||
             _global_max[2] <= k_global )
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
