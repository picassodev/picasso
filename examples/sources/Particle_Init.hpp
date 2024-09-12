#include <Picasso.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Picasso
{

template <typename ParticleType, typename VelocityType>
struct ParticleInitFunc
{
    Kokkos::Array<double, 6> block;
    double density;

    ParticleInitFunc( const Kokkos::Array<double, 6>& _block,
                      const double _density )
        : block( _block )
        , density( _density )
    {
    }

    KOKKOS_INLINE_FUNCTION bool
    operator()( const int, const double x[3], const double pv,
                typename ParticleType::particle_type& p ) const
    {
        if ( block[0] <= x[0] && x[0] <= block[3] && block[1] <= x[1] &&
             x[1] <= block[4] && block[2] <= x[2] && x[2] <= block[5] )
        {

            Picasso::get( p, Picasso::Stress() ) = 0.0;
            Picasso::get( p, VelocityType() ) = 0.0;
            Picasso::get( p, Picasso::DetDefGrad() ) = 1.0;
            Picasso::get( p, Picasso::Mass() ) = pv * density;
            Picasso::get( p, Picasso::Volume() ) = pv;
            Picasso::get( p, Picasso::Pressure() ) = 0.0;

            for ( int d = 0; d < 3; ++d )
                Picasso::get( p, Picasso::Position(), d ) = x[d];
            return true;
        }

        return false;
    }
};

template <typename ParticleType, typename VelocityType>
auto createParticleInitFunc( const ParticleType&, const VelocityType&,
                             const Kokkos::Array<double, 6>& block,
                             const double density )
{
    return ParticleInitFunc<ParticleType, VelocityType>{ block, density };
}

} // namespace Picasso
