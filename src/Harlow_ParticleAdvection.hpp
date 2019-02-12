#ifndef HARLOW_PARTICLEADVECTION_HPP
#define HARLOW_PARTICLEADVECTION_HPP

#include <Kokkos_Core.hpp>

namespace Harlow
{
namespace ParticleAdvection
{
//---------------------------------------------------------------------------//
/*!
  \brief Update particle positions using the Lagrangian grid velocity (the
  first velocity mode).

  \param dt The time step size

  \param velocity View of particle velocities

  \param position View of particle positions
*/
template<class ParticleVelocityView, class ParticlePositionView>
void move( const double dt,
           const ParticleVelocityView& velocity,
           ParticlePositionView& position )
{
    Kokkos::parallel_for(
        "particle_advection",
        Kokkos::RangePolicy<typename ParticleVelocityView::execution_space>(
            0, velocity.extent(0) ),
        KOKKOS_LAMBDA( const int p ){
            for ( int d = 0; d < 3; ++d )
                position(p,d) += velocity(p,0,d) * dt;
        } );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleAdvection
} // end namespace Harlow

#endif // end HARLOW_PARTICLEADVECTION_HPP
