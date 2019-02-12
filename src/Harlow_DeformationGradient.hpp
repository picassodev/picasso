#ifndef HARLOW_DEFORMATIONGRADIENT_HPP
#define HARLOW_DEFORMATIONGRADIENT_HPP

#include <Harlow_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Harlow
{
namespace DeformationGradient
{
//---------------------------------------------------------------------------//
// Update the particle deformation gradient using the velocity gradient
// encoded in the particle modes.
//---------------------------------------------------------------------------//
// Compute a truncated matrix exponential to time step the deformation
// gradient.
template<class Real>
KOKKOS_INLINE_FUNCTION
void computeUpdateMatrix( const Real c[3][3], Real m[3][3] )
{
    // Compute M = I + C
    m[0][0] = c[0][0] + 1.0;
    m[0][1] = c[0][1];
    m[0][2] = c[0][2];
    m[1][0] = c[1][0];
    m[1][1] = c[1][1] + 1.0;
    m[1][2] = c[1][2];
    m[2][0] = c[2][0];
    m[2][1] = c[2][1];
    m[2][2] = c[2][2] + 1.0;

    // if det(M) > 0 return
    if ( DenseLinearAlgebra::determinant(m) > 0.0 )
        return;

    // Otherwise compute the update matrix of C/2
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            m[i][j] = c[i][j] / 2.0;
    Real n[3][3];
    computeUpdateMatrix( m, n );

    // Square the result.
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            for ( int k = 0; k < 3; ++k )
                m[i][j] = n[i][k] * n[k][j];
}

//---------------------------------------------------------------------------//
/*
  \brief Update the particle deformation gradient.

  \param dt The time step size

  \param velocity View of particle velocities

  \param deformation_gradient View of particle deformation gradients
*/
template<class ParticleVelocityView,
         class ParticleDefGradView>
void update( const double dt,
             const ParticleVelocityView& velocity,
             ParticleDefGradView& deformation_gradient )
{
    Kokkos::parallel_for(
        "deformation_gradient_update",
        Kokkos::RangePolicy<typename ParticleDefGradView::execution_space>(
            0,velocity.extent(0) ),
        KOKKOS_LAMBDA( const int p ){

            // Extract the velocity gradient and scale it by the timestep
            // size.
            double c[3][3] = { {dt * velocity(p,1,Dim::I),
                                dt * velocity(p,1,Dim::J),
                                dt * velocity(p,1,Dim::K)},
                               {dt * velocity(p,2,Dim::I),
                                dt * velocity(p,2,Dim::J),
                                dt * velocity(p,2,Dim::K)},
                               {dt * velocity(p,3,Dim::I),
                                dt * velocity(p,3,Dim::J),
                                dt * velocity(p,3,Dim::K)} };

            // Compute the update matrix.
            double m[3][3];
            computeUpdateMatrix( c, m );

            // Move the deformation gradient forward in time.
            double f[3][3] = {{0.0,0.0,0.0},
                              {0.0,0.0,0.0},
                              {0.0,0.0,0.0}};
            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    for ( int k = 0; k < 3; ++k )
                        f[i][j] += m[i][k] * deformation_gradient(p,k,j);

            // Assign the results.
            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    deformation_gradient(p,i,j) = f[i][j];
        } );
}

//---------------------------------------------------------------------------//

} // end namespace DeformationGradient
} // end namespace Harlow

#endif // end HARLOW_DEFORMATIONGRADIENT_HPP
