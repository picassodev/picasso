#ifndef PICASSO_APIC_HPP
#define PICASSO_APIC_HPP

#include <Cajita.hpp>

#include <Picasso_Types.hpp>
#include <Picasso_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cmath>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Affine Particle-in-Cell (APIC)
//---------------------------------------------------------------------------//
namespace APIC
{
//---------------------------------------------------------------------------//
// Inertial tensor scale factor.
template<class SplineDataType>
typename SplineDataType::scalar_type
inertialScaling(
    const SplineDataType& sd,
    typename std::enable_if<(2==SplineDataType::order),void*>::type = 0)
{
    return 4.0 / ( sd.dx * sd.dx );
}

template<class SplineDataType>
typename SplineDataType::scalar_type
inertialScaling(
    const SplineDataType& sd,
    typename std::enable_if<(3==SplineDataType::order),void*>::type = 0)
{
    return 3.0 / ( sd.dx * sd.dx );
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a collocated momentum grid. (Second and
// Third order splines). Requires SplineValue, SplineGradient, and
// SplineDistance when constructing the spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class ParticleAffineMatrix,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity u_p,
    const ParticleAffineMatrix B_p,
    const GridMomentum& grid_momentum,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isNode<typename SplineDataType::entity_type>::value ||
      Cajita::isCell<typename SplineDataType::entity_type>::value) &&
     (SplineDataType::order==2 || SplineDataType::order==3)),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = grid_momentum.access();

    using value_type = typename GridMomentum::original_value_type;

    // Scaling factor from inertial tensor with quadratic shape
    // functions.
    value_type D_p_inv = inertialScaling( sd );

    // Project momentum.
    Vec3<value_type> distance;
    value_type wm_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Physical distance to entity.
                distance(Dim::I) = sd.d[Dim::I][i];
                distance(Dim::J) = sd.d[Dim::J][j];
                distance(Dim::K) = sd.d[Dim::K][k];

                // Compute the action of B_p on the distance scaled by the
                // intertial tensor scaling factor.
                auto D_p_inv_B_p_d = D_p_inv * B_p * distance;

                // Weight times mass.
                wm_ip = sd.w[Dim::I][i] *
                        sd.w[Dim::J][j] *
                        sd.w[Dim::K][k] *
                        m_p;

                // Interpolate particle momentum to the entity.
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
                for ( int d = 0; d < 3; ++d )
                    momentum_access( sd.s[Dim::I][i],
                                     sd.s[Dim::J][j],
                                     sd.s[Dim::K][k],
                                     d ) +=
                        wm_ip * ( u_p(d) + D_p_inv_B_p_d(d) );
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a staggered momentum grid. (Second and
// Third order splines). Requires SplineValue, SplineGradient, and
// SplineDistance when constructing the spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class ParticleAffineMatrix,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity u_p,
    const ParticleAffineMatrix B_p,
    const GridMomentum& grid_momentum,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isEdge<typename SplineDataType::entity_type>::value ||
      Cajita::isFace<typename SplineDataType::entity_type>::value) &&
     (SplineDataType::order==2 || SplineDataType::order==3)),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = grid_momentum.access();

    using value_type = typename GridMomentum::original_value_type;

    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Scaling factor from inertial tensor with quadratic shape
    // functions.
    value_type D_p_inv = inertialScaling( sd );

    // Project momentum.
    Vec3<value_type> distance;
    value_type wm_ip;
    value_type B_p_d;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Physical distance to entity.
                distance(Dim::I) = sd.d[Dim::I][i];
                distance(Dim::J) = sd.d[Dim::J][j];
                distance(Dim::K) = sd.d[Dim::K][k];

                // Compute the action of B_p on the distance scaled by the
                // intertial tensor scaling factor.
                B_p_d = ~B_p.row(dim) * distance;

                // Weight times mass.
                wm_ip = sd.w[Dim::I][i] *
                        sd.w[Dim::J][j] *
                        sd.w[Dim::K][k] *
                        m_p;

                // Interpolate particle momentum to the entity.
                momentum_access( sd.s[Dim::I][i],
                                 sd.s[Dim::J][j],
                                 sd.s[Dim::K][k],
                                 0 ) +=
                    wm_ip * ( u_p(dim) + D_p_inv * B_p_d );
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a collocated momentum grid. (First order
// splines)
template<class ParticleMass,
         class ParticleVelocity,
         class ParticleAffineMatrix,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity u_p,
    const ParticleAffineMatrix B_p,
    const GridMomentum& entity_momentum,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isNode<typename SplineDataType::entity_type>::value ||
      Cajita::isCell<typename SplineDataType::entity_type>::value) &&
     (SplineDataType::order==1)),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = entity_momentum.access();

    using value_type = typename GridMomentum::original_value_type;

    // Project momentum.
    Vec3<value_type> gm_ip;
    value_type wm_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm_ip = sd.w[Dim::I][i] *
                        sd.w[Dim::J][j] *
                        sd.w[Dim::K][k] *
                        m_p;

                // Weight gradient times mass.
                gm_ip(0) = sd.g[Dim::I][i] *
                           sd.w[Dim::J][j] *
                           sd.w[Dim::K][k] *
                           m_p;
                gm_ip(1) = sd.w[Dim::I][i] *
                           sd.g[Dim::J][j] *
                           sd.w[Dim::K][k] *
                           m_p;
                gm_ip(2) = sd.w[Dim::I][i] *
                           sd.w[Dim::J][j] *
                           sd.g[Dim::K][k] *
                           m_p;

                // Compute the action of B_p on the gradient.
                auto B_g_d = B_p * gm_ip;

                // Interpolate particle momentum to the entity.
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
                for ( int d = 0; d < 3; ++d )
                    momentum_access( sd.s[Dim::I][i],
                                     sd.s[Dim::J][j],
                                     sd.s[Dim::K][k],
                                     d ) += wm_ip * u_p(d) + B_g_d(d);
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a staggered momentum grid. (First order
// splines). Requires SplineValue and SplineGradient when constructing the
// spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class ParticleAffineMatrix,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity u_p,
    const ParticleAffineMatrix B_p,
    const GridMomentum& entity_momentum,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isEdge<typename SplineDataType::entity_type>::value ||
      Cajita::isFace<typename SplineDataType::entity_type>::value) &&
     (SplineDataType::order==1)),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = entity_momentum.access();

    using value_type = typename GridMomentum::original_value_type;

    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Project momentum.
    Vec3<value_type> gm_ip;
    value_type wm_ip;
    value_type B_g_d;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm_ip = sd.w[Dim::I][i] *
                        sd.w[Dim::J][j] *
                        sd.w[Dim::K][k] *
                        m_p;

                // Weight gradient times mass.
                gm_ip(0) = sd.g[Dim::I][i] *
                           sd.w[Dim::J][j] *
                           sd.w[Dim::K][k] *
                           m_p;
                gm_ip(1) = sd.w[Dim::I][i] *
                           sd.g[Dim::J][j] *
                           sd.w[Dim::K][k] *
                           m_p;
                gm_ip(2) = sd.w[Dim::I][i] *
                           sd.w[Dim::J][j] *
                           sd.g[Dim::K][k] *
                           m_p;

                // Compute the action of B_p on the gradient.
                B_g_d = ~B_p.row(dim) * gm_ip(d);

                // Interpolate particle momentum to the entity.
                momentum_access( sd.s[Dim::I][i],
                                 sd.s[Dim::J][j],
                                 sd.s[Dim::K][k],
                                 0 ) += wm_ip * u_p(d) + B_g_d(dim);
            }
}

//---------------------------------------------------------------------------//
// Interpolate collocated grid velocity to the particle. Requires SplineValue
// and SplineGradient when constructing the spline data.
template<class GridVelocity,
         class SplineDataType,
         class ParticleVelocity,
         class ParticleAffineMatrix>
KOKKOS_INLINE_FUNCTION
void g2p(
    const GridVelocity& entity_velocity,
    ParticleVelocity u_p,
    ParticleAffineMatrix B_p,
    const SplineDataType& sd,
    typename std::enable_if<
    (Cajita::isNode<typename SplineDataType::entity_type>::value ||
     Cajita::isCell<typename SplineDataType::entity_type>::value),
    void*>::type = 0 )
{
    using value_type = typename GridVelocity::value_type;

    // Reset the particle values.
    u_p = 0.0;
    B_p = 0.0;

    // Update particle.
    Vec3<value_type> distance;
    value_type w_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Projection weight.
                w_ip = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Entity velocity
                auto u_i = entity_velocity( sd.s[Dim::I][i],
                                            sd.s[Dim::J][j],
                                            sd.s[Dim::K][k] );

                // Update velocity.
                u_p = u_p + w_ip * u_i;

                // Physical distance to entity.
                distance(Dim::I) = sd.d[Dim::I][i];
                distance(Dim::J) = sd.d[Dim::J][j];
                distance(Dim::K) = sd.d[Dim::K][k];

                // Update affine matrix.
                B_p = B_p + w_ip * u_i * ~distance;
            }
}

//---------------------------------------------------------------------------//
// Interpolate staggered grid velocity to the particle.
template<class GridVelocity,
         class SplineDataType,
         class ParticleVelocity,
         class ParticleAffineMatrix>
KOKKOS_INLINE_FUNCTION
void g2p(
    const GridVelocity& entity_velocity,
    ParticleVelocity u_p,
    ParticleAffineMatrix B_p,
    const SplineDataType& sd,
    typename std::enable_if<
    (Cajita::isEdge<typename SplineDataType::entity_type>::value ||
     Cajita::isFace<typename SplineDataType::entity_type>::value),
    void*>::type = 0 )
{
    using value_type = typename GridVelocity::value_type;

    // Get the velocity dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Reset the particle values.
    u_p(dim) = 0.0;
    B_p.row(dim) = 0.0;

    // Update particle.
    Vec3<value_type> distance;
    value_type w_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Projection weight.
                w_ip = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Entity velocity
                auto u_i = entity_velocity( sd.s[Dim::I][i],
                                            sd.s[Dim::J][j],
                                            sd.s[Dim::K][k] );

                // Update velocity.
                u_p(dim) += w_ip * u_i(dim);

                // Physical distance to entity.
                distance(Dim::I) = sd.d[Dim::I][i];
                distance(Dim::J) = sd.d[Dim::J][j];
                distance(Dim::K) = sd.d[Dim::K][k];

                // Update affine matrix.
                B_p.row(dim) = B_p.row(dim) + w_ip * u_i * distance;
            }
}

//---------------------------------------------------------------------------//

} // end namespace APIC

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_APIC_HPP
