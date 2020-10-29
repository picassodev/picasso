#ifndef PICASSO_POLYPIC_HPP
#define PICASSO_POLYPIC_HPP

#include <Cajita.hpp>

#include <Picasso_Types.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cmath>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Polynomial Particle-in-Cell
//---------------------------------------------------------------------------//
namespace PolyPIC
{
//---------------------------------------------------------------------------//
// Linear PolyPIC
// ---------------------------------------------------------------------------//
// Interpolate particle momentum to a collocated momentum grid. (First order
// splines). Requires SplineValue and SplineDistance when constructing the
// spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity c_p,
    const GridMomentum& mu_i,
    const double dt,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isNode<typename SplineDataType::entity_type>::value ||
      Cajita::isCell<typename SplineDataType::entity_type>::value) &&
     SplineDataType::order==1),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = mu_i.access();

    using value_type = typename GridMomentum::original_value_type;

    static_assert( 8 == ParticleVelocity::extent_0,
                   "PolyPIC with linear basis requires 8 velocity modes" );

    // Affine material motion operator.
    Mat3<value_type> am_p =
        { {1.0 + dt * c_p(1,0), dt * c_p(1,1), dt * c_p(1,2) },
          {dt * c_p(2,0), 1.0 + dt * c_p(2,1), dt * c_p(2,2) },
          {dt * c_p(3,0), dt * c_p(3,1), 1.0 + dt * c_p(3,2) } };

    // Invert the affine operator.
    auto am_inv_p = LinearAlgebra::inverse( am_p );

    // Project momentum.
    Vec3<value_type> distance;
    value_type dmom;
    value_type wm;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm = m_p * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Physical distance from particle to entity.
                distance(0) = sd.d[Dim::I][i];
                distance(1) = sd.d[Dim::J][j];
                distance(2) = sd.d[Dim::K][k];

                // Compute Lagrangian mapping to node.
                auto mapping = am_inv_p * distance;

                // Contribute momentum.
                for ( int d = 0; d < 3; ++d )
                {
                    // Sum velocity modes.
                    dmom = c_p(0,d) +
                           c_p(1,d) * mapping(0) +
                           c_p(2,d) * mapping(1) +
                           c_p(3,d) * mapping(2) +
                           c_p(4,d) * mapping(0) * mapping(1) +
                           c_p(5,d) * mapping(0) * mapping(2) +
                           c_p(6,d) * mapping(1) * mapping(2) +
                           c_p(7,d) * mapping(0) * mapping(1) * mapping(2);

                    // Contribute to momentum.
                    momentum_access( sd.s[Dim::I][i],
                                     sd.s[Dim::J][j],
                                     sd.s[Dim::K][k],
                                     d ) += wm * dmom;
                }
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a staggered momentum grid. (First order
// splines). Requires SplineValue and SplineDistance when constructing the
// spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class SplineDataType,
         class GridMomentum>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass m_p,
    const ParticleVelocity c_p,
    const GridMomentum& mu_i,
    const double dt,
    const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isFace<typename SplineDataType::entity_type>::value ||
      Cajita::isEdge<typename SplineDataType::entity_type>::value) &&
     SplineDataType::order==1),void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridMomentum>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = mu_i.access();

    using value_type = typename GridMomentum::original_value_type;

    static_assert( 8 == ParticleVelocity::extent_0,
                   "PolyPIC with linear basis requires 8 velocity modes" );

    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Affine material motion operator.
    Mat3<value_type> am_p =
        { {1.0 + dt * c_p(1,0), dt * c_p(1,1), dt * c_p(1,2) },
          {dt * c_p(2,0), 1.0 + dt * c_p(2,1), dt * c_p(2,2) },
          {dt * c_p(3,0), dt * c_p(3,1), 1.0 + dt * c_p(3,2) } };

    // Invert the affine operator.
    auto am_inv_p = LinearAlgebra::inverse( am_p );

    // Project momentum.
    Vec3<value_type> distance;
    value_type dmom;
    value_type wm;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm = m_p * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Physical distance from particle to entity.
                distance(0) = sd.d[Dim::I][i];
                distance(1) = sd.d[Dim::J][j];
                distance(2) = sd.d[Dim::K][k];

                // Compute Lagrangian mapping to node.
                auto mapping = am_inv_p * distance;

                // Sum velocity modes.
                dmom = c_p(0,dim) +
                       c_p(1,dim) * mapping(0) +
                       c_p(2,dim) * mapping(1) +
                       c_p(3,dim) * mapping(2) +
                       c_p(4,dim) * mapping(0) * mapping(1) +
                       c_p(5,dim) * mapping(0) * mapping(2) +
                       c_p(6,dim) * mapping(1) * mapping(2) +
                       c_p(7,dim) * mapping(0) * mapping(1) * mapping(2);

                // Contribute to momentum.
                momentum_access( sd.s[Dim::I][i],
                                 sd.s[Dim::J][j],
                                 sd.s[Dim::K][k],
                                 0 ) += wm * dmom;
            }
}

//---------------------------------------------------------------------------//
// Interpolate colocated grid velocity to particle. (First order
// splines). Requires SplineValue and SplineGradient when constructing the
// spline data.
template<class GridVelocity,
         class ParticleVelocity,
         class SplineDataType>
KOKKOS_INLINE_FUNCTION
void g2p( const GridVelocity& u_i,
          ParticleVelocity& c_p,
          const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isNode<typename SplineDataType::entity_type>::value ||
      Cajita::isCell<typename SplineDataType::entity_type>::value) &&
     SplineDataType::order==1),void*>::type = 0 )
{
    using value_type = typename GridVelocity::value_type;

    static_assert( 8 == ParticleVelocity::extent_0,
                   "PolyPIC with linear basis requires 8 velocity modes" );

    // Reset particle velocity.
    c_p = 0.0;

    // Update particle.
    LinearAlgebra::Vector<value_type,8> basis;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute basis values.
                basis(0) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                basis(1) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                basis(2) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                basis(3) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                basis(4) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                basis(5) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                basis(6) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];
                basis(7) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];

                // Compute particle velocity.
                for ( int r = 0; r < 8; ++r )
                    for ( int d = 0; d < 3; ++d )
                        c_p(r,d) +=
                            basis(r) *
                            u_i(sd.s[Dim::I][i],sd.s[Dim::J][j],sd.s[Dim::K][k],d);
            }
}

//---------------------------------------------------------------------------//
// Interpolate staggered grid velocity to particle. (First order
// splines). Requires SplineValue and SplineGradient when constructing the
// spline data.
template<class GridVelocity,
         class ParticleVelocity,
         class SplineDataType>
KOKKOS_INLINE_FUNCTION
void g2p( const GridVelocity& u_i,
          ParticleVelocity& c_p,
          const SplineDataType& sd,
    typename std::enable_if<
    ((Cajita::isFace<typename SplineDataType::entity_type>::value ||
      Cajita::isEdge<typename SplineDataType::entity_type>::value) &&
     SplineDataType::order==1),void*>::type = 0 )
{
    using value_type = typename GridVelocity::value_type;

    static_assert( 8 == ParticleVelocity::extent_0,
                   "PolyPIC with linear basis requires 8 velocity modes" );

    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Reset particle velocity.
    c_p.column(dim) = 0.0;

    // Update particle.
    LinearAlgebra::Vector<value_type,8> basis;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute basis values.
                basis(0) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                basis(1) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                basis(2) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                basis(3) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                basis(4) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                basis(5) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                basis(6) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];
                basis(7) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];

                // Compute particle velocity.
                for ( int r = 0; r < 8; ++r )
                    c_p(r,dim) +=
                        basis(r) *
                        u_i(sd.s[Dim::I][i],sd.s[Dim::J][j],sd.s[Dim::K][k],0);

            }
}

//---------------------------------------------------------------------------//

} // end namespace PolyPIC
} // end namespace Picasso

#endif // end PICASSO_POLYPIC_HPP
