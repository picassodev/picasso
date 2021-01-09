/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

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
//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a collocated momentum
// grid. (First order splines). Requires SplineValue, SplineDistance when
// constructing the spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class SplineDataType,
         class GridMomentum,
         class GridMass>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass& m_p,
    const ParticleVelocity& c_p,
    const GridMomentum& mu_i,
    const GridMass& m_i,
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

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "PolyPIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "PolyPIC::p2g requires spline distance" );

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
    LinearAlgebra::Vector<value_type,8> basis;
    Vec3<value_type> distance;
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

                // Compute polynomial basis.
                basis(0) = 1.0;
                basis(1) = mapping(0);
                basis(2) = mapping(1);
                basis(3) = mapping(2);
                basis(4) = mapping(0) * mapping(1);
                basis(5) = mapping(0) * mapping(2);
                basis(6) = mapping(1) * mapping(2);
                basis(7) = mapping(0) * mapping(1) * mapping(2);

                // Contribute momentum.
                for ( int d = 0; d < 3; ++d )
                {
                    momentum_access( sd.s[Dim::I][i],
                                     sd.s[Dim::J][j],
                                     sd.s[Dim::K][k],
                                     d ) += wm * ~c_p.column(d) * basis;
                }

                // Contribute to mass.
                mass_access( sd.s[Dim::I][i],
                             sd.s[Dim::J][j],
                             sd.s[Dim::K][k],
                             0 ) += wm;
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a staggered momentum grid. (First
// order splines). Requires SplineValue and SplineDistance when constructing
// the spline data.
template<class ParticleMass,
         class ParticleVelocity,
         class SplineDataType,
         class GridMomentum,
         class GridMass>
KOKKOS_INLINE_FUNCTION
void p2g(
    const ParticleMass& m_p,
    const ParticleVelocity& c_p,
    const GridMomentum& mu_i,
    const GridMass& m_i,
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

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "PolyPIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "PolyPIC::p2g requires spline distance" );

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
    LinearAlgebra::Vector<value_type,8> basis;
    Vec3<value_type> distance;
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

                // Compute polynomial basis.
                basis(0) = 1.0;
                basis(1) = mapping(0);
                basis(2) = mapping(1);
                basis(3) = mapping(2);
                basis(4) = mapping(0) * mapping(1);
                basis(5) = mapping(0) * mapping(2);
                basis(6) = mapping(1) * mapping(2);
                basis(7) = mapping(0) * mapping(1) * mapping(2);

                // Contribute to momentum.
                momentum_access( sd.s[Dim::I][i],
                                 sd.s[Dim::J][j],
                                 sd.s[Dim::K][k],
                                 0 ) += wm * ~c_p.column(dim) * basis;

                // Contribute to mass.
                mass_access( sd.s[Dim::I][i],
                             sd.s[Dim::J][j],
                             sd.s[Dim::K][k],
                             0 ) += wm;
            }
}

//---------------------------------------------------------------------------//
// Interpolate collocated grid velocity to particle. (First order
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

    static_assert( SplineDataType::has_weight_values,
                   "PolyPIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_weight_physical_gradients,
                   "PolyPIC::p2g requires spline weight physical gradient" );

    // Reset particle velocity.
    c_p = 0.0;

    // Update particle.
    LinearAlgebra::Vector<value_type,8> coeff;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute coefficient values.
                coeff(0) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                coeff(1) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                coeff(2) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                coeff(3) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                coeff(4) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                coeff(5) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                coeff(6) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];
                coeff(7) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];

                // Compute particle velocity.
                c_p += coeff *
                       ~u_i(sd.s[Dim::I][i],sd.s[Dim::J][j],sd.s[Dim::K][k]);
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
                   "PolyPIC with linear coeff requires 8 velocity modes" );

    static_assert( SplineDataType::has_weight_values,
                   "PolyPIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_weight_physical_gradients,
                   "PolyPIC::p2g requires spline weight physical gradient" );

    // Get the velocity dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Reset particle velocity in the dimension we are working on.
    c_p.column(dim) = 0.0;

    // Update particle.
    LinearAlgebra::Vector<value_type,8> coeff;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute coefficient values.
                coeff(0) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                coeff(1) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
                coeff(2) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                coeff(3) = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                coeff(4) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
                coeff(5) = sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                coeff(6) = sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];
                coeff(7) = sd.g[Dim::I][i] * sd.g[Dim::J][j] * sd.g[Dim::K][k];

                // Compute particle velocity.
                c_p.column(dim) +=
                    coeff *
                    u_i(sd.s[Dim::I][i],sd.s[Dim::J][j],sd.s[Dim::K][k]);
            }
}

//---------------------------------------------------------------------------//

} // end namespace PolyPIC
} // end namespace Picasso

#endif // end PICASSO_POLYPIC_HPP
