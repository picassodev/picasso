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

#ifndef PICASSO_APIC_HPP
#define PICASSO_APIC_HPP

#include <Cajita.hpp>

#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_Types.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Affine Particle-in-Cell (APIC)
// Currently only defined for uniform grids.
//---------------------------------------------------------------------------//
namespace APIC
{
//---------------------------------------------------------------------------//
// Inertial tensor scale factor.
template <class SplineDataType>
typename SplineDataType::scalar_type KOKKOS_INLINE_FUNCTION inertialScaling(
    const SplineDataType& sd,
    typename std::enable_if<( 2 == SplineDataType::order ), void*>::type = 0 )
{
    return 4.0 / ( sd.dx[0] * sd.dx[0] );
}

template <class SplineDataType>
typename SplineDataType::scalar_type KOKKOS_INLINE_FUNCTION inertialScaling(
    const SplineDataType& sd,
    typename std::enable_if<( 3 == SplineDataType::order ), void*>::type = 0 )
{
    return 3.0 / ( sd.dx[0] * sd.dx[0] );
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a collocated momentum
// grid. (Second and Third order splines). Requires SplineValue,
// SplineDistance, and SplinePhysicalCellSize when constructing the spline
// data.
template <class ParticleMass, class ParticleField, class SplineDataType,
          class GridField, class GridMass>
KOKKOS_INLINE_FUNCTION void
p2g( const ParticleMass& m_p, const ParticleField& c_p, const GridField& mu_i,
     const GridMass& m_i, const SplineDataType& sd,
     typename std::enable_if<
         ( ( Cajita::isNode<typename SplineDataType::entity_type>::value ||
             Cajita::isCell<typename SplineDataType::entity_type>::value ) &&
           ( SplineDataType::order == 2 || SplineDataType::order == 3 ) ),
         void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridField>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto field_access = mu_i.access();

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "APIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "APIC::p2g requires spline distance" );

    static_assert( SplineDataType::has_physical_cell_size,
                   "APIC::p2g requires spline physical cell size" );

    using value_type = typename GridField::original_value_type;

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        B_p( d, 0 ) = c_p( 1, d );
        B_p( d, 1 ) = c_p( 2, d );
        B_p( d, 2 ) = c_p( 3, d );
    }

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
                distance( Dim::I ) = sd.d[Dim::I][i];
                distance( Dim::J ) = sd.d[Dim::J][j];
                distance( Dim::K ) = sd.d[Dim::K][k];

                // Compute the action of B_p on the distance scaled by the
                // intertial tensor scaling factor.
                auto D_p_inv_B_p_d = D_p_inv * B_p * distance;

                // Weight times mass.
                wm_ip =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;

                // Interpolate particle momentum to the entity.
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
                for ( int d = 0; d < ncomp; ++d )
                    field_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                  sd.s[Dim::K][k], d ) +=
                        wm_ip * ( c_p( 0, d ) + D_p_inv_B_p_d( d ) );

                // Interpolate particle mass to the entity.
                mass_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += wm_ip;
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a staggered momentum
// grid. (Second and Third order splines). Requires SplineValue,
// SplineDistance, and SplinePhysicalCellSize when constructing the spline
// data.
template <class ParticleMass, class ParticleField, class SplineDataType,
          class GridField, class GridMass>
KOKKOS_INLINE_FUNCTION void
p2g( const ParticleMass& m_p, const ParticleField& c_p, const GridField& mu_i,
     const GridMass& m_i, const SplineDataType& sd,
     typename std::enable_if<
         ( ( Cajita::isEdge<typename SplineDataType::entity_type>::value ||
             Cajita::isFace<typename SplineDataType::entity_type>::value ) &&
           ( SplineDataType::order == 2 || SplineDataType::order == 3 ) ),
         void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridField>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto field_access = mu_i.access();

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "APIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "APIC::p2g requires spline distance" );

    static_assert( SplineDataType::has_physical_cell_size,
                   "APIC::p2g requires spline physical cell size" );

    using value_type = typename GridField::original_value_type;

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    static_assert( 3 == ParticleField::extent_1,
                   "APIC requires 3 space dimensions" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        B_p( d, 0 ) = c_p( 1, d );
        B_p( d, 1 ) = c_p( 2, d );
        B_p( d, 2 ) = c_p( 3, d );
    }

    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

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
                distance( Dim::I ) = sd.d[Dim::I][i];
                distance( Dim::J ) = sd.d[Dim::J][j];
                distance( Dim::K ) = sd.d[Dim::K][k];

                // Weight times mass.
                wm_ip =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;

                // Interpolate particle momentum to the entity.
                field_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                              0 ) +=
                    wm_ip *
                    ( c_p( 0, dim ) + D_p_inv * ~B_p.row( dim ) * distance );

                // Interpolate particle mass to the entity.
                mass_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += wm_ip;
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a collocated momentum
// grid. (First order splines). Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ParticleMass, class ParticleField, class SplineDataType,
          class GridField, class GridMass>
KOKKOS_INLINE_FUNCTION void
p2g( const ParticleMass& m_p, const ParticleField& c_p, const GridField& mu_i,
     const GridMass& m_i, const SplineDataType& sd,
     typename std::enable_if<
         ( ( Cajita::isNode<typename SplineDataType::entity_type>::value ||
             Cajita::isCell<typename SplineDataType::entity_type>::value ) &&
           ( SplineDataType::order == 1 ) ),
         void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridField>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto field_access = mu_i.access();

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "APIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_weight_physical_gradients,
                   "APIC::p2g requires spline weight gradients" );

    using value_type = typename GridField::original_value_type;

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        B_p( d, 0 ) = c_p( 1, d );
        B_p( d, 1 ) = c_p( 2, d );
        B_p( d, 2 ) = c_p( 3, d );
    }
    // Project momentum.
    Vec3<value_type> gm_ip;
    value_type wm_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm_ip =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;

                // Weight gradient times mass.
                gm_ip( 0 ) =
                    sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;
                gm_ip( 1 ) =
                    sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k] * m_p;
                gm_ip( 2 ) =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k] * m_p;

                // Compute the action of B_p on the gradient.
                auto B_g_d = B_p * gm_ip;

                // Interpolate particle momentum to the entity.
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
                for ( int d = 0; d < ncomp; ++d )
                    field_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                  sd.s[Dim::K][k], d ) +=
                        wm_ip * c_p( 0, d ) + B_g_d( d );

                // Interpolate particle mass to the entity.
                mass_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += wm_ip;
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum and mass to a staggered momentum grid. (First
// order splines). Requires SplineValue and SplineGradient when constructing
// the spline data.
template <class ParticleMass, class ParticleField, class SplineDataType,
          class GridField, class GridMass>
KOKKOS_INLINE_FUNCTION void
p2g( const ParticleMass& m_p, const ParticleField& c_p, const GridField& mu_i,
     const GridMass& m_i, const SplineDataType& sd,
     typename std::enable_if<
         ( ( Cajita::isEdge<typename SplineDataType::entity_type>::value ||
             Cajita::isFace<typename SplineDataType::entity_type>::value ) &&
           ( SplineDataType::order == 1 ) ),
         void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<GridField>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto field_access = mu_i.access();

    static_assert( Cajita::P2G::is_scatter_view<GridMass>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto mass_access = m_i.access();

    static_assert( SplineDataType::has_weight_values,
                   "APIC::p2g requires spline weight values" );

    static_assert( SplineDataType::has_weight_physical_gradients,
                   "APIC::p2g requires spline weight gradients" );

    using value_type = typename GridField::original_value_type;

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    static_assert( 3 == ParticleField::extent_1,
                   "APIC requires 3 space dimensions" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        B_p( d, 0 ) = c_p( 1, d );
        B_p( d, 1 ) = c_p( 2, d );
        B_p( d, 2 ) = c_p( 3, d );
    }
    // Get the momentum dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Project momentum.
    Vec3<value_type> gm_ip;
    value_type wm_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Weight times mass.
                wm_ip =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;

                // Weight gradient times mass.
                gm_ip( 0 ) =
                    sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * m_p;
                gm_ip( 1 ) =
                    sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k] * m_p;
                gm_ip( 2 ) =
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k] * m_p;

                // Interpolate particle momentum to the entity.
                field_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                              0 ) +=
                    wm_ip * c_p( 0, dim ) + ~B_p.row( dim ) * gm_ip;

                // Interpolate particle mass to the entity.
                mass_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += wm_ip;
            }
}

//---------------------------------------------------------------------------//
// Interpolate collocated grid velocity to the particle. Requires SplineValue
// and SplineDistance when constructing the spline data.
template <class GridVelocity, class SplineDataType, class ParticleField>
KOKKOS_INLINE_FUNCTION void
g2p( const GridVelocity& u_i, ParticleField& c_p, const SplineDataType& sd,
     typename std::enable_if<
         ( Cajita::isNode<typename SplineDataType::entity_type>::value ||
           Cajita::isCell<typename SplineDataType::entity_type>::value ),
         void*>::type = 0 )
{
    using value_type = typename GridVelocity::value_type;

    static_assert( SplineDataType::has_weight_values,
                   "APIC::g2p requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "APIC::g2p requires spline distance" );

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Vector<value_type, ncomp> u_p;
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;

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
                auto u_e =
                    u_i( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k] );

                // Update velocity.
                u_p += w_ip * u_e;

                // Physical distance to entity.
                distance( Dim::I ) = sd.d[Dim::I][i];
                distance( Dim::J ) = sd.d[Dim::J][j];
                distance( Dim::K ) = sd.d[Dim::K][k];

                // Update affine matrix.
                B_p += w_ip * u_e * ~distance;
            }

    // update ParticleField
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        c_p( 0, d ) = u_p( d );
        c_p( 1, d ) = B_p( d, 0 );
        c_p( 2, d ) = B_p( d, 1 );
        c_p( 3, d ) = B_p( d, 2 );
    }
}

//---------------------------------------------------------------------------//
// Interpolate staggered grid velocity to the particle. Requires SplineValue
// and SplineDistance when constructing the spline data.
template <class GridVelocity, class SplineDataType, class ParticleField>
KOKKOS_INLINE_FUNCTION void
g2p( const GridVelocity& u_i, ParticleField& c_p, const SplineDataType& sd,
     typename std::enable_if<
         ( Cajita::isEdge<typename SplineDataType::entity_type>::value ||
           Cajita::isFace<typename SplineDataType::entity_type>::value ),
         void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "APIC::g2p requires spline weight values" );

    static_assert( SplineDataType::has_physical_distance,
                   "APIC::g2p requires spline distance" );

    using value_type = typename GridVelocity::value_type;

    static_assert( 4 == ParticleField::extent_0, "APIC 4 modes" );

    static_assert( 3 == ParticleField::extent_1,
                   "APIC requires 3 space dimensions" );

    // Number of field components.
    static constexpr int ncomp = ParticleField::extent_1;

    // Affine Matrix
    LinearAlgebra::Vector<value_type, ncomp> u_p;
    LinearAlgebra::Matrix<value_type, ncomp, 3> B_p;

    // Get the velocity dimension we are working on.
    const int dim = SplineDataType::entity_type::dim;

    // Reset the particle values.
    u_p( dim ) = 0.0;
    B_p.row( dim ) = 0.0;

    // Update particle.
    Vec3<value_type> distance;
    value_type w_ip;
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Projection weight.
                w_ip = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Update velocity.
                u_p( dim ) += w_ip * u_i( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                          sd.s[Dim::K][k] );

                // Physical distance to entity.
                distance( Dim::I ) = sd.d[Dim::I][i];
                distance( Dim::J ) = sd.d[Dim::J][j];
                distance( Dim::K ) = sd.d[Dim::K][k];

                // Update affine matrix.
                B_p.row( dim ) +=
                    w_ip *
                    u_i( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k] ) *
                    distance;
            }

    // update ParticleField
    for ( int d = 0; d < ncomp; ++d )
    {
        // transpose of c_p is already taken accounted in B_p
        c_p( 0, d ) = u_p( d );
        c_p( 1, d ) = B_p( d, 0 );
        c_p( 2, d ) = B_p( d, 1 );
        c_p( 3, d ) = B_p( d, 2 );
    }
}

//---------------------------------------------------------------------------//

} // end namespace APIC

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_APIC_HPP
