#ifndef HARLOW_VELOCITYINTERPOLATION_HPP
#define HARLOW_VELOCITYINTERPOLATION_HPP

#include <Cajita.hpp>

#include <Harlow_Splines.hpp>
#include <Harlow_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <type_traits>
#include <cmath>

namespace Harlow
{
namespace VelocityInterpolation
{
//---------------------------------------------------------------------------//
// PolyPIC basis
//---------------------------------------------------------------------------//
template<int ParticleOrder>
struct PolyPicBasisTraits;

template<>
struct PolyPicBasisTraits<FunctionOrder::Linear>
{
    static constexpr int num_mode = 4;
};

template<>
struct PolyPicBasisTraits<FunctionOrder::Bilinear>
{
    static constexpr int num_mode = 8;
};

template<>
struct PolyPicBasisTraits<FunctionOrder::Quadratic>
{
    static constexpr int num_mode = 27;
};

//---------------------------------------------------------------------------//
// Particle-to-Grid
//---------------------------------------------------------------------------//
// PolyPIC polynomial basis vectors.

// Linear
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicBasis(
    const Real* mapping,
    const Real* /*pln*/,
    const Real /*dx*/,
    const Real /*rdx*/,
    Real* coeffs,
    std::integral_constant<int,FunctionOrder::Linear> /*particle_order*/ )
{
    coeffs[0] = 1.0;
    coeffs[1] = mapping[Dim::I];
    coeffs[2] = mapping[Dim::J];
    coeffs[3] = mapping[Dim::K];
}

// Bilinear
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicBasis(
    const Real* mapping,
    const Real* pln,
    const Real dx,
    const Real rdx,
    Real* coeffs,
    std::integral_constant<int,FunctionOrder::Bilinear> /*particle_order*/ )
{
    polyPicBasis( mapping, pln, dx, rdx, coeffs,
                  std::integral_constant<int,FunctionOrder::Linear>() );

    coeffs[4] = mapping[Dim::I] * mapping[Dim::J];
    coeffs[5] = mapping[Dim::I] * mapping[Dim::K];
    coeffs[6] = mapping[Dim::J] * mapping[Dim::K];
    coeffs[7] = mapping[Dim::I] * mapping[Dim::J] * mapping[Dim::K];
}

// Quadratic.
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicBasis(
    const Real* mapping,
    const Real* pln,
    const Real dx,
    const Real rdx,
    Real* coeffs,
    std::integral_constant<int,FunctionOrder::Quadratic> /*particle_order*/ )
{
    polyPicBasis( mapping, pln, dx, rdx, coeffs,
                  std::integral_constant<int,FunctionOrder::Bilinear>() );

    Real rdx2_t_4 = 4.0 * rdx * rdx;
    Real dx2_d_4 = 0.25 * dx * dx;

    coeffs[8] = mapping[Dim::I] * mapping[Dim::I] -
                mapping[Dim::I] * pln[Dim::I] *
                ( 1.0 - rdx2_t_4 * pln[Dim::I] * pln[Dim::I] ) - dx2_d_4;

    coeffs[9] = mapping[Dim::J] * mapping[Dim::J] -
                mapping[Dim::J] * pln[Dim::J] *
                ( 1.0 - rdx2_t_4 * pln[Dim::J] * pln[Dim::J] ) - dx2_d_4;

    coeffs[10] = mapping[Dim::K] * mapping[Dim::K] -
                 mapping[Dim::K] * pln[Dim::K] *
                 ( 1.0 - rdx2_t_4 * pln[Dim::K] * pln[Dim::K] ) - dx2_d_4;

    coeffs[11] = coeffs[8] * coeffs[9];
    coeffs[12] = coeffs[9] * coeffs[10];
    coeffs[13] = coeffs[8] * coeffs[10];
    coeffs[14] = coeffs[11] * coeffs[10];

    coeffs[15] = coeffs[8] * mapping[Dim::J];
    coeffs[16] = coeffs[8] * mapping[Dim::K];
    coeffs[17] = coeffs[8] * mapping[Dim::J] * mapping[Dim::K];

    coeffs[18] = coeffs[9] * mapping[Dim::I];
    coeffs[19] = coeffs[9] * mapping[Dim::K];
    coeffs[20] = coeffs[9] * mapping[Dim::I] * mapping[Dim::K];

    coeffs[21] = coeffs[10] * mapping[Dim::I];
    coeffs[22] = coeffs[10] * mapping[Dim::J];
    coeffs[23] = coeffs[10] * mapping[Dim::I] * mapping[Dim::J];

    coeffs[24] = coeffs[11] * mapping[Dim::K];
    coeffs[25] = coeffs[13] * mapping[Dim::J];
    coeffs[26] = coeffs[12] * mapping[Dim::I];
}

//---------------------------------------------------------------------------//
// Affine material mapping from one time step to the next.
template<class ParticleVelocityView, class Real>
KOKKOS_INLINE_FUNCTION
void materialMapping(
    const int pid,
    const ParticleVelocityView& particle_velocity,
    const Real* distance,
    const Real dt,
    Real* mapping )
{
    // Create the affine projection operator using the velocity gradient.
    Real c[3][3];
    c[0][0] = dt * particle_velocity(pid,1,Dim::I) + 1.0;
    c[0][1] = dt * particle_velocity(pid,1,Dim::J);
    c[0][2] = dt * particle_velocity(pid,1,Dim::K);
    c[1][0] = dt * particle_velocity(pid,2,Dim::I);
    c[1][1] = dt * particle_velocity(pid,2,Dim::J) + 1.0;
    c[1][2] = dt * particle_velocity(pid,2,Dim::K);
    c[2][0] = dt * particle_velocity(pid,3,Dim::I);
    c[2][1] = dt * particle_velocity(pid,3,Dim::J);
    c[2][2] = dt * particle_velocity(pid,3,Dim::K) + 1.0;

    // Invert the operator.
    Real c_inv[3][3];
    DenseLinearAlgebra::inverse( c, c_inv );

    // Compute the mapping.
    DenseLinearAlgebra::matVecMultiply( c_inv, distance, mapping );
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate particle mass and momentum to the grid.

  \tparam SplineOrder The order of the grid basis to use for
  interpolation. Choices are: Linear, Quadratic, Cubic

  \tparam ParticleOrder The order of the polynomial expansion used to
  represent the particle velocity. Choices are: Linear, Bilinear,
  Quadratic. In general the spline order must be large enough to represent the
  particle modes (e.g. a quadratic paritcle order requires a quadratic or
  cubic spline order).

  \param block The grid block in which the particles are located.

  \param particle_position The particle positions.

  \param particle_velocity The particle velocities.

  \param grid_momentum The grid mass to update.

  \param grid_momentum The grid momentum to update.
*/
template<int SplineOrder,
         int ParticleOrder,
         class ParticlePositionView,
         class ParticleMassView,
         class ParticleVelocityView,
         class GridMassView,
         class GridMomentumView>
void particleToGrid(
    const Cajita::GridBlock& block,
    const double dt,
    const ParticlePositionView& particle_position,
    const ParticleMassView& particle_mass,
    const ParticleVelocityView& particle_velocity,
    GridMassView& grid_mass,
    GridMomentumView& grid_momentum )
{
    static_assert(
        ParticleOrder <= SplineOrder,
        "Particle order must be less than or equal to the spline order" );

    // Get the spatial coordinate value type.
    using Real = typename ParticlePositionView::value_type;

    // Get the spline.
    using Basis = Spline<SplineOrder>;

    // Reset the grid mass and momentum.
    Kokkos::deep_copy( grid_mass, 0.0 );
    Kokkos::deep_copy( grid_momentum, 0.0 );

    // Get the cell size.
    auto dx = block.cellSize();
    auto rdx = block.inverseCellSize();

    // Extract the low corner of the grid.
    auto low_x = block.lowCorner(Dim::I);
    auto low_y = block.lowCorner(Dim::J);
    auto low_z = block.lowCorner(Dim::K);

    // Get the stencil size.
    const int ns = Basis::num_knot;

    // Get the number of velocity modes.
    const int num_mode = PolyPicBasisTraits<ParticleOrder>::num_mode;

    // Create a scatter-view of the destination data.
    auto grid_mass_sv = Kokkos::Experimental::create_scatter_view( grid_mass );
    auto grid_mom_sv = Kokkos::Experimental::create_scatter_view( grid_momentum );

    // Loop over particles and interpolate.
    Kokkos::parallel_for(
        "mass_momentum_p2g",
        Kokkos::RangePolicy<typename ParticleVelocityView::execution_space>(
            0, particle_position.extent(0) ),
        KOKKOS_LAMBDA( const int p )
        {
            // Create the interpolation stencil.
            int offsets[ns];
            Basis::stencil( offsets );

            // Compute the logical space coordinates of the particle.
            Real plx[3] =
                { Basis::mapToLogicalGrid( particle_position(p,Dim::I), rdx, low_x ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::J), rdx, low_y ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::K), rdx, low_z ) };

            // Get the logical index of the particle.
            int pli[3] = { int(plx[Dim::I]), int(plx[Dim::J]), int(plx[Dim::K]) };

            // Get the physical location of the particle in the reference
            // frame of the stencil. This is the distance between the particle
            // and the closest node.
            Real pln[3] = { (plx[Dim::I] - pli[Dim::I]) * dx,
                            (plx[Dim::J] - pli[Dim::J]) * dx,
                            (plx[Dim::K] - pli[Dim::K]) * dx };

            // Get the particle weights.
            Real wi[ns];
            Basis::value( plx[Dim::I], wi );
            Real wj[ns];
            Basis::value( plx[Dim::J], wj );
            Real wk[ns];
            Basis::value( plx[Dim::K], wk );

            // Access the scatter views.
            auto grid_mass_sv_data = grid_mass_sv.access();
            auto grid_mom_sv_data = grid_mom_sv.access();

            // Loop over the neighboring nodes and add the field
            // contribution.
            int nidx[3];
            Real distance[3];
            Real mapping[3];
            Real mode_coeffs[num_mode];
            Real p_mom[3];
            for ( int i = 0; i < ns; ++i )
                for ( int j = 0; j < ns; ++j )
                    for ( int k = 0; k < ns; ++k )
                    {
                        // Compute the node index.
                        nidx[Dim::I] = pli[Dim::I] + offsets[i];
                        nidx[Dim::J] = pli[Dim::J] + offsets[j];
                        nidx[Dim::K] = pli[Dim::K] + offsets[k];

                        // Compute the spline weight times the particle mass.
                        Real wm = wi[i] * wj[j] * wk[k] * particle_mass(p);

                        // Add mass contribution to the node.
                        grid_mass_sv_data(
                            nidx[Dim::I], nidx[Dim::J], nidx[Dim::K], 0 ) += wm;

                        // Compute distance to node.
                        distance[Dim::I] =
                            (low_x + nidx[Dim::I] * dx) - particle_position(p,Dim::I);
                        distance[Dim::J] =
                            (low_y + nidx[Dim::J] * dx) - particle_position(p,Dim::J);
                        distance[Dim::K] =
                            (low_z + nidx[Dim::K] * dx) - particle_position(p,Dim::K);

                        // Compute the material mapping.
                        materialMapping( p, particle_velocity, distance, dt, mapping );

                        // Get the particle basis cofficients for this node.
                        polyPicBasis( mapping, pln, dx, rdx, mode_coeffs,
                                      std::integral_constant<int,ParticleOrder>() );

                        // Compute the particle momentum.
                        p_mom[Dim::I] = 0.0;
                        p_mom[Dim::J] = 0.0;
                        p_mom[Dim::K] = 0.0;
                        for ( std::size_t n = 0; n < num_mode; ++n )
                            for ( std::size_t d = 0; d < 3; ++d )
                                p_mom[d] += mode_coeffs[n] * particle_velocity(p,n,d);

                        // Add contribution of each velocity mode to the grid
                        // momentum.
                        for ( std::size_t d = 0; d < 3; ++d )
                            grid_mom_sv_data( nidx[Dim::I],
                                              nidx[Dim::J],
                                              nidx[Dim::K],
                                              d ) +=
                                p_mom[d] * wm;
                    }

        } );

    // Apply the contribution of the scatter view.
    Kokkos::Experimental::contribute( grid_mass, grid_mass_sv );
    Kokkos::Experimental::contribute( grid_momentum, grid_mom_sv );
}

//---------------------------------------------------------------------------//
// Grid-to-Particle
//---------------------------------------------------------------------------//
// PolyPIC modal weights with a quadratic particle polynomial basis. Note that
// these weights are computed with respect to the logical distance between the
// particle and the node

// Linear
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights(
    const Real* distance,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const int i,
    const int j,
    const int k,
    Real* mode_weight,
    std::integral_constant<int,FunctionOrder::Linear> /*particle_order*/ )
{
    mode_weight[0] = wi[i] * wj[j] * wk[k];

    mode_weight[1] = mode_weight[0] * distance[Dim::I] * 4.0;
    mode_weight[2] = mode_weight[0] * distance[Dim::J] * 4.0;
    mode_weight[3] = mode_weight[0] * distance[Dim::K] * 4.0;
}

// Bilinear
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights(
    const Real* distance,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const int i,
    const int j,
    const int k,
    Real* mode_weight,
    std::integral_constant<int,FunctionOrder::Bilinear> /*particle_order*/ )
{
    polyPicModeWeights( distance, wi, wj, wk, i, j, k, mode_weight,
                        std::integral_constant<int,FunctionOrder::Linear>() );

    mode_weight[4] = mode_weight[0] * distance[Dim::I] * distance[Dim::J] * 16.0;
    mode_weight[5] = mode_weight[0] * distance[Dim::J] * distance[Dim::K] * 16.0;
    mode_weight[6] = mode_weight[0] * distance[Dim::I] * distance[Dim::K] * 16.0;

    mode_weight[7] = mode_weight[0] *
                     distance[Dim::I] * distance[Dim::J] * distance[Dim::K] * 64.0;
}

// Quadratic
template<class Real>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights(
    const Real* distance,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const int i,
    const int j,
    const int k,
    Real* mode_weight,
    std::integral_constant<int,FunctionOrder::Quadratic> /*particle_order*/ )
{
    polyPicModeWeights( distance, wi, wj, wk, i, j, k, mode_weight,
                        std::integral_constant<int,FunctionOrder::Bilinear>() );

    Real mod_i = ( 1 == i ) ? -2 : 1;
    Real mod_j = ( 1 == j ) ? -2 : 1;
    Real mod_k = ( 1 == k ) ? -2 : 1;

    mode_weight[8]  = wj[j] * wk[k] * mod_i * 4.0;
    mode_weight[9]  = wi[i] * wk[k] * mod_j * 4.0;
    mode_weight[10] = wi[i] * wj[j] * mod_k * 4.0;

    mode_weight[11] = wk[k] * mod_i * mod_j * 16.0;
    mode_weight[12] = wj[j] * mod_i * mod_k * 16.0;
    mode_weight[13] = wi[i] * mod_j * mod_k * 16.0;

    mode_weight[14] = mod_i * mod_j * mod_k * 64.0;

    mode_weight[15] = wj[j] * wk[k] * distance[Dim::J] * mod_i * 16.0;
    mode_weight[16] = wj[j] * wk[k] * distance[Dim::K] * mod_i * 16.0;
    mode_weight[17] = wj[j] * wk[k] *
                      distance[Dim::J] * distance[Dim::K] * mod_i * 64.0;

    mode_weight[18] = wi[i] * wk[k] * distance[Dim::I] * mod_j * 16.0;
    mode_weight[19] = wi[i] * wk[k] * distance[Dim::K] * mod_j * 16.0;
    mode_weight[20] = wi[i] * wk[k] *
                      distance[Dim::I] * distance[Dim::K] * mod_j * 64.0;

    mode_weight[21] = wi[i] * wj[j] * distance[Dim::I] * mod_k * 16.0;
    mode_weight[22] = wi[i] * wj[j] * distance[Dim::J] * mod_k * 16.0;
    mode_weight[23] = wi[i] * wj[j] *
                      distance[Dim::I] * distance[Dim::J] * mod_k * 64.0;

    mode_weight[24] = wk[k] * distance[Dim::K] * mod_i * mod_j * 64.0;
    mode_weight[25] = wj[j] * distance[Dim::J] * mod_i * mod_k * 64.0;
    mode_weight[26] = wi[i] * distance[Dim::I] * mod_j * mod_k * 64.0;
}

//---------------------------------------------------------------------------//
/*
  \brief Interpolate grid velocity to the particles using PolyPIC

  \tparam SplineOrder The order of the grid basis to use for
  interpolation. Choices are: Linear, Quadratic, Cubic

  \tparam ParticleOrder The order of the polynomial expansion used to
  represent the particle velocity. Choices are: Linear, Bilinear,
  Quadratic. In general the spline order must be large enough to represent the
  particle modes (e.g. a quadratic paritcle order requires a quadratic or
  cubic spline order).

  \param block The grid block in which the particles are located.

  \param grid_velocity The velocity of the grid.

  \param particle_position The particle positions.

  \param particle_velocity The particle velocities to update.
*/
template<int SplineOrder,
         int ParticleOrder,
         class ParticlePositionView,
         class GridVelocityView,
         class ParticleVelocityView>
void gridToParticle(
    const Cajita::GridBlock& block,
    const GridVelocityView& grid_velocity,
    const ParticlePositionView& particle_position,
    ParticleVelocityView& particle_velocity )
{
    static_assert(
        ParticleOrder <= SplineOrder,
        "Particle order must be less than or equal to the spline order" );

    // Get the spatial coordinate value type.
    using Real = typename ParticlePositionView::value_type;

    // Get the spline.
    using Basis = Spline<SplineOrder>;

    // Reset the particle velocity
    Kokkos::deep_copy( particle_velocity, 0.0 );

    // Get the cell size.
    auto rdx = block.inverseCellSize();

    // Extract the low corner of the grid.
    auto low_x = block.lowCorner(Dim::I);
    auto low_y = block.lowCorner(Dim::J);
    auto low_z = block.lowCorner(Dim::K);

    // Get the stencil size.
    const int ns = Basis::num_knot;

    // Get the number of velocity modes.
    const int num_mode = PolyPicBasisTraits<ParticleOrder>::num_mode;

    // Loop over particles and interpolate.
    Kokkos::parallel_for(
        "velocity_g2p",
        Kokkos::RangePolicy<typename ParticleVelocityView::execution_space>(
            0, particle_position.extent(0) ),
        KOKKOS_LAMBDA( const int p )
        {
            // Create the interpolation stencil.
            int offsets[ns];
            Basis::stencil( offsets );

            // Compute the logical space coordinates of the particle.
            Real plx[3] =
                { Basis::mapToLogicalGrid( particle_position(p,Dim::I), rdx, low_x ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::J), rdx, low_y ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::K), rdx, low_z ) };

            // Get the logical index of the particle.
            int pli[3] = { int(plx[0]), int(plx[1]), int(plx[2]) };

            // Get the particle weights.
            Real wi[ns];
            Basis::value( plx[Dim::I], wi );
            Real wj[ns];
            Basis::value( plx[Dim::J], wj );
            Real wk[ns];
            Basis::value( plx[Dim::K], wk );

            // Loop over the neighboring nodes and add the field
            // contribution.
            Real mode_weight[num_mode];
            int nidx[3];
            Real distance[3];
            for ( int i = 0; i < ns; ++i )
                for ( int j = 0; j < ns; ++j )
                    for ( int k = 0; k < ns; ++k )
                    {
                        // Compute the node index.
                        nidx[Dim::I] = pli[Dim::I] + offsets[i];
                        nidx[Dim::J] = pli[Dim::J] + offsets[j];
                        nidx[Dim::K] = pli[Dim::K] + offsets[k];

                        // Compute the logical distance to node.
                        distance[Dim::I] = nidx[Dim::I] - plx[Dim::I];
                        distance[Dim::J] = nidx[Dim::J] - plx[Dim::J];
                        distance[Dim::K] = nidx[Dim::K] - plx[Dim::K];

                        // Get the particle basis weights for this node.
                        polyPicModeWeights( distance,
                                            wi, wj, wk,
                                            i, j, k,
                                            mode_weight,
                                            std::integral_constant<int,ParticleOrder>() );

                        // Add contribution of this node to each velocity mode.
                        for ( std::size_t n = 0; n < num_mode; ++n )
                            for ( std::size_t d = 0; d < 3; ++d )
                                particle_velocity( p, n, d ) +=
                                    mode_weight[n] *
                                    grid_velocity( nidx[Dim::I],
                                                   nidx[Dim::J],
                                                   nidx[Dim::K],
                                                   d );
                    }
        } );
}

//---------------------------------------------------------------------------//

} // end namespace VelocityInterpolation
} // end namespace Harlow

#endif // end HARLOW_VELOCITYINTERPOLATION_HPP
