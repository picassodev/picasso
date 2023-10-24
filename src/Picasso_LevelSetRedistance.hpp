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

#ifndef PICASSO_LEVELSETREDISTANCE_HPP
#define PICASSO_LEVELSETREDISTANCE_HPP

#include <Picasso_Types.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <limits>

namespace Picasso
{
namespace LevelSetRedistance
{
//---------------------------------------------------------------------------//
// Distance between two points. Return the distance vector as well.
KOKKOS_INLINE_FUNCTION
double distance( const double x[3], const double y[3], double z[3] )
{
    for ( int d = 0; d < 3; ++d )
        z[d] = x[d] - y[d];
    return sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] );
}

//---------------------------------------------------------------------------//
// Clamp a point into the local domain.
template <class LocalMeshType>
KOKKOS_INLINE_FUNCTION void
clampPointToLocalDomain( const LocalMeshType& local_mesh, const double dx,
                         double x[3] )
{
    for ( int d = 0; d < 3; ++d )
    {
        x[d] = Kokkos::fmin(
            local_mesh.highCorner( Cabana::Grid::Ghost(), d ) - 0.001 * dx,
            Kokkos::fmax( local_mesh.lowCorner( Cabana::Grid::Ghost(), d ) +
                              0.001 * dx,
                          x[d] ) );
    }
}

//---------------------------------------------------------------------------//
// Hopf-Lax projection into the ball of radius t_k about x.
KOKKOS_INLINE_FUNCTION
void projectToBall( const double x[3], const double t_k, double y[3] )
{
    // Compute the dist from the node to the argument on the ball.
    double z[3];
    double dist = distance( x, y, z );

    // Check the dist against the current secant root and project to the
    // boundary if outside the ball.
    if ( dist > t_k )
        for ( int d = 0; d < 3; ++d )
            y[d] = x[d] - t_k * z[d] / dist;
}

//---------------------------------------------------------------------------//
// Evaluate the Hopf-Lax formula for the signed distance estimate via gradient
// projection. Find the argmin of phi_0 on the ball and return phi_0 evaluated
// at the argmin.
template <class SignedDistanceView, class LocalMeshType, class SplineDataType>
KOKKOS_INLINE_FUNCTION double
evaluate( const SignedDistanceView& phi_0, const double sign,
          const LocalMeshType& local_mesh, const double x[3], const double t_k,
          const double tol, const int max_iter, SplineDataType& sd,
          double y[3] )
{
    // Get the cell size. We assume a uniform cell size in this implementation.
    Cabana::Grid::evaluateSpline( local_mesh, y, sd );
    double sign_dx = sign * sd.dx[0];

    // Perform gradient projections to get the minimum argument on the ball.
    double grad_phi_0[3];
    double y_old[3] = { y[0], y[1], y[2] };
    double error_num, error_div;
    double tol_sqr = tol * tol;
    for ( int i = 0; i < max_iter; ++i )
    {
        // Move the point back into the local domain if necessary.
        clampPointToLocalDomain( local_mesh, sd.dx[0], y_old );

        // Do a gradient projection.
        Cabana::Grid::evaluateSpline( local_mesh, y_old, sd );
        Cabana::Grid::G2P::gradient( phi_0, sd, grad_phi_0 );
        for ( int d = 0; d < 3; ++d )
            y[d] = y_old[d] - sign_dx * grad_phi_0[d];

        // Project the estimate to the ball.
        projectToBall( x, t_k, y );

        // Update the iterate.
        error_num = 0.0;
        error_div = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            error_num += ( y[d] - y_old[d] ) * ( y[d] - y_old[d] );
            error_div += y_old[d] * y_old[d];
            y_old[d] = y[d];
        }

        // Check for a stationary point.
        if ( error_num / error_div < tol_sqr )
            break;
    }

    // Evaluate the signed distance function at the minimum argument.
    clampPointToLocalDomain( local_mesh, sd.dx[0], y );
    double phi_argmin_eval;
    Cabana::Grid::evaluateSpline( local_mesh, y, sd );
    Cabana::Grid::G2P::value( phi_0, sd, phi_argmin_eval );
    return sign * phi_argmin_eval;
}

//---------------------------------------------------------------------------//
// Evaluate the Hopf-Lax formula multiple times to find a global minimizer for
// the given secant iterate.
template <class SignedDistanceView, class LocalMeshType, class SplineDataType,
          class RandState>
KOKKOS_INLINE_FUNCTION double
globalMin( const SignedDistanceView& phi_0, const double sign,
           const LocalMeshType& local_mesh, const double x[3], const double t_k,
           const double projection_tol, const int max_projection_iter,
           const int num_random, RandState& rand_state, SplineDataType& sd,
           double y[3] )
{
    // Estimate the argmin from the initial point.
    projectToBall( x, t_k, y );
    double phi_min = evaluate( phi_0, sign, local_mesh, x, t_k, projection_tol,
                               max_projection_iter, sd, y );

    // Compare the initial estimate to random points in the ball.
    double y_trial[3];
    double phi_trial;
    double ray[3];
    double mag;
    for ( int n = 0; n < num_random; ++n )
    {
        // Create a random point in a sphere about x larger than the current
        // t_k. Then project back to the ball of t_k so a larger fraction of
        // points will end up on the boundary of the ball while some are still
        // in the interior. The performance of this could likely be improved
        // in the future.
        mag = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            ray[d] =
                Kokkos::rand<RandState, double>::draw( rand_state, -1.0, 1.0 );
            mag += ray[d] * ray[d];
        }
        mag = t_k *
              ( 1 - Kokkos::exp( Kokkos::rand<RandState, double>::draw(
                        rand_state, -4.0, 0.0 ) ) ) /
              sqrt( mag );
        for ( int d = 0; d < 3; ++d )
        {
            y_trial[d] = x[d] + ray[d] * mag;
        }

        // Compute the random point argmin.
        phi_trial = evaluate( phi_0, sign, local_mesh, x, t_k, projection_tol,
                              max_projection_iter, sd, y_trial );

        // If less than the current value assign the results as the new
        // minimum.
        if ( Kokkos::fabs( phi_trial ) < Kokkos::fabs( phi_min ) )
        {
            phi_min = phi_trial;
            for ( int d = 0; d < 3; ++d )
                y[d] = y_trial[d];
        }
    }

    // Return the global minimum value.
    return phi_min;
}

//---------------------------------------------------------------------------//
// Redistance a signed distance function at a single entity with the Hopf-Lax
// method.
//
// NOTE - this implementation is specifically implemented for uniform
// grids. For adaptive grids that should be fine as we can map from the
// distance function from the natural system to the reference system. For
// uniform grids with general distances in each cell dimension, we will need
// some small adjustments. Particularly to how we use dx in different places.
//
// NOTE - If needed, this function could also be easily modified to return the
// normal field per section 2.4 in the reference.
template <class EntityType, class SignedDistanceView, class LocalMeshType>
KOKKOS_INLINE_FUNCTION double
redistanceEntity( EntityType, const SignedDistanceView& phi_0,
                  const LocalMeshType& local_mesh, const int entity_index[3],
                  const double secant_tol, const int max_secant_iter,
                  const int num_random, const double projection_tol,
                  const int max_projection_iter )
{
    // Grid interpolant.
    using SplineTags = Cabana::Grid::SplineDataMemberTypes<
        Cabana::Grid::SplineWeightValues,
        Cabana::Grid::SplineWeightPhysicalGradients,
        Cabana::Grid::SplinePhysicalCellSize>;
    Cabana::Grid::SplineData<double, 1, 3, EntityType, SplineTags> sd;

    // Random number generator.
    using rand_type =
        Kokkos::Random_XorShift64<typename SignedDistanceView::device_type>;
    rand_type rng( 0 );

    // Get the entity location.
    double x[3];
    local_mesh.coordinates( EntityType(), entity_index, x );

    // Uniform mesh spacing.
    int low_id[3] = { 0, 0, 0 };
    double dx = local_mesh.measure( Cabana::Grid::Edge<Dim::I>(), low_id );

    // Initial guess. The signed distance estimate is the phi value at time
    // zero.
    double t_old = 0.0;
    double phi_old =
        phi_0( entity_index[0], entity_index[1], entity_index[2], 0 );

    // Sign of phi - the secant iteration is only written in terms of positive
    // values so we need to temporarily turn negative values positive.
    double sign = copysign( 1.0, phi_old );
    phi_old *= sign;

    // First step is of size tol*dx to get the iteration started.
    double t_new = secant_tol * dx;
    double phi_new;

    // Initial argmin at the entity location. The ball radius starts at 0 so
    // this is the only point that would be in the ball.
    double y[3] = { x[0], x[1], x[2] };

    // Secant step.
    double delta_t;
    double delta_t_max = 5.0 * dx;

    // Compute the secant initial iterate.
    phi_new = globalMin( phi_0, sign, local_mesh, x, t_new, projection_tol,
                         max_projection_iter, num_random, rng, sd, y );

    // Check for convergence.
    if ( Kokkos::fabs( phi_new ) < dx * secant_tol )
        return sign * t_new;

    // Perform secant iterations to compute the signed distance. Stop when
    // either we reach our convergence tolerance or we hit a maximum number of
    // iterations.
    for ( int i = 0; i < max_secant_iter; ++i )
    {
        // Compute secant step size.
        delta_t = phi_new * ( t_old - t_new ) / ( phi_new - phi_old );

        // Clamp the step size if it exceeds the maximum step size. This
        // is needed to detect near division-by-zero resulting from a local
        // minimum.
        if ( Kokkos::fabs( delta_t ) > delta_t_max )
        {
            delta_t = ( phi_new > 0.0 ) ? delta_t_max : -delta_t_max;
        }

        // Step.
        t_old = t_new;
        t_new += delta_t;
        phi_old = phi_new;

        // Find the global minimum on the current ball.
        phi_new = globalMin( phi_0, sign, local_mesh, x, t_new, projection_tol,
                             max_projection_iter, num_random, rng, sd, y );

        // Check for convergence.
        if ( Kokkos::fabs( phi_new ) < dx * secant_tol )
        {
            break;
        }
    }

    // Return the signed distance.
    return sign * t_new;
}

//---------------------------------------------------------------------------//

} // end namespace LevelSetRedistance
} // end namespace Picasso

#endif // end PICASSO_LEVELSETREDISTANCE_HPP
