#ifndef HARLOW_LEVELSETREDISTANCE_HPP
#define HARLOW_LEVELSETREDISTANCE_HPP

#include <Harlow_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>

namespace Harlow
{
namespace LevelSet
{
//---------------------------------------------------------------------------//
// Clamp a point into the local domain.
template<class LocalMeshType>
void clampPointToLocalDomain( const LocalMeshType& local_mesh, double x[3] )
{
    for ( int d = 0; d < 3; ++d )
    {
        x[d] = fmin( local_mesh.highCorner(Cajita::Ghost(),d),
                     fmax( local_mesh.lowCorner(Cajita::Ghost(),d), x[d] ) );
    }
}

//---------------------------------------------------------------------------//
// Hopf-Lax projection innto the ball about x.
KOKKOS_INLINE_FUNCTION
void projectToBall( const double x[3],
                    const double t_k,
                    double y[3] )
{
    // Compute the distance from the node to the argument on the ball.
    double distance = sqrt( (x[0]-y[0])*(x[0]-y[0]) +
                            (x[1]-y[1])*(x[1]-y[1]) +
                            (x[2]-y[2])*(x[2]-y[2]) );

    // Check the distance against the current secant root.
    if ( distance > t_k )
        for ( int d = 0; d < 3; ++ d )
            y[d] = x[d] - t_k * (x[d] - y[d]) / distance;
}

//---------------------------------------------------------------------------//
// Hopf-Lax projection onto the boundary of the ball about x.
KOKKOS_INLINE_FUNCTION
void projectToBallBoundary( const double x[3],
                            const double t_k,
                            double y[3] )
{
    // Compute the distance from the node to the argument on the ball.
    double distance = sqrt( (x[0]-y[0])*(x[0]-y[0]) +
                            (x[1]-y[1])*(x[1]-y[1]) +
                            (x[2]-y[2])*(x[2]-y[2]) );

    // Project to the boundary.
    if ( distance > 0.0 )
        for ( int d = 0; d < 3; ++ d )
            y[d] = x[d] - t_k * (x[d] - y[d]) / distance;
}

//---------------------------------------------------------------------------//
// Evaluate the Hopf-Lax formula for the signed distance estimate via gradient
// projection. Find the argmin of phi_0 on the ball and return phi_0 evaluated
// at the argmin.
template<class SignedDistanceView, class LocalMeshType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
double evaluate( const SignedDistanceView& phi_0,
                 const double sign,
                 const LocalMeshType& local_mesh,
                 const double x[3],
                 const double t_k,
                 const double tol,
                 const int max_iter,
                 SplineDataType& sd,
                 double y[3] )
{
    // Perform gradient projections to get the minimum argument on the ball.
    double grad_phi_0[3];
    double step;
    double step_mag;
    double y_j[3];
    for ( int i = 0; i < max_iter; ++i )
    {
        // Do a gradient projection.
        clampPointToLocalDomain( local_mesh, y );
        Cajita::evaluateSpline( local_mesh, y, sd );
        Cajita::G2P::gradient( phi_0, sd, grad_phi_0 );
        for ( int d = 0; d < 3; ++d )
            y_j[d] = y[d] - sign * sd.dx * grad_phi_0[d];

        // Project the estimate to the ball.
        projectToBall( x, t_k, y_j );

        // Update the iterate.
        step_mag = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            step = y_j[d] - y[d];
            step_mag += step*step;
            y[d] = y_j[d];
        }

        // Check for convergence.
        if ( step_mag < tol )
            break;
    }

    // Evaluate the signed distance function at the minimum argument.
    clampPointToLocalDomain( local_mesh, y );
    double phi_argmin_eval;
    Cajita::evaluateSpline( local_mesh, y, sd );
    Cajita::G2P::value( phi_0, sd, phi_argmin_eval );
    return sign * phi_argmin_eval;
}

//---------------------------------------------------------------------------//
// Evaluate the Hopf-Lax formula multiple times to find a global minimizer.
template<class SignedDistanceView,
         class LocalMeshType,
         class SplineDataType,
         class RandState>
KOKKOS_INLINE_FUNCTION
double globalMin( const SignedDistanceView& phi_0,
                  const double sign,
                  const LocalMeshType& local_mesh,
                  const double x[3],
                  const double t_k,
                  const double projection_tol,
                  const int max_projection_iter,
                  const int num_random,
                  RandState& rand_state,
                  SplineDataType& sd,
                  double y[3] )
{
    // Use project y_0 to the boundary as the first argmin evaluation.
    double y_trial[3] = { y[0], y[1], y[2] };
    projectToBallBoundary( x, t_k, y_trial );
    clampPointToLocalDomain( local_mesh, y_trial );
    double phi_trial;
    Cajita::evaluateSpline( local_mesh, y_trial, sd );
    Cajita::G2P::value( phi_0, sd, phi_trial );
    phi_trial *= sign;

    // Find the argmin from the initial point.
    double phi_min =
        evaluate( phi_0, sign, local_mesh, x, t_k,
                  projection_tol, max_projection_iter, sd, y );

    // If less than the current value assign the results as the new
    // minimum.
    if ( phi_trial < phi_min )
    {
        phi_min = phi_trial;
        for ( int d = 0; d < 3; ++d )
            y[d] = y_trial[d];
    }

    // Evaluate at random points.
    double ray[3];
    double mag;
    for ( int n = 0; n < num_random; ++n )
    {
        // Create a random ray with random length in a sphere larger than the
        // current t_k. The project back to the ball of t_k so a larger
        // fraction of points will end up on the boundary of the ball while
        // some are still in the interior.
        mag = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            ray[d] = Kokkos::rand<RandState,double>::draw( rand_state, -1.0, 1.0 );
            mag += ray[d]*ray[d];
        }
        mag = Kokkos::rand<RandState,double>::draw( rand_state, 0.0, 2.0*t_k ) / sqrt(mag);
        for ( int d = 0; d < 3; ++d )
        {
            y_trial[d] = x[d] + ray[d]*mag;
        }
        projectToBall( x, t_k, y_trial );

        // Compute the random point argmin.
        phi_trial =
            evaluate( phi_0, sign, local_mesh, x, t_k,
                      projection_tol, max_projection_iter, sd, y_trial );

        // If less than the current value assign the results as the new
        // minimum.
        if ( phi_trial < phi_min )
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
template<class EntityType,
         class SignedDistanceView,
         class LocalMeshType>
KOKKOS_INLINE_FUNCTION
double redistanceEntity( EntityType,
                         const SignedDistanceView& phi_0,
                         const LocalMeshType& local_mesh,
                         const int entity_index[3],
                         const double secant_tol,
                         const int max_secant_iter,
                         const int num_random,
                         const double projection_tol,
                         const int max_projection_iter )
{
    // Grid interpolant.
    Cajita::SplineData<double,1,EntityType> sd;

    // Random number generator.
    using rand_type =
        Kokkos::Random_XorShift64<typename SignedDistanceView::device_type>;
    rand_type rng( 0 );

    // Get the entity location.
    double x[3];
    local_mesh.coordinates( EntityType(), entity_index, x );

    // Uniform mesh spacing.
    int low_id[3] = {0, 0, 0};
    double dx = local_mesh.measure( Cajita::Edge<Dim::I>(), low_id );

    // Initial guess. The signed distance estimate is the phi value at time
    // zero.
    double t_old = 0.0;
    double phi_old =
        phi_0( entity_index[0], entity_index[1], entity_index[2], 0 );

    // Sign of phi - the secant iteration is only written in terms of positive
    // values so we need to temporarily turn negative values positive.
    double sign = copysign( 1.0, phi_old );
    phi_old *= sign;

    // First step is of size dx to get the iteration started.
    double t_new = dx;
    double phi_new;

    // Initial argmin at the entity location. The ball radius starts at 0 so
    // this is the only point that would be in the ball.
    double y[3] = { x[0], x[1], x[2] };

    // Secant step.
    double delta_t;
    double delta_t_max = 5.0 * dx;

    // Perform secant iterations to compute the signed distance.
    for ( int i = 0; i < max_secant_iter; ++i )
    {
        // Find the global minimum on the current ball.
        phi_new = globalMin( phi_0, sign, local_mesh, x, t_new,
                             projection_tol, max_projection_iter,
                             num_random, rng, sd, y );

        // Update the secant step size. Check first to see if we divide by
        // zero. If we don't then we haven't reached a minimum yet.
        if ( fabs(phi_new-phi_old) > secant_tol )
        {
            delta_t = phi_new * ( t_old - t_new ) / ( phi_new - phi_old );
        }

        // Division by zero means a possible minimum.
        else
        {
            // Check for a false local minimum. If phi isn't small enough to
            // stop the iteration step one cell farther outward to see if we
            // find a better minimum.
            if ( fabs(phi_new) > secant_tol )
            {
                delta_t = (phi_new > 0.0) ? dx : -dx;
            }

            // Otherwise we are at a real minimum so no need to step further.
            else
            {
                break;
            }
        }

        // Clamp the step size.
        delta_t = fmin( delta_t_max, fmax(-delta_t_max,delta_t) );

        // Step.
        t_old = t_new;
        t_new += delta_t;
        phi_old = phi_new;
    }

    // Return the distance.
    return t_new * sign;
}

//---------------------------------------------------------------------------//

} // end namespace LevelSet
} // end namespace Harlow

#endif // end HARLOW_LEVELSETREDISTANCE_HPP
