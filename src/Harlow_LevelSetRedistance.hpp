#ifndef HARLOW_LEVELSET_HPP
#define HARLOW_LEVELSET_HPP

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
// Hopf-Lax projection onto the ball about x.
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
// Evaluate the Hopf-Lax formula for the signed distance estimate via gradient
// projection. Find the argmin of phi_0 on the ball and return phi_0 evaluated
// at the argmin.
template<class SignedDistanceView, class LocalMeshType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
double evaluate( const SignedDistanceView& phi_0,
                 const LocalMeshType& local_mesh,
                 const double x[3],
                 const double t_k,
                 const int num_iter,
                 SplineDataType& sd,
                 double y[3] )
{
    // Perform a fixed number of gradient projections to get the minimum
    // argument on the ball.
    double grad_phi_0[3];
    for ( int i = 0; i < num_iter; ++i )
    {
        Cajita::evaluateSpline( local_mesh, y, sd );
        Cajita::G2P::gradient( phi_0, sd, grad_phi_0 );
        for ( int d = 0; d < 3; ++d )
            y[d] -= sd.dx * grad_phi_0[d];
        projectToBall( x, t_k, y );
    }

    // Evaluate the minimum argument on the ball.
    double phi_argmin_eval;
    Cajita::evaluateSpline( local_mesh, y, sd );
    Cajita::G2P::value( phi_0, sd, phi_argmin_eval );
    return phi_argmin_eval;
}

//---------------------------------------------------------------------------//
// Evaluate the Hopf-Lax formula multiple times to find a global minimizer.
template<class SignedDistanceView,
         class LocalMeshType,
         class SplineDataType,
         class RandState>
KOKKOS_INLINE_FUNCTION
double globalMin( const SignedDistanceView& phi_0,
                  const LocalMeshType& local_mesh,
                  const double x[3],
                  const double t_k,
                  const int num_eval_iter,
                  const int num_random,
                  RandState& rand_state,
                  SplineDataType& sd,
                  double y[3] )
{
    // Use y_0 as the first argmin evaluation.
    double y_trial[3] = { y[0], y[1], y[2] };
    projectToBall( x, t_k, y_trial );
    double phi_min =
        evaluate( phi_0, local_mesh, x, t_k, num_eval_iter, sd, y_trial );

    // Evaluate at random points.
    double phi_trial;
    for ( int n = 0; n < num_random; ++n )
    {
        // Create a random point.
        y_trial[0] =
            Kokkos::rand<RandState,double>::draw( rand_state, 0.0, 1.0 );
        y_trial[1] =
            Kokkos::rand<RandState,double>::draw( rand_state, 0.0, 1.0 );
        y_trial[2] =
            Kokkos::rand<RandState,double>::draw( rand_state, 0.0, 1.0 );

        // Project random point to the ball.
        projectToBall( x, t_k, y_trial );

        // Compute the random point argmin.
        phi_trial =
            evaluate( phi_0, local_mesh, x, t_k, num_eval_iter, sd, y_trial );

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
         class LocalMeshType,
         class RandState>
KOKKOS_INLINE_FUNCTION
double redistanceEntity( const SignedDistanceView& phi_0,
                         const LocalMeshType& local_mesh,
                         const int entity_index[3],
                         const int num_secant_iter,
                         const int num_random,
                         const int num_eval_iter,
                         const double init_guess = 0.0 )
{
    // Grid interpolant.
    Cajita::SplineDataType<double,1,EntityType> sd;

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

    // Compute initial guess.
    double t_old = init_guess;
    double y[3] = { 0.0, 0.0, 0.0 };
    double phi_old = globalMin( phi_0, local_mesh, x, t_old,
                                num_eval_iter, num_random, rng, sd, y );

    // First step is of size dx to get the iteration started.
    double t_new = dx
    double phi_new;

    // Secant step.
    double delta_t;
    double delta_t_max = 5.0 * dx;

    // Perform a fixed number of secant iterations to compute the signed
    // distance.
    for ( int i = 0; i < num_secant_iter )
    {
        // Find the global minimum on the current ball.
        phi_new = globalMin( phi_0, local_mesh, x, t_new,
                             num_eval_iter, num_random, rng, sd, y );

        // Update the secant step size.
        delta_t = phi_new * ( t_old - t_new ) / ( phi_new - phi_old );

        // Check that we don't violate the maximum step size. If we did,
        // restrict the step size to the maximum.
        if ( fabs(delta_t) > delta_t_max )
        {
            if ( phi_new > 0.0 )
            {
                delta_t = delta_t_max;
            }
            else
            {
                delta_t = -delta_t_max;
            }
        }

        // Step.
        t_old = t_new;
        t_new += delta_t;
        phi_old = phi_new;
    }

    // Return the distance.
    return t_new;
}

//---------------------------------------------------------------------------//

} // end namespace LevelSet
} // end namespace Harlow

#endif // end HARLOW_LEVELSET_HPP
