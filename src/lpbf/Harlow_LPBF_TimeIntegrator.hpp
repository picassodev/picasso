#ifndef PICASSO_LPBF_TIMEINTEGRATOR_HPP
#define PICASSO_LPBF_TIMEINTEGRATOR_HPP

#include <Picasso_LPBF_AuxiliaryFieldTypes.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace Picasso
{
namespace LPBF
{
namespace TimeIntegrator
{
//---------------------------------------------------------------------------//
// Explicit time step.
template<class ExecutionSpace, class ProblemManagerType>
void step( const ExecutionSpace& exec_space,
           const ProblemManagerType& pm,
           const double time )
{
    // Primary state manager.
    const auto& state = *(pm.state());

    // Auxiliary field manager.
    const auto& aux = *(pm.auxiliaryFields());

    // Particle list.
    const auto& pl = *(pm.particleList());

    // Level set.
    const auto& ls = *(pm.levelSet());

    // Laser source.
    const auto& laser = pm.laserSource();

    // Material properties.
    auto kappa = pm.thermalConductivity();
    auto c_v = pm.specificHeatCapacity();

    // Time step size.
    auto dt = pm.timeStepSize();

    // Cell size
    auto dx = pm.mesh()->cellSize();

    // Local grid.
    const auto& local_grid = *(pm.mesh()->localGrid());

    // Get the particle data.
    auto x_p = pl.slice( Field::LogicalPosition() );
    auto m_p = pl.slice( Field::Mass() );
    auto v_p = pl.slice( Field::Volume() );
    auto e_p = pl.slice( Field::InternalEnergy() );

    // Get views of grid data.
    auto m_i = state.view( FieldLocation::Node(), Field::Mass() );
    auto e_i = state.view( FieldLocation::Node(), Field::InternalEnergy() );
    auto e_i_star = aux.view( FieldLocation::Node(),
                             UpdatedInternalEnergy() );
    auto phi_i = ls.getSignedDistance()->view();
    auto phi_hat_i = aux.view( FieldLocation::Node(), Field::SignedDistance() );

    // Reset write views.
    Kokkos::deep_copy( m_i, 0.0 );
    Kokkos::deep_copy( e_i, 0.0 );
    Kokkos::deep_copy( e_i_star, 0.0 );

    // Create the scatter views we need.
    auto m_i_sv = Kokkos::Experimental::create_scatter_view( m_i );
    auto e_i_sv = Kokkos::Experimental::create_scatter_view( e_i );
    auto e_i_star_sv = Kokkos::Experimental::create_scatter_view( e_i_star );

    // Build the local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Project particle mass and energy to the grid.
    Kokkos::parallel_for(
        "lpbf_p2g_mass_energy",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pl.size() ),
        KOKKOS_LAMBDA( const int p ){

            // Get the particle position.
            double x[3] = { x_p(p,0), x_p(p,1), x_p(p,2) };

            // Setup interpolation to the nodes.
            Cajita::SplineData<double,2,Cajita::Node> sd;
            Cajita::evaluateSpline( local_mesh, x, sd );

            // Project mass to the grid.
            Cajita::P2G::value( m_p(p), sd, m_i_sv );

            // Project energy to the grid.
            Cajita::P2G::value( m_p(p)*e_p(p), sd, e_i_sv );
        });

    // Complete local energy and mass scatter.
    Kokkos::Experimental::contribute( m_i, m_i_sv );
    Kokkos::Experimental::contribute( e_i, e_i_sv );

    // Complete the global energy and mass scatter.
    state.scatter( FieldLocation::Node(), Field::Mass() );
    state.scatter( FieldLocation::Node(), Field::InternalEnergy() );

    // Compute grid specific internal energy and regularize the signed
    // distance function.
    auto global_nodes =
        pm.mesh()->localGrid()->indexSpace(
            Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
    Kokkos::parallel_for(
        "lpbf_scale_grid_energy",
        Cajita::createExecutionPolicy(global_nodes,exec_space),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){

            // Only compute energy if a node has mass
            e_i(i,j,k,0) = ( m_i(i,j,k,0) > 0.0)
                           ? e_i(i,j,k,0) / m_i(i,j,k,0) : 0.0;

            // Regularize the signed distance.
            phi_hat_i(i,j,k,0) =
                0.5 - 1.0 / (1.0 + exp(phi_i(i,j,k,0)*2.0/dx) );
        });

    // Gather energy and signed distance.
    state.gather( FieldLocation::Node(), Field::InternalEnergy() );
    aux.gather( FieldLocation::Node(), Field::SignedDistance() );

    // Compute the energy increment.
    Kokkos::parallel_for(
        "lpbf_g2p2g_energy_increment",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pl.size() ),
        KOKKOS_LAMBDA( const int p ){

            // Get the particle position.
            double x[3] = { x_p(p,0), x_p(p,1), x_p(p,2) };

            // Setup interpolation to the nodes.
            Cajita::SplineData<double,2,Cajita::Node> sd;
            Cajita::evaluateSpline( local_mesh, x, sd );

            // Project energy gradient to the particle.
            double grad_e[3];
            Cajita::G2P::gradient( e_i, sd, grad_e );

            // Compute the particle flux.
            double scale = v_p(p) * kappa * c_v;
            for ( int d = 0; d < 3; ++d )
                grad_e[d] *= scale;

            // Project the heat flux divergence to the grid.
            Cajita::P2G::divergence( grad_e, sd, e_i_star_sv );

            // Project regularized signed distance to the particle.
            double phi_hat_p;
            Cajita::G2P::value( phi_hat_i, sd, phi_hat_p );

            // Evaluate the source term.
            double s_p = laser( x, phi_hat_p, time );

            // Project the source terms to the grid.
            Cajita::P2G::value( v_p(p)*s_p, sd, e_i_star_sv );
        });

    // Complete local energy increment scatter.
    Kokkos::Experimental::contribute( e_i_star, e_i_star_sv );

    // Complete the global energy increment scatter.
    aux.scatter( FieldLocation::Node(), UpdatedInternalEnergy() );

    // Compute updated grid energy from thermal fluxes and diffusion.
    Kokkos::parallel_for(
        "lpbf_compute_grid_energy_increment",
        Cajita::createExecutionPolicy(global_nodes,exec_space),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){

            // Only compute energy if a node has mass
            e_i_star(i,j,k,0) =
                ( m_i(i,j,k,0) > 0.0 )
                ? e_i(i,j,k,0) + dt * e_i_star(i,j,k,0) / m_i(i,j,k,0)
                : 0.0;
        });

    // Gather energy increment.
    aux.gather( FieldLocation::Node(), UpdatedInternalEnergy() );

    // Update particle energy with the thermal increment.
    Kokkos::parallel_for(
        "lpbf_update_particle_energy",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pl.size() ),
        KOKKOS_LAMBDA( const int p ){

            // Get the particle position.
            double x[3] = { x_p(p,0), x_p(p,1), x_p(p,2) };

            // Setup interpolation to the nodes.
            Cajita::SplineData<double,2,Cajita::Node> sd;
            Cajita::evaluateSpline( local_mesh, x, sd );

            // Interpolate energy increment to particle.
            double e;
            Cajita::G2P::value( e_i, sd, e );
            double e_star;
            Cajita::G2P::value( e_i_star, sd, e_star );
            e_p(p) += e_star - e;
        });
}

//---------------------------------------------------------------------------//

} // end namespace TimeIntegrator
} // end namespace LPBF
} // end namespace Picasso

#endif // end PICASSO_LPBF_TIMEINTEGRATOR_HPP
