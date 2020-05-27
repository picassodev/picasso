#ifndef PICASSO_FLIP_TIMEINTEGRATOR_HPP
#define PICASSO_FLIP_TIMEINTEGRATOR_HPP

#include <Picasso_FLIP_AuxiliaryFieldTypes.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

namespace Picasso
{
namespace FLIP
{
namespace TimeIntegrator
{
//---------------------------------------------------------------------------//
// Explicit time step.
template<class ExecutionSpace, class ProblemManagerType>
void step( const ExecutionSpace& exec_space,
           const ProblemManagerType& pm )
{
    // Grid fields.
    const auto& fields = pm.fields();

    // Particles.
    const auto& particles = pm.particleList();

    // Equation of state.
    auto eos = pm.eos();

    // Time step size.
    auto dt = pm.timeStepSize();

    // Time integration coeffcieint.
    auto theta = pm.theta();

    // Uniform cell size and volume.
    auto dx = pm.mesh()->cellSize();
    double rdx = 1.0 / dx;
    double cell_volume = dx * dx * dx;

    // Time integration coefficient.
    double volume_theta_dt = cell_volume * theta * delta_t;

    // Local grid.
    const auto& local_grid = *(pm.mesh()->localGrid());

    // Local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get slices of particle data.
    auto x_p = particles.slice( Field::LogicalPosition() );
    auto u_p = particles.slice( Field::Velocity() );
    auto m_p = particles.slice( Field::Mass() );
    auto e_p = particles.slice( Field::InternalEnergy() );

    // Get views of cell data.
    auto rho_c = fields->view( FieldLocation::Cell(), Field::Density() );
    auto p_c = fields->view( FieldLocation::Cell(), Field::Pressure() );
    auto e_c = fields->view( FieldLocation::Cell(), Field::InternalEnergy() );
    auto u_theta_div_c =
        fields->view( FieldLocation::Cell(), VelocityThetaDivergence() );
    auto comp_c = fields->view( FieldLocation::Cell(), CompressionTerm() );

    // Get views of node data.
    auto m_i = fields->view( FieldLocation::Node(), Field::Mass() );
    auto u_i = fields->view( FieldLocation::Node(), Field::Velocity() );
    auto u_old_i = fields->view( FieldLocation::Node(), VelocityOld() );
    auto u_theta_i = fields->view( FieldLocation::Node(), VelocityTheta() );
    auto acc_theta_i = fields->view( FieldLocation::Node(), AcclerationTheta() );
    auto acc_squared_i = fields->view( FieldLocation::Node(), AcclerationSquared() );

    // Reset write views.
    Kokkos::deep_copy( rho_c, 0.0 );
    Kokkos::deep_copy( e_c, 0.0 );
    Kokkos::deep_copy( u_theta_div_c, 0.0 );
    Kokkos::deep_copy( m_i, 0.0 );
    Kokkos::deep_copy( u_old_i, 0.0 );

    // Create scatter views.
    auto rho_c_sv = Kokkos::Experimental::create_scatter_view( rho_c );
    auto e_c_sv = Kokkos::Experimental::create_scatter_view( e_c );
    auto u_theta_div_c_sv = Kokkos::Experiuental::create_scatter_view( u_theta_div_c );
    auto m_i_sv = Kokkos::Experimental::create_scatter_view( m_i );
    auto u_old_i_sv = Kokkos::Experiuental::create_scatter_view( u_old_i );

    // P2G - Compute initial grid state.
    Kokkos::parallel_for(
        "flip_p2g",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, particles.size() ),
        KOKKOS_LAMBDA( const int p ){

            // Get the particle position.
            double x[3] = { x_p(p,0), x_p(p,1), x_p(p,2) };

            // Second order interpolation to nodes.
            Cajita::SplineData<double,2,Cajita::Node> sd_i;
            Cajita::evaluateSpline( local_mesh, x, sd_i );

            // Third order interpolation to cells.
            Cajita::SplineData<double,3,Cajita::Cell> sd_c;
            Cajita::evaluateSpline( local_mesh, x, sd_c );

            // Project mass to nodes.
            Cajita::P2G::value( m_p(p), sd_i, m_i_sv );

            // Project density to cells.
            Cajita::P2G::value( m_p(p), sd_c, rho_c_sv );

            // Project momentum to nodes.
            double momentum[3] = { m_p(p)*u_p(p,0),
                                   m_p(p)*u_p(p,1)
                                   m_p(p)*u_p(p,2) };
            Cajita::P2G::value( momentum, sd_i, u_old_i_sv );

            // Project internal energy to nodes.
            Cajita::P2G::value( e_p(p), sd_c, e_c_sv );
        } );

    // Complete local scatter.
    Kokkos::Experimental::contribute( m_i, m_i_sv );
    Kokkos::Experimental::contribute( u_old_i, u_old_i_sv );
    Kokkos::Experimental::contribute( rho_c, rho_c_sv );
    Kokkos::Experimental::contribute( e_c, e_c_sv );

    // Complete the global scatter.
    fields->scatter( FieldLocation::Node(), Field::Mass() );
    fields->scatter( FieldLocation::Node(), VelocityOld() );
    fields->scatter( FieldLocation::Cell(), Field::Density() );
    fields->scatter( FieldLocation::Cell(), Field::InternalEnergy() );

    // Finish cell density and specific internal energy computation and
    // compute the pressure.
    auto owned_cells =
        local_grid.index( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    Kokkos::parallel_for(
        "update_cell_density_and_energy",
        Cajita::createExecution(owned_cells,exec_space),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){

            // Only update if there is non-zero mass.
            if ( rho_c(i,j,k,0) > 0.0 )
            {
                // Update specific internal energy. Note that the density hasn't
                // been modified yet to avoid an extra multiplication by volume.
                e_c(i,j,k,0) /= rho_c(i,j,k,0);

                // Update density
                rho_c(i,j,k,0) /= cell_volume;

                // Compute pressure.
                p_c(i,j,k,0) =
                    eos( Field::Pressure(), e_c(i,j,k,0), rho_c(i,j,k,0) );
            }

            // Otherwise everything is zero.
            else
            {
                rho_c(i,j,k,0) = 0.0;
                e_c(i,j,k,0) = 0.0;
                p_c(i,j,k,0) = 0.0;
            }
        });

    // Gather specific internal energy, density, and pressure.
    fields->gather( FieldLocation::Cell(), Field::Density() );
    fields->gather( FieldLocation::Cell(), Field::InternalEnergy() );
    fields->gather( FieldLocation::Cell(), Field::Pressure() );

    // Compute node velocities and apply boundary conditions.
    auto owned_nodes =
        local_grid.index( Cajita::Own(), Cajita::Node(), Cajita::Local() );
    Kokkos::parallel_for(
        "compute_node_velocity",
        KOKKOS_LAMBDA( const int i, const int j, const int k ){

            // Only update this node if there is mass.
            if ( m_i(i,j,k,0) > 0.0 )
            {
                // Geometric coefficient.
                double dic[2][2][2][3];

                dic[0][0][0][0] = -0.25 * rdx;
                dic[1][0][0][0] =  0.25 * rdx;
                dic[1][1][0][0] =  0.25 * rdx;
                dic[0][1][0][0] = -0.25 * rdx;
                dic[0][0][1][0] = -0.25 * rdx;
                dic[1][0][1][0] =  0.25 * rdx;
                dic[1][1][1][0] =  0.25 * rdx;
                dic[0][1][1][0] = -0.25 * rdx;

                dic[0][0][0][1] = -0.25 * rdx;
                dic[1][0][0][1] = -0.25 * rdx;
                dic[1][1][0][1] =  0.25 * rdx;
                dic[0][1][0][1] =  0.25 * rdx;
                dic[0][0][1][1] = -0.25 * rdx;
                dic[1][0][1][1] = -0.25 * rdx;
                dic[1][1][1][1] =  0.25 * rdx;
                dic[0][1][1][1] =  0.25 * rdx;

                dic[0][0][0][2] = -0.25 * rdx;
                dic[1][0][0][2] = -0.25 * rdx;
                dic[1][1][0][2] = -0.25 * rdx;
                dic[0][1][0][2] = -0.25 * rdx;
                dic[0][0][1][2] =  0.25 * rdx;
                dic[1][0][1][2] =  0.25 * rdx;
                dic[1][1][1][2] =  0.25 * rdx;
                dic[0][1][1][2] =  0.25 * rdx;

                // Compute pressure gradient.
                double grad_p[3] = {0.0,0.0,0.0};
                for ( ic = 0; ic < 2; ++ic )
                    for ( jc = 0; jc < 2; ++jc )
                        for ( kc = 0; kc < 2; ++kc )
                            for ( int d = 0; d < 3; ++d )
                            {
                                grad_p[d] +=
                                    dic[ic][jc][kc][d] *
                                    p_c(i+ic-1,j+jc-1,k+kc-1,0);
                            }

                // Compute coefficient.
                double coeff = volume_theta_dt / m_i(i,j,k,0);

                // Compute theta velocity.
                for ( int d = 0; d < 3; ++d )
                    u_theta_i(i,j,k,d) =
                        u_old_i(i,j,k,d) + grad_p[d] * coeff;

                // Apply boundary conditions to theta velocity.
                // FIXME: make sure this happens before the updated velocity
                // is computed.

                // Project divergence of theta velocity to to cell centers.
                auto u_theta_div_c_sv_a = u_theta_div_c_sv.access();
                double u_div_c;
                for ( ic = 0; ic < 2; ++ic )
                    for ( jc = 0; jc < 2; ++jc )
                        for ( kc = 0; kc < 2; ++kc )
                        {
                            u_div_c = 0.0;
                            for ( int d = 0; d < 3; ++d )
                            {
                                u_div_c += dic[ic][jc][kc][d] *
                                           u_theta_i(i,j,k,d);
                            }
                            u_theta_div_c_sv_a(i+ic-1,j+jc-1,k_kc-1) += u_div_c;
                        }

                // Compute updated velocity.
                for ( int d = 0; d < 3; ++d )
                    u_i(i,j,k,d) = (u_theta_i(i,j,k,d) -
                                    (1.0 - theta)*u_old_i(i,j,k,d)) / theta;
            }

            // Otherwise zero velocity.
            else
            {
                for ( int d = 0; d < 3; ++d )
                {
                    u_i(i,j,k,d) = 0.0;
                    u_old_i(i,j,k,d) = 0.0;
                    u_theta_i(i,j,k,d) = 0.0;
                }
            }
        });

    // Compute the local divergence scatter.
    Kokkos::Experimental::contribute( u_theta_div_c, u_theta_div_c_sv );

    // Compute the global diveregence scatter.
    fields->scatter( FieldLocation::Cell(), VelocityThetaDivergence() );

    // Gather velocities.
    fields->gather( FieldLocation::Node(), Field::Velocity() );
    fields->gather( FieldLocation::Node(), VelocityOld() );
    fields->gather( FieldLocation::Node(), VelocityTheta() );
    fields->gather( FieldLocation::Cell(), VelocityThetaDivergence() );

    // Evaluate the compression term and theta acceleration term.
    // FIXME: this shouldn't be needed when cajita can do G2P with a general
    // grid access functor.
    auto ghosted_cells =
        local_grid.index( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );
    Kokkos::parallel_for(
        "stuff_we_will_fix_cells",
        Cajita::createExecution(ghosted_cells,exec_space),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            comp_c(i,j,k,0) = p_c(i,j,k,0) * u_theta_div_c(i,j,k,0) /
                              ( rho_c(i,j,k,0) * e_c(i,j,k,0) );
        });
    auto ghosted_nodes =
        local_grid.index( Cajita::Ghost(), Cajita::Node(), Cajita::Local() );
    Kokkos::parallel_for(
        "stuff_we_will_fix_nodes",
        Cajita::createExecution(ghosted_nodes,exec_space),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            double value = 0.0;
            for ( int d = 0; d < 3; ++d )
                value += (u_i(i,j,k,d) - u_old_i(i,j,k,d)) *
                         (u_i(i,j,k,d) - u_old_i(i,j,k,d));
            acc_theta_i(i,j,k,0) = value;

            double u_0_sqr = 0.0;
            double u_1_sqr = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                u_0_sqr += u_old_i(i,j,k,d) * u_old_i(i,j,k,d);
                u_1_sqr += u_i(i,j,k,d) * u_i(i,j,k,d);
            }
            acc_squared_i(i,j,k,0) = u_1_sqr - u_0_sqr;
        });

    // G2P - update particles
    Kokkos::parallel_for(
        "flip_g2p",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, particles.size() ),
        KOKKOS_LAMBDA( const int p ){

            // Second order interpolation to nodes.
            Cajita::SplineData<double,2,Cajita::Node> sd_i;
            Cajita::evaluateSpline( local_mesh, x, sd_i );

            // Third order interpolation to cells.
            Cajita::SplineData<double,3,Cajita::Cell> sd_c;
            Cajita::evaluateSpline( local_mesh, x, sd_c );

            // Store old particle velocity.
            double u_p_0[3] = { u_p(p,0), u_p(p,1), u_p(p,2) };

            // Project velocities to particle
            double u_p_old[3];
            Cajita::G2P( u_old_i, sd_i, u_p_old );
            double u_p_theta[3];
            Cajita::G2P( u_theta_i, sd_i, u_p_theta );
            double u_p_new[3];
            Cajita::G2P( u_i, sd_i, u_p_new );

            // Compute new particle velocity and position.
            // FIXME: decide how this position update changes if the grid is
            // adaptive.
            double u_p_1[3];
            for ( int d = 0; d < 3; ++d )
            {
                u_p_1[d] = u_p_new[d] - u_p_old[d];
                u_p(p,d) = u_p_1[d];
                x_p(p,d) += u_p_theta[d] * delta_t;
            }

            // Update particle internal energy.
            double comp_p;
            Cajita::G2P::value( comp_c, sd_c, comp_p );

            double acc_theta_p;
            Cajita::G2P::value( acc_theta_i, sd_i, acc_theta_p );

            double u_p_0_sqr = 0.0;
            double u_p_1_sqr = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                u_p_0_sqr += u_p_0[d] * u_p_0[d];
                u_p_1_sqr += u_p_1[d] * u_p_1[d];
            }

            e_p(p) = e_p(p) * ( 1.0 - comp_p * delta_t ) +
                     m_p(p) * acc_theta_p * ( theta - 0.5 ) +
                     m_p(p) * ( acc_squared_p - (u_p_1_sqr - u_p_0_sqr) ) * 0.5;
        });
}

//---------------------------------------------------------------------------//

} // end namespace TimeIntegrator
} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_TIMEINTEGRATOR_HPP
