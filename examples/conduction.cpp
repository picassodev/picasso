#include <Harlow_DenseLinearAlgebra.hpp>
#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_ParticleInit.hpp>
#include <Harlow_SiloParticleWriter.hpp>
#include <Harlow_VelocityInterpolation.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

#include <cstdlib>
#include <array>
#include <cmath>

//---------------------------------------------------------------------------//
using Cajita::Dim;

//---------------------------------------------------------------------------//
struct Phase
{
    enum Values
    {
        solid = 0,
        liquid = 1
    };
};

//---------------------------------------------------------------------------//
// Pure material with a linear approximation and the critical temperature
// equal to the formation temperature.
template<class Scalar>
struct Material
{
    // Thermal conductivity.
    Kokkos::Array<Scalar,2> k;

    // Specific heat at constant volume.
    Kokkos::Array<Scalar,2> c_v;

    // Density.
    Kokkos::Array<Scalar,2> rho;

    // Critical temperature. This is a pure material model so no mushy
    // zone. We would change this in the module for handling phase change.
    Scalar tc;

    // Heat transfer coefficient between phases.
    Scalar hc;

    // Compute the temperature from the internal energy.
    KOKKOS_INLINE_FUNCTION
    Scalar temperature( const Scalar e, const int phase ) const
    {
        return e / c_v[phase];
    }

    // Compute the internal energy from the temperature.
    KOKKOS_INLINE_FUNCTION
    Scalar energy( const Scalar t, const int phase ) const
    {
        return c_v[phase] * t;
    }


    // Evaluate heat transfer between phases.
    KOKKOS_INLINE_FUNCTION
    void phaseHeatTransfer(
        const Scalar& mass_s, const Scalar& mass_l,
        const Scalar& temperature_s, const Scalar& temperature_l,
        Scalar& energy_s, Scalar& energy_l ) const
    {
        // Compute phase volumes.
        Scalar vs = rho[Phase::solid] / mass_s;
        Scalar vl = rho[Phase::liquid] / mass_l;
        Scalar v = vs + vl;

        // Update energy.
        energy_s += (vs/v) * (vl/v) * hc * (temperature_l - temperature_s);
        energy_l += (vs/v) * (vl/v) * hc * (temperature_s - temperature_l);
    }

    // Evaluate phase change at a point and compute temperatures.
    KOKKOS_INLINE_FUNCTION
    void phaseChange( Scalar& mass_s, Scalar& mass_l,
                      Scalar& energy_s, Scalar& energy_l,
                      Scalar& temperature_s, Scalar& temperature_l ) const
    {
        // Compute total mass.
        Scalar m_total = mass_l + mass_s;

        // If no mass we have nothing to do.
        if ( 0.0 == m_total )
            return;

        // Compute average energy.
        Scalar e_ave = ( mass_s * energy_s + mass_l * energy_l ) / m_total;

        // Compute energies at critical phase temperatures.
        Scalar ec_s = energy( tc, Phase::solid );
        Scalar ec_l = energy( tc, Phase::liquid );

        // Pure solid case.
        if ( e_ave <= ec_s )
        {
            // Update mass.
            mass_s = m_total;
            mass_l = 0.0;

            // Update energy.
            energy_s = e_ave;
            energy_l = 0.0;

            // Update temperature.
            temperature_s = temperature( energy_s, Phase::solid );
            temperature_l = 0.0;
        }

        // Solid and liquid mixture case.
        else if ( ec_s < e_ave && e_ave < ec_l )
        {
            // Update mass.
            mass_s = m_total * (ec_l - e_ave) / (ec_l - ec_s);
            mass_l = m_total * (e_ave - ec_s) / (ec_l - ec_s);

            // Update energy.
            energy_s = ec_s;
            energy_l = ec_l;

            // Update temperature.
            temperature_s = tc;
            temperature_l = tc;
        }

        // Pure liquid case.
        else if ( ec_l <= e_ave )
        {
            // Update mass.
            mass_s = 0.0;
            mass_l = m_total;

            // Update energy.
            energy_s = 0.0;
            energy_l = e_ave;

            // Update temperature.
            temperature_s = 0.0;
            temperature_l = temperature( energy_l, Phase::liquid );
        }
    }
};

//---------------------------------------------------------------------------//
struct Boundary
{
    enum Values
    {
        x_low  = 0,
        x_high = 1,
        y_low  = 2,
        y_high = 3,
        z_low  = 4,
        z_high = 5,
        free   = 6
    };
};

//---------------------------------------------------------------------------//
struct SolidParticleField
{
    enum Values
    {
        x = 0, // position
        m = 1, // mass
        e = 2, // internal energy
        t = 3  // temperature
    };
};

//---------------------------------------------------------------------------//
struct LiquidParticleField
{
    enum Values
    {
        x = 0, // position
        m = 1, // mass
        e = 2, // internal energy
        t = 3  // temperature
    };
};

//---------------------------------------------------------------------------//
void solve( const int num_cell,
            const int ppc,
            const double t_final,
            const double delta_t,
            const int write_freq )
{
    // Material properties (approximately titanium)
    Material<double> mat;

    // // Multi-phase change properties.
    // mat.tc = 1650.0;
    // mat.hc = 10000.0;

    // // Solid properties.
    // mat.k[Phase::solid] = 20.0;
    // mat.c_v[Phase::solid] = 0.5;
    // mat.rho[Phase::solid] = 4500.0;

    // // Liquid properties.
    // mat.k[Phase::liquid] = 10.0;
    // mat.c_v[Phase::liquid] = 0.79;
    // mat.rho[Phase::liquid] = 4100.0;

    // Multi-phase change properties.
    mat.tc = 0.0;
    mat.hc = 10000.0;

    // Solid properties.
    mat.k[Phase::solid] = 0.002;
    mat.c_v[Phase::solid] = 0.5;
    mat.rho[Phase::solid] = 1.0;

    // Liquid properties.
    mat.k[Phase::liquid] = 0.0005;
    mat.c_v[Phase::liquid] = 4.0;
    mat.rho[Phase::liquid] = 1.0;

    // Initial temperature at critical point.
    double init_temp = mat.tc;

    // Types.
    using execution_space = Kokkos::Serial;
    using memory_space = Kokkos::HostSpace;
    using device_type = Kokkos::Device<execution_space,memory_space>;
    using Cajita::Dim;

    // Spline discretization.
    const int spline_order = 1;
    using sd_type = Cajita::SplineData<double,spline_order,Cajita::Node>;

    // Build the global mesh.
    const double cell_size = 2.0 / num_cell;
    const std::array<double,3> global_low_corner = { 0.0, -0.1, -0.1 };
    const std::array<double,3> global_high_corner = { 1.0, 0.1, 0.1 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Global grid.
    const std::array<bool,3> periodic = {false, true, true};
    auto global_grid = Cajita::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, Cajita::UniformDimPartitioner() );

    // Local grid.
    const int halo_width = 1;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Index spaces.
    auto node_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Node(), Cajita::Local() );

    // Local mesh.
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    // Grid field layouts. One degree-of-freedom per phase
    auto node_layout = Cajita::createArrayLayout( local_grid, 2, Cajita::Node() );
    auto node_halo =
        Cajita::createHalo<double,device_type>( *node_layout, Cajita::FullHaloPattern() );

    // Mass for each phase
    auto mass_new =
        Cajita::createArray<double,device_type>( "mass_new", node_layout );
    auto mass_new_s = Cajita::createSubarray( *mass_new, 0, 1 );
    auto mass_new_l = Cajita::createSubarray( *mass_new, 1, 2 );

    auto mass_old =
        Cajita::createArray<double,device_type>( "mass_old", node_layout );
    auto mass_old_s = Cajita::createSubarray( *mass_old, 0, 1 );
    auto mass_old_l = Cajita::createSubarray( *mass_old, 1, 2 );

    auto mass_update =
        Cajita::createArray<double,device_type>( "mass_update", node_layout );
    auto mass_update_s = Cajita::createSubarray( *mass_update, 0, 1 );
    auto mass_update_l = Cajita::createSubarray( *mass_update, 1, 2 );

    // Energy for each phase
    auto energy_new =
        Cajita::createArray<double,device_type>( "energy_new", node_layout );
    auto energy_new_s = Cajita::createSubarray( *energy_new, 0, 1 );
    auto energy_new_l = Cajita::createSubarray( *energy_new, 1, 2 );

    auto energy_old =
        Cajita::createArray<double,device_type>( "energy_old", node_layout );
    auto energy_old_s = Cajita::createSubarray( *energy_old, 0, 1 );
    auto energy_old_l = Cajita::createSubarray( *energy_old, 1, 2 );

    auto energy_update =
        Cajita::createArray<double,device_type>( "energy_update", node_layout );
    auto energy_update_s = Cajita::createSubarray( *energy_update, 0, 1 );
    auto energy_update_l = Cajita::createSubarray( *energy_update, 1, 2 );

    // Temperature for each phase
    auto temperature =
        Cajita::createArray<double,device_type>( "temperature", node_layout );
    auto temperature_s = Cajita::createSubarray( *temperature, 0, 1 );
    auto temperature_l = Cajita::createSubarray( *temperature, 1, 2 );

    // Combined views.
    auto mass_old_view = mass_old->view();
    auto mass_new_view = mass_new->view();
    auto mass_update_view = mass_update->view();
    auto energy_old_view = energy_old->view();
    auto energy_new_view = energy_new->view();
    auto energy_update_view = energy_update->view();
    auto temperature_view = temperature->view();

    // Solid views
    auto mass_old_view_s = mass_old_s->view();
    auto mass_new_view_s = mass_new_s->view();
    auto mass_update_view_s = mass_update_s->view();
    auto energy_old_view_s = energy_old_s->view();
    auto energy_new_view_s = energy_new_s->view();
    auto energy_update_view_s = energy_update_s->view();
    auto temperature_view_s = temperature_s->view();

    // Liquid views
    auto mass_old_view_l = mass_old_l->view();
    auto mass_new_view_l = mass_new_l->view();
    auto mass_update_view_l = mass_update_l->view();
    auto energy_old_view_l = energy_old_l->view();
    auto energy_new_view_l = energy_new_l->view();
    auto energy_update_view_l = energy_update_l->view();
    auto temperature_view_l = temperature_l->view();

    // Initialize fields.
    Cajita::ArrayOp::assign( *mass_old, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *mass_new, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *mass_update, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_old, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_new, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_update, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *temperature, 0.0, Cajita::Ghost() );

    // Solid particles.
    using SolidParticleTypes =
        Cabana::MemberTypes<double[3], // position
                            double,    // mass
                            double,    // internal energy
                            double>;   // temperature
    using SolidParticleList = Cabana::AoSoA<SolidParticleTypes,device_type>;
    using SolidParticle = typename SolidParticleList::tuple_type;

    // Create solid particles. All particles start as solid.
    double pvolume = cell_size * cell_size * cell_size / (ppc*ppc*ppc);
    SolidParticleList particles_s( "solid_particles" );
    auto create_func_s = KOKKOS_LAMBDA( double x[3], SolidParticle& p ) {
        for ( int d = 0; d < 3; ++d )
        {
            Cabana::get<SolidParticleField::x>(p,d) = x[d];
        }
        Cabana::get<SolidParticleField::m>(p) = pvolume * mat.rho[Phase::solid];
        Cabana::get<SolidParticleField::e>(p) = mat.energy( init_temp, Phase::solid );
        Cabana::get<SolidParticleField::t>(p) = init_temp;
        return true;
    };
    Harlow::initializeParticles(
        Harlow::InitUniform(), execution_space(), *local_grid, ppc, create_func_s, particles_s );

    // Get slices for each solid particle field.
    auto x_p_s = Cabana::slice<SolidParticleField::x>( particles_s, "position" );
    auto m_p_s = Cabana::slice<SolidParticleField::m>( particles_s, "mass" );
    auto e_p_s = Cabana::slice<SolidParticleField::e>( particles_s, "energy" );
    auto t_p_s = Cabana::slice<SolidParticleField::t>( particles_s, "temperature" );

    // Initialize solid grid temperature with particle temperature.
    // For now just scale the temperature by the number of particles
    // in a cell - the future we need to do a mass-averaged
    // deposition to avoid a scaling by the number of particles per
    // cell.
    Cajita::p2g( Cajita::createScalarValueP2G(t_p_s,1.0/(ppc*ppc*ppc)),
                 x_p_s, x_p_s.size(), Cajita::Spline<1>(),
                 *node_halo, *temperature_s );

    // Initialize solid grid mass with particle mass.
    Cajita::p2g( Cajita::createScalarValueP2G(m_p_s,1.0),
                 x_p_s, x_p_s.size(), Cajita::Spline<1>(),
                 *node_halo, *mass_new_s );
    Cajita::ArrayOp::copy( *mass_old_s, *mass_new_s, Cajita::Ghost() );

    // Compute grid energy from temperature
    Kokkos::parallel_for(
        "grid_init",
        Cajita::createExecutionPolicy(node_space,execution_space()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            energy_new_view(i,j,k,Phase::solid) =
                mat.energy( temperature_view(i,j,k,Phase::solid), Phase::solid );
            energy_old_view(i,j,k,Phase::solid) =
                energy_new_view(i,j,k,Phase::solid);
        });

    // Liquid particles.
    using LiquidParticleTypes =
        Cabana::MemberTypes<double[3], // position
                            double,    // mass
                            double,    // internal energy
                            double>;   // temperature
    using LiquidParticleList = Cabana::AoSoA<LiquidParticleTypes,device_type>;
    using LiquidParticle = typename LiquidParticleList::tuple_type;

    // Create liquid particles. All particles start as solid.
    LiquidParticleList particles_l( "liquid_particles" );
    auto create_func_l = KOKKOS_LAMBDA( double x[3], LiquidParticle& p ) {
        for ( int d = 0; d < 3; ++d )
        {
            Cabana::get<LiquidParticleField::x>(p,d) = x[d];
        }
        Cabana::get<LiquidParticleField::m>(p) = 0.0;
        Cabana::get<LiquidParticleField::e>(p) = 0.0;
        Cabana::get<LiquidParticleField::t>(p) = 0.0;
        return true;
    };
    Harlow::initializeParticles(
        Harlow::InitUniform(), execution_space(), *local_grid, ppc, create_func_l, particles_l );

    // Get slices for each liquid particle field.
    auto x_p_l = Cabana::slice<LiquidParticleField::x>( particles_l, "position" );
    auto m_p_l = Cabana::slice<LiquidParticleField::m>( particles_l, "mass" );
    auto e_p_l = Cabana::slice<LiquidParticleField::e>( particles_l, "energy" );
    auto t_p_l = Cabana::slice<LiquidParticleField::t>( particles_l, "temperature" );

    // Empty source.
    auto eval_source = KOKKOS_LAMBDA( const double[3], const double ) {
        return 0.0;
    };

    // Time step.
    int num_step = t_final / delta_t;
    double time = 0.0;
    for ( int t = 0; t < num_step; ++t )
    {
        // Print time step info.
        if ( 0 == global_grid->blockId() &&
             0 == t % write_freq )
        {
            std::cout << "Step " << t+1 << "/" << num_step
                      << " - time " << time << std::endl;
        }

        // Reinitialize solid slices after communications.
        x_p_s = Cabana::slice<SolidParticleField::x>( particles_s, "position" );
        m_p_s = Cabana::slice<SolidParticleField::m>( particles_s, "mass" );
        e_p_s = Cabana::slice<SolidParticleField::e>( particles_s, "energy" );
        t_p_s = Cabana::slice<SolidParticleField::t>( particles_s, "temperature" );

        // Reinitialize liquid slices after communications.
        x_p_l = Cabana::slice<LiquidParticleField::x>( particles_l, "position" );
        m_p_l = Cabana::slice<LiquidParticleField::m>( particles_l, "mass" );
        e_p_l = Cabana::slice<LiquidParticleField::e>( particles_l, "energy" );
        t_p_l = Cabana::slice<LiquidParticleField::t>( particles_l, "temperature" );

        // Output particles
        if ( 0 == t % write_freq )
        {
            Cajita::BovWriter::writeTimeStep( t, time, *mass_new_s );
            Cajita::BovWriter::writeTimeStep( t, time, *energy_new_s );
            Cajita::BovWriter::writeTimeStep( t, time, *temperature_s );
            // Harlow::SiloParticleWriter::writeTimeStep(
            //     *global_grid, t, time, x_p_s, m_p_s, e_p_s, t_p_s );
            // Harlow::SiloParticleWriter::writeTimeStep(
            //     "liquid", *global_grid, t, time, x_p_l, m_p_l, e_p_l, t_p_l );
        }

        // Gather grid values.
        node_halo->gather( *mass_old );
        node_halo->gather( *mass_new );
        node_halo->gather( *energy_old );
        node_halo->gather( *energy_new );
        node_halo->gather( *temperature );

        // Clear the scatter arrays.
        Cajita::ArrayOp::assign( *energy_update, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *mass_update, 0.0, Cajita::Ghost() );

        // Mass and energy update scatter views for solid phase.
        auto energy_update_sv_s =
            Kokkos::Experimental::create_scatter_view( energy_update_view_s );
        auto mass_update_sv_s =
            Kokkos::Experimental::create_scatter_view( mass_update_view_s );

        // Mass and update scatter views for liquid phase.
        auto energy_update_sv_l =
            Kokkos::Experimental::create_scatter_view( energy_update_view_l );
        auto mass_update_sv_l =
            Kokkos::Experimental::create_scatter_view( mass_update_view_l );

        // Solid particle loop.
        Kokkos::parallel_for(
            "particle_step_solid",
            Kokkos::RangePolicy<execution_space>(0,particles_s.size()),
            KOKKOS_LAMBDA( const int p ){

                // Get the particle position.
                double px[3] = { x_p_s(p,Dim::I), x_p_s(p,Dim::J), x_p_s(p,Dim::K) };

                // Evaluate the spline.
                sd_type sd;
                Cajita::evaluateSpline( local_mesh, px, sd );

                // Update particle mass with the grid mass increment.
                double m_old, m_new;
                Cajita::G2P::value( mass_old_view_s, sd, m_old );
                Cajita::G2P::value( mass_new_view_s, sd, m_new );
                m_p_s(p) += m_new - m_old;

                // Compute the new particle volume.
                double volume = m_p_s(p) / mat.rho[Phase::solid];

                // Update particle energy with the grid energy increment.
                double e_old, e_new;
                Cajita::G2P::value( energy_old_view_s, sd, e_old );
                Cajita::G2P::value( energy_new_view_s, sd, e_new );
                e_p_s(p) += e_new - e_old;

                // Update particle temperature.
                t_p_s(p) = mat.temperature( e_p_s(p), Phase::solid );

                // Compute temperature gradient at the particle.
                double grad_t[3];
                Cajita::G2P::gradient( temperature_view_s, sd, grad_t );

                // Compute the volume-scaled heat flux.
                for ( int d = 0; d < 3; ++d )
                    grad_t[d] *= volume * mat.k[Phase::solid];

                // Project flux divergence back to the grid.
                Cajita::P2G::divergence( grad_t, sd, energy_update_sv_s );

                // Add source term.
                Cajita::P2G::value( eval_source(px,time) * volume, sd, energy_update_sv_s );

                // Project mass to grid.
                Cajita::P2G::value( m_p_s(p), sd, mass_update_sv_s );
            });

        // Complete the particle-grid scatter to the nodes.
        Kokkos::Experimental::contribute( mass_update_view_s, mass_update_sv_s );
        Kokkos::Experimental::contribute( energy_update_view_s, energy_update_sv_s );

        // Liquid particle loop
        Kokkos::parallel_for(
            "particle_step_liquid",
            Kokkos::RangePolicy<execution_space>(0,particles_l.size()),
            KOKKOS_LAMBDA( const int p ){

                // Get the particle position.
                double px[3] = { x_p_l(p,Dim::I), x_p_l(p,Dim::J), x_p_l(p,Dim::K) };

                // Evaluate the spline.
                sd_type sd;
                Cajita::evaluateSpline( local_mesh, px, sd );

                // Update particle mass with the grid mass increment.
                double m_old, m_new;
                Cajita::G2P::value( mass_old_view_l, sd, m_old );
                Cajita::G2P::value( mass_new_view_l, sd, m_new );
                m_p_l(p) += m_new - m_old;

                // Compute the new particle volume.
                double volume = m_p_l(p) / mat.rho[Phase::liquid];

                // Update particle energy with the grid energy increment.
                double e_old, e_new;
                Cajita::G2P::value( energy_old_view_l, sd, e_old );
                Cajita::G2P::value( energy_new_view_l, sd, e_new );
                e_p_l(p) += e_new - e_old;

                // Update particle temperature.
                t_p_l(p) = mat.temperature( e_p_l(p), Phase::liquid );

                // Compute temperature gradient at the particle.
                double grad_t[3];
                Cajita::G2P::gradient( temperature_view_l, sd, grad_t );

                // Compute the volume-scaled heat flux.
                for ( int d = 0; d < 3; ++d )
                    grad_t[d] *= volume * mat.k[Phase::liquid];

                // Project flux divergence back to the grid.
                Cajita::P2G::divergence( grad_t, sd, energy_update_sv_l );

                // Add source term.
                Cajita::P2G::value( eval_source(px,time) * volume, sd, energy_update_sv_l );

                // Project mass to grid.
                Cajita::P2G::value( m_p_l(p), sd, mass_update_sv_l );
            });

        // Complete the particle-grid scatter to the nodes.
        Kokkos::Experimental::contribute( mass_update_view_l, mass_update_sv_l );
        Kokkos::Experimental::contribute( energy_update_view_l, energy_update_sv_l );

        // Complete parallel scatter.
        node_halo->scatter( *mass_update );
        node_halo->scatter( *energy_update );

        // Grid update
        Kokkos::parallel_for(
            "grid_update",
            Cajita::createExecutionPolicy(node_space,execution_space()),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Update mass.
                for ( int ph = 0; ph < 2; ++ph )
                {
                    mass_old_view(i,j,k,ph) = mass_new_view(i,j,k,ph);
                    mass_new_view(i,j,k,ph) = mass_update_view(i,j,k,ph);
                }

                // Update energy.
                for ( int ph = 0; ph < 2; ++ph )
                {
                    energy_old_view(i,j,k,ph) = energy_new_view(i,j,k,ph);

                    if ( mass_new_view(i,j,k,ph) > 0.0 )
                    {
                        energy_new_view(i,j,k,ph) +=
                            energy_update_view(i,j,k,ph) * delta_t / mass_new_view(i,j,k,ph);
                    }
                    else
                    {
                        energy_new_view(i,j,k,ph) = 0.0;
                    }
                }

                // Evaluate new masses, energies, and temperatures based on
                // phase change.
                // mat.phaseChange( mass_new_view(i,j,k,Phase::solid),
                //                  mass_new_view(i,j,k,Phase::liquid),
                //                  energy_new_view(i,j,k,Phase::solid),
                //                  energy_new_view(i,j,k,Phase::liquid),
                //                  temperature_view(i,j,k,Phase::solid),
                //                  temperature_view(i,j,k,Phase::liquid) );
            });

        // Boundary conditions. Set the left node boundary temp to be 10
        // degrees above melting, set the right node boundary to be 10 degrees
        // below melting.
        int global_node_min = 0;
        int global_node_max =
            global_grid->globalNumEntity( Cajita::Node(), Dim::I );
        auto global_space = local_grid->indexSpace(
            Cajita::Own(),Cajita::Node(),Cajita::Global());
        auto i_off = global_grid->globalOffset( Dim::I );
        auto j_off = global_grid->globalOffset( Dim::J );
        auto k_off = global_grid->globalOffset( Dim::K );
        Kokkos::parallel_for(
            "boundary_condition",
            Cajita::createExecutionPolicy(global_space,execution_space()),
            KOKKOS_LAMBDA( const int i, const int j, const int k){
                if ( global_node_min == i )
                    temperature_view(i-i_off,j-j_off,k-k_off,Phase::solid) =
                        mat.tc + 10.0;
                else if ( global_node_max - 1 == i )
                    temperature_view(i-i_off,j-j_off,k-k_off,Phase::solid) =
                        mat.tc - 10.0;
                else
                    temperature_view(i-i_off,j-j_off,k-k_off,Phase::solid) =
                        mat.temperature(
                            energy_new_view(i-i_off,j-j_off,k-k_off,Phase::solid),
                            Phase::solid);
            });

        // Redistribute particles.
        Harlow::ParticleCommunication::redistribute(
            *local_grid, particles_s,
            std::integral_constant<std::size_t,SolidParticleField::x>() );
        Harlow::ParticleCommunication::redistribute(
            *local_grid, particles_l,
            std::integral_constant<std::size_t,LiquidParticleField::x>() );

        // Update time
        time += delta_t;
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::initialize( argc, argv );

    // number of cells
    int num_cell = std::atoi( argv[1] );

    // material points per cell
    int ppc = std::atoi( argv[2] );

    // end time.
    double t_final = std::atof( argv[3] );

    // time step size
    double delta_t = std::atof( argv[4] );

    // write frequency
    int write_freq = std::atof( argv[5] );

    // run the problem.
    solve( num_cell, ppc, t_final, delta_t, write_freq );

    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
