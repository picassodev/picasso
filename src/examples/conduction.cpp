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
struct Phase
{
    enum Values
    {
        solid = 0,
        fluid = 1
    };
};

//---------------------------------------------------------------------------//
struct ParticleField
{
    enum Values
    {
        x = 0, // position
        m = 1, // mass
        v = 2, // volume
        e = 3, // internal energy
        t = 4  // temperature
    };
};

struct Material
{
    // Thermal conductivity.
    double k;

    // Specific heat at constant volume.
    double c_v;

    // Density
    double rho;

    // Critical temperature
    double t_c;

    // Compute the temperature from the internal energy.
    double temperature( const double e ) const
    {
        return e / c_v;
    }

    // Compute the internal energy from the temperature.
    double energy( const double t ) const
    {
        return c_v * t;
    }
};

//---------------------------------------------------------------------------//
void solve( const int num_cell,
            const int ppc,
            const double t_final,
            const double delta_t,
            const int write_freq )
{
    // Material 0 (approximately solid titanium)
    Material mat_0;
    mat_0.k = 20.0;
    mat_0.c_v = 0.5;
    mat_0.rho = 4500.0;
    mat_0.t_c = 1650.0;

    // Initial temperature.
    double init_temp = 50.0;

    // Thermal source. Spherical source from -r to r centered at the origin.
    double source_radius = 0.25;
    double source_strength = 1.0e8;
    auto eval_source = KOKKOS_LAMBDA( const double x[3] ) {
        double r2 =
        x[Dim::I] * x[Dim::I] +
        x[Dim::J] * x[Dim::J] +
        x[Dim::K] * x[Dim::K];
        return ( r2 < source_radius * source_radius ) ? source_strength : 0.0;
    };

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
    const std::array<double,3> global_low_corner = { -1.0, -1.0, -1.0 };
    const std::array<double,3> global_high_corner = { 1.0, 1.0, 1.0 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Global grid.
    const std::array<bool,3> periodic = {true, true, true};
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

    // Grid fields.
    auto node_layout = Cajita::createArrayLayout( local_grid, 1, Cajita::Node() );
    auto node_halo =
        Cajita::createHalo<double,device_type>( *node_layout, Cajita::FullHaloPattern() );
    auto mass =
        Cajita::createArray<double,device_type>( "mass", node_layout );
    auto energy_update =
        Cajita::createArray<double,device_type>( "energy_update", node_layout );
    auto energy_old =
        Cajita::createArray<double,device_type>( "energy_old", node_layout );
    auto energy_new =
        Cajita::createArray<double,device_type>( "energy_new", node_layout );

    // Initialize fields.
    Cajita::ArrayOp::assign( *mass, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_update, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_old, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *energy_new, 0.0, Cajita::Ghost() );

    // Particle data types.
    using ParticleTypes =
        Cabana::MemberTypes<double[3], // position
                            double,    // mass
                            double,    // volume
                            double,    // internal energy
                            double>;   // temperature
    using ParticleList = Cabana::AoSoA<ParticleTypes,device_type>;
    using Particle = typename ParticleList::tuple_type;

    // Create particles.
    double pvolume = cell_size * cell_size * cell_size / ppc;
    ParticleList particles( "particles" );
    auto create_func = KOKKOS_LAMBDA( double x[3], Particle& p ) {
        for ( int d = 0; d < 3; ++d )
        {
            Cabana::get<ParticleField::x>(p,d) = x[d];
        }
        Cabana::get<ParticleField::m>(p) = pvolume * mat_0.rho;
        Cabana::get<ParticleField::v>(p) = pvolume;
        Cabana::get<ParticleField::e>(p) = mat_0.energy( init_temp );
        Cabana::get<ParticleField::t>(p) = init_temp;
        return true;
    };
    Harlow::initializeParticles( *local_grid, ppc, create_func, particles );

    // Get slices for each phase.
    auto x_p = Cabana::slice<ParticleField::x>( particles, "position" );
    auto m_p = Cabana::slice<ParticleField::m>( particles, "mass" );
    auto v_p = Cabana::slice<ParticleField::v>( particles, "volume" );
    auto e_p = Cabana::slice<ParticleField::e>( particles, "energy" );
    auto t_p = Cabana::slice<ParticleField::t>( particles, "temperature" );

    // Initialize grid energy with particle energy.
    Cajita::p2g( Cajita::createScalarValueP2G(e_p,1.0),
                 x_p, x_p.size(), Cajita::Spline<1>(),
                 *node_halo, *energy_old );
    Cajita::ArrayOp::copy( *energy_new, *energy_old, Cajita::Own() );

    // Time step.
    int num_step = t_final / delta_t;
    double time = 0.0;
    for ( int t = 0; t < num_step; ++t )
    {
        // Print time step info.
        if ( 0 == global_grid->blockId() &&
             0 == t % write_freq )
            std::cout << "Step " << t+1 << "/" << num_step
                      << " - time " << time << std::endl;

        // Reinitialize slices after communications.
        x_p = Cabana::slice<ParticleField::x>( particles, "position" );
        m_p = Cabana::slice<ParticleField::m>( particles, "mass" );
        v_p = Cabana::slice<ParticleField::v>( particles, "volume" );
        e_p = Cabana::slice<ParticleField::e>( particles, "energy" );
        t_p = Cabana::slice<ParticleField::t>( particles, "temperature" );

        // Output particles
        if ( 0 == t % write_freq )
            Harlow::SiloParticleWriter::writeTimeStep(
                *global_grid, t, time, x_p, m_p, v_p, e_p, t_p );

        // Gather grid values.
        node_halo->gather( *energy_old );
        node_halo->gather( *energy_new );

        // Views.
        auto update_view = energy_update->view();
        auto mass_view = mass->view();
        auto energy_old_view = energy_old->view();
        auto energy_new_view = energy_new->view();

        // Mass and update scatter views.
        auto update_sv =
            Kokkos::Experimental::create_scatter_view( update_view );
        auto mass_sv =
            Kokkos::Experimental::create_scatter_view( mass_view );

        // Clear the scatter arrays.
        Cajita::ArrayOp::assign( *energy_update, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *mass, 0.0, Cajita::Ghost() );

        // Particle loop
        Kokkos::parallel_for(
            "particle_step",
            Kokkos::RangePolicy<execution_space>(0,particles.size()),
            KOKKOS_LAMBDA( const int p ){

                // Spline data.
                sd_type sd;

                // Get the particle position.
                double px[3] = { x_p(p,Dim::I), x_p(p,Dim::J), x_p(p,Dim::K) };

                // Evaluate the spline.
                Cajita::evaluateSpline( local_mesh, px, sd );

                // Update particle energy with the grid energy increment.
                double e_old;
                Cajita::G2P::value( energy_old_view, sd, e_old );
                double e_new;
                Cajita::G2P::value( energy_new_view, sd, e_new );
                e_p(p) += e_new - e_old;

                // Update the particle temperature
                t_p(p) = mat_0.temperature( e_p(p) );

                // Project energy gradient to the particle.
                double grad_e[3];
                Cajita::G2P::gradient( energy_new_view, sd, grad_e );

                // Scale the gradient.
                for ( int d = 0; d < 3; ++d )
                    grad_e[d] *= v_p(p) * mat_0.k / mat_0.c_v;

                // Project the divergence of the scaled gradient back to the
                // grid.
                Cajita::P2G::divergence( grad_e, sd, update_sv );

                // Add the source term.
                double source = v_p(p) * eval_source( px );
                Cajita::P2G::value( source, sd, update_sv );

                // Project mass to the grid.
                Cajita::P2G::value( m_p(p), sd, mass_sv );
            });

        // Complete the particle-grid scatter to the nodes.
        Kokkos::Experimental::contribute( mass_view, mass_sv );
        Kokkos::Experimental::contribute( update_view, update_sv );
        node_halo->scatter( *mass );
        node_halo->scatter( *energy_update );

        // Compute grid energy.
        Kokkos::parallel_for(
            "compute_grid_energy",
            Cajita::createExecutionPolicy( node_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                energy_old_view(i,j,k,0) = energy_new_view(i,j,k,0);
                if ( mass_view(i,j,k,0) > 0.0 )
                {
                    energy_new_view(i,j,k,0) +=
                        update_view(i,j,k,0) * delta_t / mass_view(i,j,k,0);
                }
                else
                {
                    energy_new_view(i,j,k,0) = 0.0;
                }
            });

        // Redistribute particles.
        Harlow::ParticleCommunication::redistribute(
            *local_grid, particles,
            std::integral_constant<std::size_t,ParticleField::x>() );

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
