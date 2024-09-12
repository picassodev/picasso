#include <Picasso.hpp>

#include "sources/Particle_Init.hpp"
#include "sources/Picasso_BoundaryCondition.hpp"
#include "sources/Picasso_ExplicitMomentumUpdate.hpp"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

using namespace Picasso;

//---------------------------------------------------------------------------//
// DamBreak example
template <class InterpolationType, class ParticleVelocity>
void DamBreak()
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = exec_space::memory_space;

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "dam_break.json" );

    // Global bounding box.
    auto global_box = copy<double, 6>( inputs["global_box"] );
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = createUniformMesh( memory_space(), inputs, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    Cabana::ParticleTraits<Stress, ParticleVelocity, Position, Mass, Pressure,
                           Volume, DetDefGrad>
        fields;
    auto particles = Cabana::Grid::createParticleList<memory_space>(
        "test_particles", fields );

    // Initialize particles
    auto particle_box = copy<double, 6>( inputs["particle_box"] );
    double density = inputs["density"];

    auto momentum_init_functor = createParticleInitFunc(
        particles, ParticleVelocity(), particle_box, density );

    double ppc = inputs["ppc"];
    auto local_grid = mesh->localGrid();
    Cabana::Grid::createParticles( Cabana::InitUniform(), exec_space(),
                                   momentum_init_functor, particles, ppc,
                                   *local_grid );

    // Boundary index space
    auto bc_index = local_grid->boundaryIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Node(), -1, 1, 0 );
    using bc_index_type = decltype( bc_index );

    BoundaryCondition<bc_index_type> bc{ bc_index };

    // Properties
    auto gamma = inputs["gamma"];
    auto bulk_modulus = inputs["bulk_modulus"];
    auto gravity = copy<double, 3>( inputs["gravity"] );
    Picasso::Properties props( gamma, bulk_modulus, gravity );

    // Time integragor
    auto dt = inputs["dt"];
    auto time_integrator = Picasso::createExplicitMomentumIntegrator(
        mesh, InterpolationType(), ParticleVelocity(), props, dt );
    auto fm = Picasso::createFieldManager( mesh );
    time_integrator.setup( *fm );

    // steps
    auto final_time = inputs["final_time"];
    auto write_frequency = inputs["write_frequency"];
    while ( time_integrator.totalTime() < final_time )
    {
        // Write particle fields.
        Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", MPI_COMM_WORLD,
            time_integrator.totalSteps(), time_integrator.totalTime(),
            particles.size(), particles.slice( Picasso::Position() ),
            particles.slice( Picasso::Pressure() ),
            particles.slice( ParticleVelocity() ),
            particles.slice( Picasso::Mass() ),
            particles.slice( Picasso::Volume() ) );

        printf( "aaa\n" );

        // Step.
        time_integrator.step( exec_space(), *fm, particles, *local_grid, bc );
    }
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
            First argument - file name for output \n \
            \n \
            Example: \n \
            $/: ./DamBreak inputs/dam_break.json\n" );
    std::string filename = argv[1];

    DamBreak<APicTag, Picasso::APIC::Field::Velocity>();

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
