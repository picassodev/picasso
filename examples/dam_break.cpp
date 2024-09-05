#include <Picasso.hpp>

#include "sources/Particle_Init.hpp"
#include "sources/Picasso_BoundaryCondition.hpp"
#include "sources/Picasso_ExplicitMomentumUpdate.hpp"
#include "sources/Picasso_Output.hpp"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

using namespace Picasso;

template <typename Scalar, std::size_t N>
Kokkos::Array<Scalar, N> parserGetArray( InputParser parser,
                                         const std::string name )
{
    const auto& pt = parser.propertyTree();
    Kokkos::Array<Scalar, N> array;

    int d = 0;
    for ( auto& element : pt.get_child( name ) )
    {
        array[d] = element.second.get_value<Scalar>();
        ++d;
    }
    return array;
}
//---------------------------------------------------------------------------//
// DamBreak example
template <class InterpolationType, class ParticleVelocity>
void DamBreak()
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = exec_space::memory_space;

    // Get inputs for mesh.
    InputParser parser( "dam_break.json", "json" );
    const auto& pt = parser.propertyTree();

    // Global bounding box.
    auto global_box = parserGetArray<double, 6>( parser, "global_box" );
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = createUniformMesh( memory_space(), pt, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    Cabana::ParticleTraits<Example::Stress, ParticleVelocity, Example::Position,
                           Example::Mass, Example::Pressure, Example::Volume,
                           Example::DetDefGrad>
        fields;
    auto particles = Cabana::Grid::createParticleList<memory_space>(
        "test_particles", fields );

    // Initialize particles
    auto particle_box = parserGetArray<double, 6>( parser, "particle_box" );
    auto density = pt.get<double>( "density" );

    auto momentum_init_functor = createParticleInitFunc(
        particles, ParticleVelocity(), particle_box, density );

    auto ppc = pt.get<int>( "ppc" );
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
    auto gamma = pt.get<double>( "gamma" );
    auto bulk_modulus = pt.get<double>( "bulk_modulus" );
    auto gravity = parserGetArray<double, 3>( parser, "gravity" );
    Picasso::Properties props( gamma, bulk_modulus, gravity );

    // Time integragor
    auto dt = pt.get<double>( "dt" );
    auto time_integrator = Picasso::createExplicitMomentumIntegrator(
        mesh, InterpolationType(), ParticleVelocity(), props, dt );
    auto fm = Picasso::createFieldManager( mesh );
    time_integrator.setup( *fm );

    // steps
    auto final_time = pt.get<double>( "final_time" );
    auto write_frequency = pt.get<double>( "write_frequency" );
    while ( time_integrator.totalTime() < final_time )
    {
        // Write particle fields.
        Picasso::Output::outputParticles(
            MPI_COMM_WORLD, exec_space(), ParticleVelocity(),
            time_integrator.totalSteps(), write_frequency,
            time_integrator.totalTime(), particles );

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

    DamBreak<Picasso::APIC::APicTag, Picasso::APIC::Field::Velocity>();

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
