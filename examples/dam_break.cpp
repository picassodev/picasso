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

//---------------------------------------------------------------------------//
// DamBreak example
template <class InterpolationType, class ParticleVelocity>
void DamBreak()
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = exec_space::memory_space;

    // Global bounding box.
    double cell_size = 0.05;
    std::array<int, 3> global_num_cell = { 20, 20, 20 };
    std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Get inputs for mesh.
    InputParser parser( "dam_break.json", "json" );
    const auto& pt = parser.propertyTree();

    Kokkos::Array<double, 6> global_box = {
        global_low_corner[0],  global_low_corner[1],  global_low_corner[2],
        global_high_corner[0], global_high_corner[1], global_high_corner[2] };
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
    Kokkos::Array<double, 6> block = { 0.0, 0.0, 0.0, 0.4, 0.4, 0.4 };
    double density = 1e3;

    auto momentum_init_functor =
        createParticleInitFunc( particles, ParticleVelocity(), block, density );

    int ppc = 2;
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
    double gamma = 7.0;
    double bulk_modulus = 1.0e+5;
    Kokkos::Array<double, 3> gravity = { 0.0, 0.0, -9.8 };
    Picasso::Properties props( gamma, bulk_modulus, gravity );

    // Time integragor
    auto time_integrator = Picasso::createExplicitMomentumIntegrator(
        mesh, InterpolationType(), ParticleVelocity(), props, 1.0e-4 );
    auto fm = Picasso::createFieldManager( mesh );
    time_integrator.setup( *fm );

    // steps
    while ( time_integrator.totalTime() < 1.0 )
    {
        // Write particle fields.
        Picasso::Output::outputParticles(
            MPI_COMM_WORLD, exec_space(), ParticleVelocity(),
            time_integrator.totalSteps(), 10, time_integrator.totalTime(),
            particles );

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
