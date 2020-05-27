#include <Picasso_LPBF_Solver.hpp>

#include <Picasso_InputParser.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

//---------------------------------------------------------------------------//
void run( const Picasso::InputParser& input )
{
    // Get the input.
    const auto& ptree = input.propertyTree();

    // Create the solver.
    auto device_type = ptree.get<std::string>("device_type");
    auto solver =
        Picasso::LPBF::createSolver( device_type, ptree, MPI_COMM_WORLD );

    // Solve the problem.
    solver->solve();
}

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize environment.
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Parse input.
    Picasso::InputParser input( argc, argv );

    // Run.
    run( input );

    // Finalize the environment.
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
