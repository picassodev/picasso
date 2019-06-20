#include <Harlow_SiloParticleWriter.hpp>
#include <Harlow_Types.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <silo.h>

#include <mpi.h>

#include <memory>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void writeTest()
{
    // Let MPI compute the partitioning for this test.
    Cajita::UniformDimPartitioner partitioner;

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 22, 19, 21 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    std::vector<bool> is_dim_periodic = {false,false,false};
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD,
                                                 partitioner,
                                                 is_dim_periodic,
                                                 global_low_corner,
                                                 global_high_corner,
                                                 cell_size );

    // Allocate particles in the center of each cell.
    auto block = Cajita::createBlock( global_grid, 0 );
    auto owned_cell_space = block->indexSpace( Cajita::Own(), Cajita::Cell() );
    int num_particle = owned_cell_space.size();
    using DataTypes = Cabana::MemberTypes<double[3],   // coords
                                          double[3],   // vec
                                          float[3][3], // matrix
                                          int>;        // id.
    Cabana::AoSoA<DataTypes,TEST_MEMSPACE> aosoa( "particles", num_particle );
    auto coords = Cabana::slice<0>( aosoa, "coords" );
    auto vec = Cabana::slice<1>( aosoa, "vec" );
    auto matrix = Cabana::slice<2>( aosoa, "matrix" );
    auto ids = Cabana::slice<3>( aosoa, "ids" );

    // Put the particles in the center of each cell.
    int i_off = global_grid->globalOffset(Dim::I);
    int j_off = global_grid->globalOffset(Dim::J);
    int k_off = global_grid->globalOffset(Dim::K);
    auto aosoa_mirror = Cabana::create_mirror_view(
        Kokkos::HostSpace(), aosoa );
    auto coords_mirror = Cabana::slice<0>( aosoa_mirror, "coords" );
    auto vec_mirror = Cabana::slice<1>( aosoa_mirror, "vec" );
    auto matrix_mirror = Cabana::slice<2>( aosoa_mirror, "matrix" );
    auto ids_mirror = Cabana::slice<3>( aosoa_mirror, "ids" );
    int pid = 0;
    for ( int i = 0; i < owned_cell_space.extent(Dim::I); ++i )
        for ( int j = 0; j < owned_cell_space.extent(Dim::J); ++j )
            for ( int k = 0; k < owned_cell_space.extent(Dim::K); ++k, ++pid )
            {
                coords_mirror( pid, Dim::I ) =
                    block->lowCorner(Cajita::Own(),Dim::I) + (i+0.5) * cell_size;
                coords_mirror( pid, Dim::J ) =
                    block->lowCorner(Cajita::Own(),Dim::J) + (j+0.5) * cell_size;
                coords_mirror( pid, Dim::K ) =
                    block->lowCorner(Cajita::Own(),Dim::K) + (k+0.5) * cell_size;

                ids_mirror( pid ) = i + i_off + j + j_off + k + k_off;

                vec_mirror( pid, Dim::I ) = i + i_off;
                vec_mirror( pid, Dim::J ) = j + j_off;
                vec_mirror( pid, Dim::K ) = k + k_off;

                for ( int d = 0; d < 2; ++d )
                {
                    matrix_mirror(pid,d,Dim::I) = i + i_off;
                    matrix_mirror(pid,d,Dim::J) = j + j_off;
                    matrix_mirror(pid,d,Dim::K) = k + k_off;
                }
            }
    Cabana::deep_copy( aosoa, aosoa_mirror );

    // Write a time step to file.
    double time = 7.64;
    double step = 892;
    SiloParticleWriter::writeTimeStep(
        *global_grid, step, time, coords, ids, matrix, vec );

    // Move the particles and write again.
    double time_step_size = 0.32;
    time += time_step_size;
    ++step;
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            coords_mirror(p,d) += 1.32;
    Cabana::deep_copy( coords, coords_mirror );
    SiloParticleWriter::writeTimeStep(
        *global_grid, step, time, coords, ids, matrix, vec );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, write_test )
{
    writeTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
