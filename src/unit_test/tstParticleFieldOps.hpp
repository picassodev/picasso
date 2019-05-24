#include <Cajita_GridBlock.hpp>

#include <Harlow_ParticleFieldOps.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void resizeTest()
{
    // Create some fields.
    int size_1 = 839;
    Kokkos::View<double*[3][2],TEST_MEMSPACE> view_1( "view_1", size_1 );
    Kokkos::View<int*[3],TEST_MEMSPACE> view_2( "view_2", size_1 );
    Kokkos::View<float*,TEST_MEMSPACE> view_3( "view_3", size_1 );

    // Resize them.
    int size_2 = 9083;
    ParticleFieldOps::resize( size_2, view_1, view_2, view_3 );
    EXPECT_EQ( size_2, view_1.extent(0) );
    EXPECT_EQ( size_2, view_2.extent(0) );
    EXPECT_EQ( size_2, view_3.extent(0) );
}

//---------------------------------------------------------------------------//
void cellBinningTest()
{
    // Create a local grid block.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location =
        { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 2;
    Cajita::GridBlock block( low_corner, num_cell, boundary_location,
                             periodic, cell_size, halo_width );

    // Put a particle in the center of each cell. The kokkos 3d binning
    // operator will sort by X first and Z last so build them in opposite
    // order (X the fastest and Z the slowest). Create a list of ids to sort
    // as well that should be in linear order when sorted.
    int num_particle = num_cell[Dim::I] * num_cell[Dim::J] * num_cell[Dim::K];
    Kokkos::View<double*[3],Kokkos::HostSpace> coords_host( "coords", num_particle );
    Kokkos::View<int*,Kokkos::HostSpace> ids_host( "ids", num_particle );
    int pid = 0;
    for ( int k = 0; k < num_cell[Dim::K]; ++k )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int i = 0; i < num_cell[Dim::I]; ++i, ++pid )
            {
                coords_host(pid,Dim::I) =
                    block.lowCorner(Dim::I) + (i + 0.5) * cell_size;
                coords_host(pid,Dim::J) =
                    block.lowCorner(Dim::J) + (j + 0.5) * cell_size;
                coords_host(pid,Dim::K) =
                    block.lowCorner(Dim::K) + (k + 0.5) * cell_size;
                ids_host(pid) = k + num_cell[Dim::K] * ( j + num_cell[Dim::J] * i );
            }

    // Copy to the test space.
    auto coords = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), coords_host );
    auto ids = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), ids_host );

    // Sort.
    ParticleFieldOps::binByCellAndPermute( block, coords, ids );

    // Copy back to the host.
    Kokkos::deep_copy( coords_host, coords );
    Kokkos::deep_copy( ids_host, ids );

    // Loop through in opposite order and check that they got sorted.
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
            {
                EXPECT_EQ( coords_host(pid,Dim::I),
                           block.lowCorner(Dim::I) + (i + 0.5) * cell_size );
                EXPECT_EQ( coords_host(pid,Dim::J),
                           block.lowCorner(Dim::J) + (j + 0.5) * cell_size );
                EXPECT_EQ( coords_host(pid,Dim::K),
                           block.lowCorner(Dim::K) + (k + 0.5) * cell_size );
                EXPECT_EQ( ids_host(pid), pid );
            }
}

//---------------------------------------------------------------------------//
void keyBinningTest()
{
    // Create a local grid block.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location =
        { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 2;
    Cajita::GridBlock block( low_corner, num_cell, boundary_location,
                             periodic, cell_size, halo_width );

    // Put a particle in the center of each cell. The kokkos 1d binning
    // operator will sort the particles linearly by cell id. Create a list of
    // ids to sort as well that should be in linear order when sorted.
    int num_particle = num_cell[Dim::I] * num_cell[Dim::J] * num_cell[Dim::K];
    Kokkos::View<double*[3],Kokkos::HostSpace> coords_host( "coords", num_particle );
    Kokkos::View<int*,Kokkos::HostSpace> ids_host( "ids", num_particle );
    int pid = 0;
    for ( int k = 0; k < num_cell[Dim::K]; ++k )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int i = 0; i < num_cell[Dim::I]; ++i, ++pid )
            {
                coords_host(pid,Dim::I) =
                    block.lowCorner(Dim::I) + (i + 0.5) * cell_size;
                coords_host(pid,Dim::J) =
                    block.lowCorner(Dim::J) + (j + 0.5) * cell_size;
                coords_host(pid,Dim::K) =
                    block.lowCorner(Dim::K) + (k + 0.5) * cell_size;
                ids_host(pid) = k + num_cell[Dim::K] * ( j + num_cell[Dim::J] * i );
            }

    // Copy to the test space.
    auto coords = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), coords_host );
    auto ids = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), ids_host );

    // Sort.
    ParticleFieldOps::binByKeyAndPermute(
        ids, 1, 0, pid, coords );

    // Copy back to the host.
    Kokkos::deep_copy( coords_host, coords );
    Kokkos::deep_copy( ids_host, ids );

    // Loop through in opposite order and check that they got sorted.
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
            {
                EXPECT_EQ( coords_host(pid,Dim::I),
                           block.lowCorner(Dim::I) + (i + 0.5) * cell_size );
                EXPECT_EQ( coords_host(pid,Dim::J),
                           block.lowCorner(Dim::J) + (j + 0.5) * cell_size );
                EXPECT_EQ( coords_host(pid,Dim::K),
                           block.lowCorner(Dim::K) + (k + 0.5) * cell_size );
                EXPECT_EQ( ids_host(pid), pid );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, resize_test )
{
    resizeTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, cell_binning_test )
{
    cellBinningTest();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, key_binning_test )
{
    keyBinningTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
