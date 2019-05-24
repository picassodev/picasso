#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GridBlock.hpp>

#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_ParticleFieldOps.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <memory>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void redistributeTest( const std::vector<int>& ranks_per_dim,
                       const std::vector<bool>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 22, 19, 21 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<Cajita::GlobalGrid>(
        MPI_COMM_WORLD,
        ranks_per_dim,
        is_dim_periodic,
        global_low_corner,
        global_high_corner,
        cell_size );

    // Get the local block with a halo of 2.
    const int halo_size = 2;
    Cajita::GridBlock block;
    block.assign( global_grid->block(), halo_size );

    // Get my local ranks.
    int my_linear_rank;
    MPI_Comm_rank( global_grid->cartesianComm(), &my_linear_rank );
    int my_cart_rank[3];
    MPI_Cart_coords( global_grid->cartesianComm(), my_linear_rank, 3, my_cart_rank );

    // Allocate a maximum number of particles assuming we have a halo on every
    // boundary.
    int num_particle = block.numEntity( MeshEntity::Cell, Dim::I ) *
                       block.numEntity( MeshEntity::Cell, Dim::J ) *
                       block.numEntity( MeshEntity::Cell, Dim::K );
    Kokkos::View<double*[3],Kokkos::HostSpace> coords( "coords", num_particle );
    Kokkos::View<int*,Kokkos::HostSpace> linear_ids( "linear_ids", num_particle );
    Kokkos::View<int*[3][3],Kokkos::HostSpace> matrix_ids( "matrix_ids", num_particle );

    // Determine if a given neighbor has entities. Zero always returns true.
    auto has_entities =
        [&]( const int dim, const int logical_index ){
            bool halo_check = true;
            if ( -1 == logical_index )
                halo_check = block.hasHalo(2*dim);
            else if ( 1 == logical_index )
                halo_check = block.hasHalo(2*dim+1);
            return halo_check;
        };


    // Put particles in the center of every cell including halo cells if we
    // have them. Their ids should be equivalent to that of the rank they are
    // going to.
    int pid = 0;
    for ( int nk = -1; nk < 2; ++nk )
        for ( int nj = -1; nj < 2; ++nj )
            for ( int ni = -1; ni < 2; ++ni )
            {
                if ( has_entities(Dim::I,ni) &&
                     has_entities(Dim::J,nj) &&
                     has_entities(Dim::K,nk) )
                {
                    for ( int k = block.haloEntityBegin(MeshEntity::Cell,Dim::K,nk,halo_size);
                          k < block.haloEntityEnd(MeshEntity::Cell,Dim::K,nk,halo_size);
                          ++k )
                        for ( int j = block.haloEntityBegin(MeshEntity::Cell,Dim::J,nj,halo_size);
                              j < block.haloEntityEnd(MeshEntity::Cell,Dim::J,nj,halo_size);
                              ++j )
                            for ( int i = block.haloEntityBegin(MeshEntity::Cell,Dim::I,ni,halo_size);
                                  i < block.haloEntityEnd(MeshEntity::Cell,Dim::I,ni,halo_size);
                                  ++i )
                            {
                                // Set the coordinates at the cell center.
                                coords(pid,Dim::I) =
                                    block.lowCorner(Dim::I) + (i + 0.5) * cell_size;
                                coords(pid,Dim::J) =
                                    block.lowCorner(Dim::J) + (j + 0.5) * cell_size;
                                coords(pid,Dim::K) =
                                    block.lowCorner(Dim::K) + (k + 0.5) * cell_size;

                                // Set the cartesian rank
                                int neighbor_cart_rank[3] =
                                    { my_cart_rank[Dim::I] + ni,
                                      my_cart_rank[Dim::J] + nj,
                                      my_cart_rank[Dim::K] + nk };

                                // Set the linear rank
                                MPI_Cart_rank( global_grid->cartesianComm(),
                                               neighbor_cart_rank,
                                               &linear_ids(pid) );

                                // Set the cartesian rank. We map back so we
                                // get the wrap-around effect for periodic
                                // boundaries (e.g. when the logical index
                                // places us out of bounds).
                                MPI_Cart_coords( global_grid->cartesianComm(),
                                                 linear_ids(pid),
                                                 3,
                                                 neighbor_cart_rank );
                                for ( int d0 = 0; d0 < 3; ++d0 )
                                    for ( int d1 = 0; d1 < 3; ++d1 )
                                        matrix_ids( pid, d0, d1 ) = neighbor_cart_rank[d1];

                                // Increment the particle count.
                                ++pid;
                            }
                }
            }
    num_particle = pid;

    // Resize.
    ParticleFieldOps::resize( num_particle, coords, linear_ids, matrix_ids );

    // Copy the data to the test space.
    auto coords_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), coords );
    auto linear_ids_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), linear_ids );
    auto matrix_ids_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), matrix_ids );

    // Redistribute the particles.
    ParticleCommunication::redistribute(
        *global_grid, coords_mirror, linear_ids_mirror, matrix_ids_mirror );

    // Copy back to check.
    Kokkos::deep_copy( coords, coords_mirror );
    Kokkos::deep_copy( linear_ids, linear_ids_mirror );
    Kokkos::deep_copy( matrix_ids, matrix_ids_mirror );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.extent(0), num_particle );
    EXPECT_EQ( linear_ids.extent(0), num_particle );
    EXPECT_EQ( matrix_ids.extent(0), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( linear_ids(p), my_linear_rank );

        for ( int d0 = 0; d0 < 3; ++d0 )
            for ( int d1 = 0; d1 < 3; ++d1 )
                EXPECT_EQ( matrix_ids(p,d0,d1), my_cart_rank[d1] );
    }

    // Check that all of the particles are now in the local domain.
    Cajita::GridBlock local_block;
    local_block.assign( global_grid->block(), 0 );
    double low_c[3] = { local_block.lowCorner( Dim::I ),
                        local_block.lowCorner( Dim::J ),
                        local_block.lowCorner( Dim::K ) };
    double high_c[3] = { low_c[Dim::I] +
                         local_block.numEntity( MeshEntity::Cell, Dim::I ) *
                         local_block.cellSize(),
                         low_c[Dim::J] +
                         local_block.numEntity( MeshEntity::Cell, Dim::J ) *
                         local_block.cellSize(),
                         low_c[Dim::K] +
                         local_block.numEntity( MeshEntity::Cell, Dim::K ) *
                         local_block.cellSize() };
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_TRUE( coords(p,d) >= low_c[d] );
            EXPECT_TRUE( coords(p,d) <= high_c[d] );
        }
}

//---------------------------------------------------------------------------//
// The objective of this test is to check how the redistribution works when we
// have no particles to redistribute. In this case we put no particles in the
// halo so no communication should occur. This ensures the graph communication
// works when some neighbors get no data.
void localOnlyTest( const std::vector<int>& ranks_per_dim,
                    const std::vector<bool>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 22, 19, 21 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<Cajita::GlobalGrid>(
        MPI_COMM_WORLD,
        ranks_per_dim,
        is_dim_periodic,
        global_low_corner,
        global_high_corner,
        cell_size );

    // Get the local block with a halo of 2.
    const int halo_size = 2;
    Cajita::GridBlock block;
    block.assign( global_grid->block(), halo_size );

    // Get my local ranks.
    int my_linear_rank;
    MPI_Comm_rank( global_grid->cartesianComm(), &my_linear_rank );
    int my_cart_rank[3];
    MPI_Cart_coords( global_grid->cartesianComm(), my_linear_rank, 3, my_cart_rank );

    // Allocate particles
    int num_particle = block.localNumEntity( MeshEntity::Cell, Dim::I ) *
                       block.localNumEntity( MeshEntity::Cell, Dim::J ) *
                       block.localNumEntity( MeshEntity::Cell, Dim::K );
    Kokkos::View<double*[3],Kokkos::HostSpace> coords( "coords", num_particle );
    Kokkos::View<int*,Kokkos::HostSpace> linear_ids( "linear_ids", num_particle );
    Kokkos::View<int*[3][3],Kokkos::HostSpace> matrix_ids( "matrix_ids", num_particle );

    // Put particles in the center of every local cell.
    int pid = 0;
    for ( int k = block.localEntityBegin(MeshEntity::Cell,Dim::K);
          k < block.localEntityEnd(MeshEntity::Cell,Dim::K);
          ++k )
        for ( int j = block.localEntityBegin(MeshEntity::Cell,Dim::J);
              j < block.localEntityEnd(MeshEntity::Cell,Dim::J);
              ++j )
            for ( int i = block.localEntityBegin(MeshEntity::Cell,Dim::I);
                  i < block.localEntityEnd(MeshEntity::Cell,Dim::I);
                  ++i )
            {
                // Set the coordinates at the cell center.
                coords(pid,Dim::I) =
                    block.lowCorner(Dim::I) + (i + 0.5) * cell_size;
                coords(pid,Dim::J) =
                    block.lowCorner(Dim::J) + (j + 0.5) * cell_size;
                coords(pid,Dim::K) =
                    block.lowCorner(Dim::K) + (k + 0.5) * cell_size;

                // Set the cartesian rank
                for ( int d0 = 0; d0 < 3; ++d0 )
                    for ( int d1 = 0; d1 < 3; ++d1 )
                        matrix_ids( pid, d0, d1 ) = my_cart_rank[d1];

                // Set the linear rank
                linear_ids(pid) = my_linear_rank;

                // Increment the particle count.
                ++pid;
            }

    // Copy the data to the test space.
    auto coords_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), coords );
    auto linear_ids_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), linear_ids );
    auto matrix_ids_mirror = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), matrix_ids );

    // Redistribute the particles.
    ParticleCommunication::redistribute(
        *global_grid, coords_mirror, linear_ids_mirror, matrix_ids_mirror );

    // Copy back to check.
    Kokkos::deep_copy( coords, coords_mirror );
    Kokkos::deep_copy( linear_ids, linear_ids_mirror );
    Kokkos::deep_copy( matrix_ids, matrix_ids_mirror );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.extent(0), num_particle );
    EXPECT_EQ( linear_ids.extent(0), num_particle );
    EXPECT_EQ( matrix_ids.extent(0), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( linear_ids(p), my_linear_rank );

        for ( int d0 = 0; d0 < 3; ++d0 )
            for ( int d1 = 0; d1 < 3; ++d1 )
                EXPECT_EQ( matrix_ids(p,d0,d1), my_cart_rank[d1] );
    }

    // Check that all of the particles are now in the local domain.
    Cajita::GridBlock local_block;
    local_block.assign( global_grid->block(), 0 );
    double low_c[3] = { local_block.lowCorner( Dim::I ),
                        local_block.lowCorner( Dim::J ),
                        local_block.lowCorner( Dim::K ) };
    double high_c[3] = { low_c[Dim::I] +
                         local_block.numEntity( MeshEntity::Cell, Dim::I ) *
                         local_block.cellSize(),
                         low_c[Dim::J] +
                         local_block.numEntity( MeshEntity::Cell, Dim::J ) *
                         local_block.cellSize(),
                         low_c[Dim::K] +
                         local_block.numEntity( MeshEntity::Cell, Dim::K ) *
                         local_block.cellSize() };
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_TRUE( coords(p,d) >= low_c[d] );
            EXPECT_TRUE( coords(p,d) <= high_c[d] );
        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, not_periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Boundaries are not periodic.
    std::vector<bool> is_dim_periodic = {false,false,false};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Every boundary is periodic
    std::vector<bool> is_dim_periodic = {true,true,true};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    redistributeTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, local_only_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Every boundary is periodic
    std::vector<bool> is_dim_periodic = {true,true,true};
    localOnlyTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//

} // end namespace Test
