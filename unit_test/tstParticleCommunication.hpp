#include <Cajita.hpp>

#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_Types.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <memory>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void redistributeTest( const Cajita::ManualPartitioner& partitioner,
                       const std::array<bool,3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 22, 19, 21 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner,
                                                        global_high_corner,
                                                        global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD,
                                                 global_mesh,
                                                 is_dim_periodic,
                                                 partitioner );

    // Create local block with a halo of 2.
    const int halo_size = 2;
    auto block = Cajita::createLocalGrid( global_grid, halo_size );
    auto local_mesh = Cajita::createLocalMesh<Kokkos::HostSpace>( *block );

    // Allocate a maximum number of particles assuming we have a halo on every
    // boundary.
    auto ghosted_cell_space =
        block->indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );
    int num_particle = ghosted_cell_space.size();
    using MemberTypes = Cabana::MemberTypes<double[3],int>;
    using ParticleContainer = Cabana::AoSoA<MemberTypes,Kokkos::HostSpace>;
    ParticleContainer particles( "particles", num_particle );
    auto coords = Cabana::slice<0>( particles, "coords" );
    auto linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Put particles in the center of every cell including halo cells if we
    // have them. Their ids should be equivalent to that of the rank they are
    // going to.
    int pid = 0;
    for ( int nk = -1; nk < 2; ++nk )
        for ( int nj = -1; nj < 2; ++nj )
            for ( int ni = -1; ni < 2; ++ni )
            {
                auto neighbor_rank = block->neighborRank(ni,nj,nk);
                if ( neighbor_rank >= 0 )
                {
                    auto shared_space = block->sharedIndexSpace(
                        Cajita::Ghost(),Cajita::Cell(),ni,nj,nk);
                    for ( int k = shared_space.min(Dim::K);
                          k < shared_space.max(Dim::K);
                          ++k )
                        for ( int j = shared_space.min(Dim::J);
                              j < shared_space.max(Dim::J);
                              ++j )
                            for ( int i = shared_space.min(Dim::I);
                                  i < shared_space.max(Dim::I);
                                  ++i )
                            {
                                // Set the coordinates at the cell center.
                                coords(pid,Dim::I) =
                                    local_mesh.lowCorner(Cajita::Ghost(),Dim::I) +
                                    (i + 0.5) * cell_size;
                                coords(pid,Dim::J) =
                                    local_mesh.lowCorner(Cajita::Ghost(),Dim::J) +
                                    (j + 0.5) * cell_size;
                                coords(pid,Dim::K) =
                                    local_mesh.lowCorner(Cajita::Ghost(),Dim::K) +
                                    (k + 0.5) * cell_size;

                                // Set the linear ids as the linear rank of
                                // the neighbor.
                                linear_ids(pid) = neighbor_rank;

                                // Increment the particle count.
                                ++pid;
                            }
                }
            }
    num_particle = pid;

    // Copy to the device space.
    particles.resize( num_particle );

    auto particles_mirror = Cabana::create_mirror_view_and_copy(
        TEST_DEVICE(), particles );

    // Redistribute the particles.
    ParticleCommunication::redistribute(
        *block,
        Cabana::slice<0>(particles_mirror),
        particles_mirror,
        true );

    // Copy back to check.
    particles = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particles_mirror );
    coords = Cabana::slice<0>( particles, "coords" );
    linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.size(), num_particle );
    EXPECT_EQ( linear_ids.size(), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_EQ( linear_ids(p), global_grid->blockId() );

    // Check that all of the particles are now in the local domain.
    double low_c[3] = { local_mesh.lowCorner( Cajita::Own(), Dim::I ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::J ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    double high_c[3] = { local_mesh.highCorner( Cajita::Own(), Dim::I ),
                         local_mesh.highCorner( Cajita::Own(), Dim::J ),
                         local_mesh.highCorner( Cajita::Own(), Dim::K ) };
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
void localOnlyTest( const Cajita::ManualPartitioner& partitioner,
                    const std::array<bool,3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 22, 19, 21 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner,
                                                        global_high_corner,
                                                        global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD,
                                                 global_mesh,
                                                 is_dim_periodic,
                                                 partitioner );

    // Get the local block with a halo of 2.
    const int halo_size = 2;
    auto block = Cajita::createLocalGrid( global_grid, halo_size );
    auto local_mesh = Cajita::createLocalMesh<Kokkos::HostSpace>( *block );

    // Allocate particles
    auto owned_cell_space =
        block->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    int num_particle = owned_cell_space.size();
    using MemberTypes = Cabana::MemberTypes<double[3],int>;
    using ParticleContainer = Cabana::AoSoA<MemberTypes,Kokkos::HostSpace>;
    ParticleContainer particles( "particles", num_particle );
    auto coords = Cabana::slice<0>( particles, "coords" );
    auto linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Put particles in the center of every local cell.
    int pid = 0;
    for ( int k = 0; k < owned_cell_space.extent(Dim::K); ++k )
        for ( int j = 0; j < owned_cell_space.extent(Dim::J); ++j )
            for ( int i = 0; i < owned_cell_space.extent(Dim::I); ++i )
            {
                // Set the coordinates at the cell center.
                coords(pid,Dim::I) =
                    local_mesh.lowCorner(Cajita::Own(),Dim::I) +
                    (i + 0.5) * cell_size;
                coords(pid,Dim::J) =
                    local_mesh.lowCorner(Cajita::Own(),Dim::J) +
                    (j + 0.5) * cell_size;
                coords(pid,Dim::K) =
                    local_mesh.lowCorner(Cajita::Own(),Dim::K) +
                    (k + 0.5) * cell_size;

                // Set the linear rank
                linear_ids(pid) = global_grid->blockId();

                // Increment the particle count.
                ++pid;
            }

    // Copy to the device space.
    auto particles_mirror = Cabana::create_mirror_view_and_copy(
        TEST_DEVICE(), particles );

    // Redistribute the particles.
    ParticleCommunication::redistribute(
        *block,
        Cabana::slice<0>(particles_mirror),
        particles_mirror,
        true );

    // Copy back to check.
    particles = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particles_mirror );
    coords = Cabana::slice<0>( particles, "coords" );
    linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.size(), num_particle );
    EXPECT_EQ( linear_ids.size(), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_EQ( linear_ids(p), global_grid->blockId() );

    // Check that all of the particles are now in the local domain.
    double low_c[3] = { local_mesh.lowCorner( Cajita::Own(), Dim::I ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::J ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    double high_c[3] = { local_mesh.highCorner( Cajita::Own(), Dim::I ),
                         local_mesh.highCorner( Cajita::Own(), Dim::J ),
                         local_mesh.highCorner( Cajita::Own(), Dim::K ) };
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
    std::array<int,3> ranks_per_dim = {0,0,0};
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool,3> is_dim_periodic = {false,false,false};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool,3> is_dim_periodic = {true,true,true};

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = Cajita::ManualPartitioner( ranks_per_dim );
    redistributeTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, local_only_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool,3> is_dim_periodic = {true,true,true};
    localOnlyTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//

} // end namespace Test
