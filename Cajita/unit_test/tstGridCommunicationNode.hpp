#include <Cajita_GridCommunication.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GridField.hpp>
#include <Cajita_GridExecPolicy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
// Check the gather for a scalar field.
template<class ViewType>
void checkGather( const GridBlock& block,
                  typename ViewType::value_type data_val,
                  ViewType field )
{
    // -I face.
    if ( block.hasHalo(DomainBoundary::LowX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                EXPECT_EQ( field(0,j,k,0), data_val );

    // +I face.
    if ( block.hasHalo(DomainBoundary::HighX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                EXPECT_EQ( field(block.numEntity(MeshEntity::Node,Dim::I)-1,j,k,0), data_val );

    // -J face.
    if ( block.hasHalo(DomainBoundary::LowY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                EXPECT_EQ( field(i,0,k,0), data_val );

    // +J face.
    if ( block.hasHalo(DomainBoundary::HighY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                EXPECT_EQ( field(i,block.numEntity(MeshEntity::Node,Dim::J)-1,k,0), data_val );

    // -K face.
    if ( block.hasHalo(DomainBoundary::LowZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            EXPECT_EQ( field(i,j,0,0), data_val );

    // +K face.
    if ( block.hasHalo(DomainBoundary::HighZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
                EXPECT_EQ( field(i,j,block.numEntity(MeshEntity::Node,Dim::K)-1,0), data_val );
}

//---------------------------------------------------------------------------//
// Check the gather for a vector field.
template<class ViewType>
void checkVectorGather( const GridBlock& block,
                        typename ViewType::value_type data_val,
                        ViewType field )
{
    // -I face.
    if ( block.hasHalo(DomainBoundary::LowX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                    EXPECT_EQ( field(0,j,k,d), data_val );

    // +I face.
    if ( block.hasHalo(DomainBoundary::HighX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                    EXPECT_EQ( field(block.numEntity(MeshEntity::Node,Dim::I)-1,j,k,d), data_val );

    // -J face.
    if ( block.hasHalo(DomainBoundary::LowY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                    EXPECT_EQ( field(i,0,k,d), data_val );

    // +J face.
    if ( block.hasHalo(DomainBoundary::HighY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K);
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K); ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                    EXPECT_EQ( field(i,block.numEntity(MeshEntity::Node,Dim::J)-1,k,d), data_val );

    // -K face.
    if ( block.hasHalo(DomainBoundary::LowZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
              j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
            for ( unsigned d = 0; d < field.extent(3); ++d)
                EXPECT_EQ( field(i,j,0,d), data_val );

    // +K face.
    if ( block.hasHalo(DomainBoundary::HighZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I);
              i < block.localEntityEnd(MeshEntity::Node,Dim::I); ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J);
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J); ++j )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                    EXPECT_EQ( field(i,j,block.numEntity(MeshEntity::Node,Dim::K)-1,d), data_val );
}

//---------------------------------------------------------------------------//
// Check the scatter for a scalar field. Note that some halo nodes go to
// multiple neighbors so these will get multiple contributions.
template<class ViewType>
void checkScatter( const GridBlock& block,
                   typename ViewType::value_type data_val,
                   ViewType field )
{
    // Start with the interior halo values. Interior values should have
    // received a contribution from only one neighbor.

    // -I face.
    if ( block.hasHalo(DomainBoundary::LowX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
              j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
            {
                int i = 2;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }

    // +I face.
    if ( block.hasHalo(DomainBoundary::HighX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
              j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j)
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
            {
                int i = block.numEntity(MeshEntity::Node,Dim::I)-3;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }

    // -J face.
    if ( block.hasHalo(DomainBoundary::LowY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
            {
                int j = 2;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }

    // +J face.
    if ( block.hasHalo(DomainBoundary::HighY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
            {
                int j = block.numEntity(MeshEntity::Node,Dim::J)-3;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }

    // -K face.
    if ( block.hasHalo(DomainBoundary::LowZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
            {
                int k = 2;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }

    // +K face.
    if ( block.hasHalo(DomainBoundary::HighZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
            {
                int k = block.numEntity(MeshEntity::Node,Dim::K)-3;
                EXPECT_EQ( field(i,j,k,0), 2*data_val );
            }
}

//---------------------------------------------------------------------------//
// Check the scatter for a vector field. Note that some halo nodes go to
// multiple neighbors so these will get multiple contributions.
template<class ViewType>
void checkVectorScatter( const GridBlock& block,
                         typename ViewType::value_type data_val,
                         ViewType field )
{
    // Start with the interior halo values. Interior values should have
    // received a contribution from only one neighbor.

    // -I face.
    if ( block.hasHalo(DomainBoundary::LowX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
              j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int i = 2;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }

    // +I face.
    if ( block.hasHalo(DomainBoundary::HighX) )
        for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
              j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j)
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int i = block.numEntity(MeshEntity::Node,Dim::I)-3;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }

    // -J face.
    if ( block.hasHalo(DomainBoundary::LowY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int j = 2;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }

    // +J face.
    if ( block.hasHalo(DomainBoundary::HighY) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int k = block.localEntityBegin(MeshEntity::Node,Dim::K) + 2;
                  k < block.localEntityEnd(MeshEntity::Node,Dim::K) - 2; ++k )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int j = block.numEntity(MeshEntity::Node,Dim::J)-3;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }

    // -K face.
    if ( block.hasHalo(DomainBoundary::LowZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int k = 2;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }

    // +K face.
    if ( block.hasHalo(DomainBoundary::HighZ) )
        for ( int i = block.localEntityBegin(MeshEntity::Node,Dim::I) + 2;
              i < block.localEntityEnd(MeshEntity::Node,Dim::I) - 2; ++i )
            for ( int j = block.localEntityBegin(MeshEntity::Node,Dim::J) + 2;
                  j < block.localEntityEnd(MeshEntity::Node,Dim::J) - 2; ++j )
                for ( unsigned d = 0; d < field.extent(3); ++d)
                {
                    int k = block.numEntity(MeshEntity::Node,Dim::K)-3;
                    EXPECT_EQ( field(i,j,k,d), 2*data_val );
                }
}

//---------------------------------------------------------------------------//
void gatherScatterTest( const std::vector<int>& ranks_per_dim,
                        const std::vector<bool>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<GlobalGrid>(
        MPI_COMM_WORLD,
        ranks_per_dim,
        is_dim_periodic,
        global_low_corner,
        global_high_corner,
        cell_size );

    // Create a scalar node field on the grid.
    int halo_width = 1;
    GridField<double,TEST_MEMSPACE> grid_field(
        global_grid,
        1,
        MeshEntity::Node,
        halo_width,
        "TestField" );

    // Fill the locally owned field with data.
    double data_val = 2.3;
    auto field = grid_field.data();
    Kokkos::deep_copy( field, 0.0 );
    Kokkos::parallel_for(
        "test field fill",
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>(
            grid_field.block(),grid_field.location()),
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            field(i,j,k,0) = data_val; }
        );

    // Gather into the halo.
    GridCommunication::gather(
        grid_field, halo_width, GridCommunication::GraphCommNodeTag() );

    // Check the gather. The halo should have received the data value if its
    // not on a physical boundary or if it is on a periodic boundary.
    auto field_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), field );
    checkGather( grid_field.block(), data_val, field_mirror );

    // Now scatter back.
    GridCommunication::scatter(
        grid_field, halo_width, GridCommunication::GraphCommNodeTag() );

    // Check the scatter. The interior nodes on the boundary should now have
    // 2x the data value if not on a physical boundary or if the boundary is
    // periodic.
    Kokkos::deep_copy( field_mirror, field );
    checkScatter( grid_field.block(), data_val, field_mirror );
}

//---------------------------------------------------------------------------//
void vectorFieldTest()
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Boundaries are not periodic.
    std::vector<bool> is_dim_periodic = {false,false,false};

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<GlobalGrid>(
        MPI_COMM_WORLD,
        ranks_per_dim,
        is_dim_periodic,
        global_low_corner,
        global_high_corner,
        cell_size );

    // Create a vector node field on the grid.
    int halo_width = 1;
    GridField<double,TEST_MEMSPACE> grid_field(
        global_grid,
        3,
        MeshEntity::Node,
        halo_width,
        "TestField" );

    // Fill the locally owned field with data.
    double data_val = 2.3;
    auto field = grid_field.data();
    Kokkos::deep_copy( field, 0.0 );
    Kokkos::parallel_for(
        "test field fill",
        GridExecution::createLocalEntityPolicy<TEST_EXECSPACE>(
            grid_field.block(),grid_field.location()),
        KOKKOS_LAMBDA(const int i, const int j, const int k){
            for ( int d = 0; d < 3; ++d )
                field(i,j,k,d) = data_val; }
        );

    // Gather into the halo.
    GridCommunication::gather(
        grid_field, halo_width, GridCommunication::GraphCommNodeTag() );

    // Check the gather. The halo should have received the data value if its
    // not on a physical boundary or if it is on a periodic boundary.
    auto field_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), field );
    checkVectorGather( grid_field.block(), data_val, field_mirror );

    // Now scatter back.
    GridCommunication::scatter(
        grid_field, halo_width, GridCommunication::GraphCommNodeTag() );

    // Check the scatter. The interior nodes on the boundary should now have
    // 2x the data value if not on a physical boundary or if the boundary is
    // periodic.
    Kokkos::deep_copy( field_mirror, field );
    checkVectorScatter( grid_field.block(), data_val, field_mirror );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, graph_not_periodic_test )
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
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, graph_periodic_test )
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
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    gatherScatterTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, graph_vector_field_test )
{
    vectorFieldTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
