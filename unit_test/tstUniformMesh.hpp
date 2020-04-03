#include <Harlow_UniformMesh.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_InputParser.hpp>

#include <Harlow_ParticleInit.hpp>
#include <Harlow_SiloParticleWriter.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void constructionTest()
{
    // Global parameters.
    Kokkos::Array<float,6> global_box = { -10.0, -10.0, -10.0,
                                          10.0, 10.0, 10.0 };
    double cell_size = 0.5;
    int num_cell = 20.0 / cell_size;
    int minimum_halo_size = 1;

    // Get inputs for mesh 1.
    InputParser parser_1( "uniform_mesh_test_1.json", "json" );
    auto pt1 = parser_1.propertyTree();

    // Make mesh 1.
    UniformMesh<TEST_MEMSPACE> mesh_1(
        pt1, global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Check cell sizes.
    EXPECT_EQ( mesh_1.cellSize(), cell_size );

    // Check grid 1.
    const auto& global_grid_1 = mesh_1.localGrid().globalGrid();
    const auto& global_mesh_1 = global_grid_1.globalMesh();

    EXPECT_EQ( global_mesh_1.lowCorner(0), global_box[0] );
    EXPECT_EQ( global_mesh_1.lowCorner(1), global_box[1]-cell_size );
    EXPECT_EQ( global_mesh_1.lowCorner(2), global_box[2] );

    EXPECT_EQ( global_mesh_1.highCorner(0), global_box[3] );
    EXPECT_EQ( global_mesh_1.highCorner(1), global_box[4]+cell_size );
    EXPECT_EQ( global_mesh_1.highCorner(2), global_box[5] );

    EXPECT_EQ( global_grid_1.globalNumEntity(Cajita::Cell(),0), num_cell );
    EXPECT_EQ( global_grid_1.globalNumEntity(Cajita::Cell(),1), num_cell + 2);
    EXPECT_EQ( global_grid_1.globalNumEntity(Cajita::Cell(),2), num_cell );

    EXPECT_TRUE( global_grid_1.isPeriodic(0) );
    EXPECT_FALSE( global_grid_1.isPeriodic(1) );
    EXPECT_TRUE( global_grid_1.isPeriodic(2) );

    EXPECT_EQ( mesh_1.localGrid().haloCellWidth(), 2 );

    // Get inputs for mesh 2.
    InputParser parser_2( "uniform_mesh_test_2.json", "json" );
    auto pt2 = parser_2.propertyTree();

    // Make mesh 2.
    UniformMesh<TEST_MEMSPACE> mesh_2(
        pt2, global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Check cell sizes.
    EXPECT_EQ( mesh_2.cellSize(), cell_size );

    // Check grid 2.
    const auto& global_grid_2 = mesh_2.localGrid().globalGrid();
    const auto& global_mesh_2 = global_grid_2.globalMesh();

    EXPECT_EQ( global_mesh_2.lowCorner(0), global_box[0]-cell_size );
    EXPECT_EQ( global_mesh_2.lowCorner(1), global_box[1] );
    EXPECT_EQ( global_mesh_2.lowCorner(2), global_box[2]-cell_size );

    EXPECT_EQ( global_mesh_2.highCorner(0), global_box[3]+cell_size );
    EXPECT_EQ( global_mesh_2.highCorner(1), global_box[4] );
    EXPECT_EQ( global_mesh_2.highCorner(2), global_box[5]+cell_size );

    EXPECT_EQ( global_grid_2.globalNumEntity(Cajita::Cell(),0), num_cell + 2 );
    EXPECT_EQ( global_grid_2.globalNumEntity(Cajita::Cell(),1), num_cell );
    EXPECT_EQ( global_grid_2.globalNumEntity(Cajita::Cell(),2), num_cell + 2 );

    EXPECT_FALSE( global_grid_2.isPeriodic(0) );
    EXPECT_TRUE( global_grid_2.isPeriodic(1) );
    EXPECT_FALSE( global_grid_2.isPeriodic(2) );

    EXPECT_EQ( mesh_2.localGrid().haloCellWidth(), 1 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, construction_test )
{
    constructionTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
