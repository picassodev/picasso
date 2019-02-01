#include <Harlow_Types.hpp>
#include <Harlow_GlobalGrid.hpp>
#include <Harlow_GridField.hpp>
#include <Harlow_GridFieldVectorOps.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void vectorOpTest()
{
    // Let MPI compute the partitioning for this test.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 101, 85, 99 };
    std::vector<bool> is_dim_periodic = {false,false,false};
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<GlobalGrid>( MPI_COMM_WORLD,
                                                     ranks_per_dim,
                                                     is_dim_periodic,
                                                     global_low_corner,
                                                     global_high_corner,
                                                     cell_size );

    int i_begin = global_grid->block().localCellBegin(Dim::I);
    int i_end = global_grid->block().localCellEnd(Dim::I);
    int j_begin = global_grid->block().localCellBegin(Dim::J);
    int j_end = global_grid->block().localCellEnd(Dim::J);
    int k_begin = global_grid->block().localCellBegin(Dim::K);
    int k_end = global_grid->block().localCellEnd(Dim::K);


    // TEST SCALAR FIELDS
    //-------------------
    GridField<double,TEST_EXECSPACE> scalar_A(
        global_grid, MeshEntity::Cell, 0, "scalar_A" );

    // Assign some data.
    double a_val = -3.23;
    GridFieldVectorOp::assign( a_val, scalar_A );

    // Check the assignment.
    auto scalar_A_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_A.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_EQ( scalar_A_mirror(i,j,k), a_val );

    // Check infinity norm.
    double expected_norm_inf = 1347.5;
    if ( 0 == comm_rank )
        scalar_A_mirror( 3, 2, 1 ) = expected_norm_inf;
    Kokkos::deep_copy( scalar_A.data(), scalar_A_mirror );
    auto norm_inf = GridFieldVectorOp::normInf( scalar_A );
    EXPECT_EQ( norm_inf, expected_norm_inf );

    // Reset the vector.
    GridFieldVectorOp::assign( a_val, scalar_A );

    // Check the 1-norm.
    int total_num_cell = global_grid->numCell(Dim::I) *
                         global_grid->numCell(Dim::J) *
                         global_grid->numCell(Dim::K);
    auto norm_1 = GridFieldVectorOp::norm1( scalar_A );
    double expected_norm_1 = total_num_cell * std::abs(a_val);
    EXPECT_FLOAT_EQ( norm_1, expected_norm_1 );

    // Check the 2-norm.
    auto norm_2 = GridFieldVectorOp::norm2( scalar_A );
    double expected_norm_2 = std::sqrt( total_num_cell * a_val * a_val );
    EXPECT_FLOAT_EQ( norm_2, expected_norm_2 );

    // Check the dot product.
    double b_val = 4.33;
    GridField<double,TEST_EXECSPACE> scalar_B(
        global_grid, MeshEntity::Cell, 0, "scalar_B" );
    GridFieldVectorOp::assign( b_val, scalar_B );
    auto dot_product = GridFieldVectorOp::dot( scalar_A, scalar_B );
    double expected_dot = total_num_cell * a_val * b_val;
    EXPECT_FLOAT_EQ( dot_product, expected_dot );

    // Check the scaling.
    double alpha = -8.99;
    GridFieldVectorOp::scale( alpha, scalar_A );
    Kokkos::deep_copy( scalar_A_mirror, scalar_A.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_EQ( scalar_A_mirror(i,j,k), a_val * alpha );

    double beta = 12.2;
    GridFieldVectorOp::scale( beta, scalar_B );
    auto scalar_B_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_B.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_EQ( scalar_B_mirror(i,j,k), b_val * beta );

    // Check the 2 vector update.
    GridField<double,TEST_EXECSPACE> scalar_C(
        global_grid, MeshEntity::Cell, 0, "scalar_C" );
    GridFieldVectorOp::update( 1.0 / alpha,
                               scalar_A,
                               1.0 / beta,
                               scalar_B,
                               scalar_C );
    auto scalar_C_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_C.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_EQ( scalar_C_mirror(i,j,k), a_val + b_val );

    // Check the 3 vector update.
    GridField<double,TEST_EXECSPACE> scalar_D(
        global_grid, MeshEntity::Cell, 0, "scalar_D" );
    double gamma = -12.1;
    GridFieldVectorOp::update( 1.0 / alpha,
                               scalar_A,
                               1.0 / beta,
                               scalar_B,
                               gamma,
                               scalar_C,
                               scalar_D );
    auto scalar_D_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_D.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_EQ( scalar_D_mirror(i,j,k),
                           (1.0 + gamma ) * (a_val + b_val) );


    // TEST Rank-1 FIELDS
    //-------------------
    GridField<double[3],TEST_EXECSPACE> rank_1_A(
        global_grid, MeshEntity::Cell, 0, "rank_1_A" );
    GridFieldVectorOp::assign( a_val, rank_1_A );

    GridField<double[3],TEST_EXECSPACE> rank_1_B(
        global_grid, MeshEntity::Cell, 0, "rank_1_B" );
    GridFieldVectorOp::assign( b_val, rank_1_B );

    GridField<double[3],TEST_EXECSPACE> rank_1_C(
        global_grid, MeshEntity::Cell, 0, "rank_1_C" );

    GridField<double[3],TEST_EXECSPACE> rank_1_D(
        global_grid, MeshEntity::Cell, 0, "rank_1_D" );

    // Check the 2 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_1_A,
                               beta,
                               rank_1_B,
                               rank_1_C );
    auto rank_1_C_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_1_C.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    EXPECT_EQ( rank_1_C_mirror(i,j,k,n0),
                               alpha * a_val + beta * b_val );

    // Check the 3 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_1_A,
                               beta,
                               rank_1_B,
                               gamma,
                               rank_1_C,
                               rank_1_D );
    auto rank_1_D_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_1_D.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    EXPECT_EQ( rank_1_D_mirror(i,j,k,n0),
                               (1.0 + gamma) * (alpha * a_val + beta * b_val) );


    // TEST Rank-2 FIELDS
    //-------------------
    GridField<double[3][2],TEST_EXECSPACE> rank_2_A(
        global_grid, MeshEntity::Cell, 0, "rank_2_A" );
    GridFieldVectorOp::assign( a_val, rank_2_A );

    GridField<double[3][2],TEST_EXECSPACE> rank_2_B(
        global_grid, MeshEntity::Cell, 0, "rank_2_B" );
    GridFieldVectorOp::assign( b_val, rank_2_B );

    GridField<double[3][2],TEST_EXECSPACE> rank_2_C(
        global_grid, MeshEntity::Cell, 0, "rank_2_C" );

    GridField<double[3][2],TEST_EXECSPACE> rank_2_D(
        global_grid, MeshEntity::Cell, 0, "rank_2_D" );

    // Check the 2 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_2_A,
                               beta,
                               rank_2_B,
                               rank_2_C );
    auto rank_2_C_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_2_C.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    for ( int n1 = 0; n1 < 2; ++n1 )
                        EXPECT_EQ( rank_2_C_mirror(i,j,k,n0,n1),
                                   alpha * a_val + beta * b_val );

    // Check the 3 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_2_A,
                               beta,
                               rank_2_B,
                               gamma,
                               rank_2_C,
                               rank_2_D );
    auto rank_2_D_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_2_D.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    for ( int n1 = 0; n1 < 2; ++n1 )
                        EXPECT_EQ( rank_2_D_mirror(i,j,k,n0,n1),
                                   (1.0 + gamma) * (alpha * a_val + beta * b_val) );

    // TEST Rank-3 FIELDS
    //-------------------
    GridField<double[3][2][4],TEST_EXECSPACE> rank_3_A(
        global_grid, MeshEntity::Cell, 0, "rank_3_A" );
    GridFieldVectorOp::assign( a_val, rank_3_A );

    GridField<double[3][2][4],TEST_EXECSPACE> rank_3_B(
        global_grid, MeshEntity::Cell, 0, "rank_3_B" );
    GridFieldVectorOp::assign( b_val, rank_3_B );

    GridField<double[3][2][4],TEST_EXECSPACE> rank_3_C(
        global_grid, MeshEntity::Cell, 0, "rank_3_C" );

    GridField<double[3][2][4],TEST_EXECSPACE> rank_3_D(
        global_grid, MeshEntity::Cell, 0, "rank_3_D" );

    // Check the 2 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_3_A,
                               beta,
                               rank_3_B,
                               rank_3_C );
    auto rank_3_C_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_3_C.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    for ( int n1 = 0; n1 < 2; ++n1 )
                        for ( int n2 = 0; n2 < 4; ++n2 )
                            EXPECT_EQ( rank_3_C_mirror(i,j,k,n0,n1,n2),
                                       alpha * a_val + beta * b_val );

    // Check the 3 vector update.
    GridFieldVectorOp::update( alpha,
                               rank_3_A,
                               beta,
                               rank_3_B,
                               gamma,
                               rank_3_C,
                               rank_3_D );
    auto rank_3_D_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), rank_3_D.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                for ( int n0 = 0; n0 < 3; ++n0 )
                    for ( int n1 = 0; n1 < 2; ++n1 )
                        for ( int n2 = 0; n2 < 4; ++n2 )
                            EXPECT_EQ( rank_3_D_mirror(i,j,k,n0,n1,n2),
                                       (1.0 + gamma) * (alpha * a_val + beta * b_val) );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, vector_ops )
{
    vectorOpTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
