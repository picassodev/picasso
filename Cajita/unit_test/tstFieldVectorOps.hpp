#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_Field.hpp>
#include <Cajita_FieldVectorOps.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void vectorOpTest()
{
    // Let MPI compute the partitioning for this test.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    UniformDimPartitioner partitioner;

    // Create the global grid.
    double cell_size = 0.23;
    std::vector<int> global_num_cell = { 23, 17, 21 };
    std::vector<bool> is_dim_periodic = {false,false,false};
    std::vector<double> global_low_corner = { 1.2, 3.3, -2.8 };
    std::vector<double> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_grid = std::make_shared<GlobalGrid>( MPI_COMM_WORLD,
                                                     partitioner,
                                                     is_dim_periodic,
                                                     global_low_corner,
                                                     global_high_corner,
                                                     cell_size );

    int i_begin = global_grid->block().localEntityBegin(MeshEntity::Cell,Dim::I);
    int i_end = global_grid->block().localEntityEnd(MeshEntity::Cell,Dim::I);
    int j_begin = global_grid->block().localEntityBegin(MeshEntity::Cell,Dim::J);
    int j_end = global_grid->block().localEntityEnd(MeshEntity::Cell,Dim::J);
    int k_begin = global_grid->block().localEntityBegin(MeshEntity::Cell,Dim::K);
    int k_end = global_grid->block().localEntityEnd(MeshEntity::Cell,Dim::K);


    // TEST SCALAR FIELDS
    //-------------------
    Field<double,TEST_EXECSPACE> scalar_A(
        global_grid, 1, MeshEntity::Cell, 0, "scalar_A" );

    // Assign some data.
    double a_val = -3.23;
    FieldVectorOps::assign( a_val, scalar_A );

    // Check the assignment.
    auto scalar_A_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_A.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_DOUBLE_EQ( scalar_A_mirror(i,j,k,0), a_val );

    // Check infinity norm.
    double expected_norm_inf = 1347.5;
    if ( 0 == comm_rank )
        scalar_A_mirror( 3, 2, 1, 0 ) = expected_norm_inf;
    Kokkos::deep_copy( scalar_A.data(), scalar_A_mirror );
    auto norm_inf = FieldVectorOps::normInf( scalar_A );
    EXPECT_DOUBLE_EQ( norm_inf, expected_norm_inf );

    // Reset the vector.
    FieldVectorOps::assign( a_val, scalar_A );

    // Check the 1-norm.
    int total_num_cell = global_grid->numEntity(MeshEntity::Cell,Dim::I) *
                         global_grid->numEntity(MeshEntity::Cell,Dim::J) *
                         global_grid->numEntity(MeshEntity::Cell,Dim::K);
    auto norm_1 = FieldVectorOps::norm1( scalar_A );
    double expected_norm_1 = total_num_cell * std::abs(a_val);
    EXPECT_FLOAT_EQ( norm_1, expected_norm_1 );

    // Check the 2-norm.
    auto norm_2 = FieldVectorOps::norm2( scalar_A );
    double expected_norm_2 = std::sqrt( total_num_cell * a_val * a_val );
    EXPECT_FLOAT_EQ( norm_2, expected_norm_2 );

    // Check the dot product.
    double b_val = 4.33;
    Field<double,TEST_EXECSPACE> scalar_B(
        global_grid, 1, MeshEntity::Cell, 0, "scalar_B" );
    FieldVectorOps::assign( b_val, scalar_B );
    auto dot_product = FieldVectorOps::dot( scalar_A, scalar_B );
    double expected_dot = total_num_cell * a_val * b_val;
    EXPECT_FLOAT_EQ( dot_product, expected_dot );

    // Check the scaling.
    double alpha = -8.99;
    FieldVectorOps::scale( alpha, scalar_A );
    Kokkos::deep_copy( scalar_A_mirror, scalar_A.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_DOUBLE_EQ( scalar_A_mirror(i,j,k,0), a_val * alpha );

    double beta = 12.2;
    FieldVectorOps::scale( beta, scalar_B );
    auto scalar_B_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_B.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_DOUBLE_EQ( scalar_B_mirror(i,j,k,0), b_val * beta );

    // Check the 2 vector update.
    Field<double,TEST_EXECSPACE> scalar_C(
        global_grid, 1, MeshEntity::Cell, 0, "scalar_C" );
    FieldVectorOps::update( 1.0 / alpha,
                            scalar_A,
                            1.0 / beta,
                            scalar_B,
                            scalar_C );
    auto scalar_C_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_C.data() );
    for ( int i = i_begin; i < i_end; ++i )
        for ( int j = j_begin; j < j_end; ++j )
            for ( int k = k_begin; k < k_end; ++k )
                EXPECT_DOUBLE_EQ( scalar_C_mirror(i,j,k,0), a_val + b_val );

    // Check the 3 vector update.
    Field<double,TEST_EXECSPACE> scalar_D(
        global_grid, 1, MeshEntity::Cell, 0, "scalar_D" );
    double gamma = -12.1;
    FieldVectorOps::update( 1.0 / alpha,
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
                EXPECT_DOUBLE_EQ( scalar_D_mirror(i,j,k,0),
                                  (1.0 + gamma ) * (a_val + b_val) );


    // TEST Rank-1 FIELDS
    //-------------------
    Field<double,TEST_EXECSPACE> rank_1_A(
        global_grid, 3, MeshEntity::Cell, 0, "rank_1_A" );
    FieldVectorOps::assign( a_val, rank_1_A );

    Field<double,TEST_EXECSPACE> rank_1_B(
        global_grid, 3, MeshEntity::Cell, 0, "rank_1_B" );
    FieldVectorOps::assign( b_val, rank_1_B );

    Field<double,TEST_EXECSPACE> rank_1_C(
        global_grid, 3, MeshEntity::Cell, 0, "rank_1_C" );

    Field<double,TEST_EXECSPACE> rank_1_D(
        global_grid, 3, MeshEntity::Cell, 0, "rank_1_D" );

    // Check the 2 vector update.
    FieldVectorOps::update( alpha,
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
                    EXPECT_DOUBLE_EQ( rank_1_C_mirror(i,j,k,n0),
                                      alpha * a_val + beta * b_val );

    // Check the 3 vector update.
    FieldVectorOps::update( alpha,
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
                    EXPECT_DOUBLE_EQ( rank_1_D_mirror(i,j,k,n0),
                                      (1.0 + gamma) * (alpha * a_val + beta * b_val) );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, vector_ops )
{
    vectorOpTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
