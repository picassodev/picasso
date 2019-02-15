#include <Harlow_DenseLinearAlgebra.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
// Fixture
class harlow_dense_linear_algebra : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};

//---------------------------------------------------------------------------//
void svdTest()
{
    double a[3][3] ={{1.0,2.0,3.0}, {4.0,5.0,0.0}, {4.0,2.0,5.0}};
    double trans_a[3][3];
   
    // A = a^T * a
    double A[3][3];
    DenseLinearAlgebra::transpose(a, trans_a);
    DenseLinearAlgebra::multiply_AB( trans_a, a, A);

    // eigenvalue S, eignevector matrix X
    double eigen_value[3];
    double X[3][3];

    DenseLinearAlgebra::eigen(A, eigen_value, X);

    EXPECT_FLOAT_EQ( 79.742519722959429, eigen_value[0]);
    EXPECT_FLOAT_EQ( 18.493780325096871, eigen_value[1]);
    EXPECT_FLOAT_EQ(  1.763699951943707, eigen_value[2]);

    EXPECT_FLOAT_EQ(  0.627750144535748, X[0][0]);
    EXPECT_FLOAT_EQ(  0.580440082644498, X[1][0]);
    EXPECT_FLOAT_EQ(  0.518670479683388, X[2][0]);

    EXPECT_FLOAT_EQ( -0.174111197014272, X[0][1]);
    EXPECT_FLOAT_EQ( -0.544733973795557, X[1][1]);
    EXPECT_FLOAT_EQ(  0.820335412418090, X[2][1]);

    EXPECT_FLOAT_EQ( -0.758692986068544, X[0][2]);
    EXPECT_FLOAT_EQ(  0.605272011786890, X[1][2]);
    EXPECT_FLOAT_EQ(  0.240895713199397, X[2][2]);

    // SVD by U S V
    double S[3][3];
    double U[3][3];
    double V[3][3];

    DenseLinearAlgebra::svd(A, U, S, V);
    
    EXPECT_FLOAT_EQ( 0.627750144535749 , U[0][0]);
    EXPECT_FLOAT_EQ( 0.580440082644498 , U[1][0]);
    EXPECT_FLOAT_EQ( 0.518670479683388 , U[2][0]);

    EXPECT_FLOAT_EQ( -0.174111197014272 , U[0][1]);
    EXPECT_FLOAT_EQ( -0.544733973795557 , U[1][1]);
    EXPECT_FLOAT_EQ(  0.820335412418090 , U[2][1]);
  
    EXPECT_FLOAT_EQ( -0.758692986068544 , U[0][2]);
    EXPECT_FLOAT_EQ(  0.605272011786890 , U[1][2]);
    EXPECT_FLOAT_EQ(  0.240895713199397 , U[2][2]);
     
    EXPECT_FLOAT_EQ( 0.627750144535748 , V[0][0]);
    EXPECT_FLOAT_EQ( 0.580440082644498 , V[1][0]);
    EXPECT_FLOAT_EQ( 0.518670479683387 , V[2][0]);

    EXPECT_FLOAT_EQ( -0.174111197014272 , V[0][1]);
    EXPECT_FLOAT_EQ( -0.544733973795557 , V[1][1]);
    EXPECT_FLOAT_EQ(  0.820335412418090 , V[2][1]);
  
    EXPECT_FLOAT_EQ( -0.758692986068544 , V[0][2]);
    EXPECT_FLOAT_EQ(  0.605272011786891 , V[1][2]);
    EXPECT_FLOAT_EQ(  0.240895713199397 , V[2][2]);
 
    
}
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( harlow_dense_linear_algebra, svd_test )
{
    svdTest();
}

////---------------------------------------------------------------------------//
//TEST_F( harlow_grid_block, assign_test )
//{
//    assignTest();
//}
//
////---------------------------------------------------------------------------//
//TEST_F( harlow_grid_block, periodic_test )
//{
//    periodicTest();
//}

//---------------------------------------------------------------------------//

} // end namespace Test
