/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Picasso_DenseLinearAlgebra.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void svdTest()
{

    // Create multiple particle deformation tensor
    int num_particle = 117;

    double a[3][3] ={{1.0,2.0,3.0}, {4.0,5.0,0.0}, {4.0,2.0,5.0}};
    Kokkos::View<double*[3][3], Kokkos::HostSpace> a_host("deformation_gradient", num_particle);

    for(int p=0; p<num_particle; p++)
    {
       for(int i=0; i<3; i++)
       {
          for(int j=0; j<3; j++)
             a_host(p,i,j) = a[i][j];
        }
     }

    // Creat mirror view in device
    auto a_device = Kokkos::create_mirror_view_and_copy(TEST_MEMSPACE(), a_host);

    // Creat eigen_value, eigen_vector and U,V in device
    Kokkos::View<double*[3], TEST_MEMSPACE>    eigen_value_device("eigen_value_device", num_particle);
    Kokkos::View<double*[3][3], TEST_MEMSPACE> eigen_vector_device("eigen_vector_device", num_particle);
    Kokkos::View<double*[3][3], TEST_MEMSPACE> U_device("U_device", num_particle);
    Kokkos::View<double*[3][3], TEST_MEMSPACE> V_device("V_device", num_particle);

    auto eigen_svd_lambda = KOKKOS_LAMBDA(const int p){

                        double sliced_a[3][3];
                        double trans_sliced_a[3][3];
                        double A[3][3];
                        double eigen_value[3];
                        double X[3][3];

                        for(int i=0; i<3; i++)
                        {
                           for(int j=0; j<3; j++)
                              sliced_a[i][j] = a_device(p,i,j);
                        }

                        // A= a^T * a
                        DenseLinearAlgebra::transpose(sliced_a, trans_sliced_a);
                        DenseLinearAlgebra::matMatMultiply( trans_sliced_a, sliced_a, A);

                        // eigenvalue,  eignevector
                        DenseLinearAlgebra::eigen(A, eigen_value, X);


                        for(int i=0; i<3; i++)
                        {
                           eigen_value_device(p,i) = eigen_value[i];

                           for(int j=0; j<3; j++)
                              eigen_vector_device(p,i,j)  = X[i][j];
                        }


                        // SVD by U S V
                        double S[3];
                        double U[3][3];
                        double V[3][3];

                        DenseLinearAlgebra::svd(A, U, S, V);

                        for(int i=0; i<3; i++)
                        {
                           for(int j=0; j<3; j++)
                           {
                              U_device(p,i,j)  = U[i][j];
                              V_device(p,i,j)  = V[i][j];
                           }
                        }};

    Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(0, num_particle), eigen_svd_lambda);

    // Back to Host
    auto eigen_value_host  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigen_value_device);
    auto eigen_vector_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigen_vector_device);

    auto U_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U_device);
    auto V_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), V_device);


    // compare to  Matlab resulta
    for(int p=0; p<num_particle; p++)
    {
       EXPECT_FLOAT_EQ( 79.742519722959429, eigen_value_host(p,0) );
       EXPECT_FLOAT_EQ( 18.493780325096871, eigen_value_host(p,1) );
       EXPECT_FLOAT_EQ(  1.763699951943707, eigen_value_host(p,2) );

       EXPECT_FLOAT_EQ(  0.627750144535748, eigen_vector_host(p,0,0) );
       EXPECT_FLOAT_EQ(  0.580440082644498, eigen_vector_host(p,1,0) );
       EXPECT_FLOAT_EQ(  0.518670479683388, eigen_vector_host(p,2,0) );

       EXPECT_FLOAT_EQ( -0.174111197014272, eigen_vector_host(p,0,1) );
       EXPECT_FLOAT_EQ( -0.544733973795557, eigen_vector_host(p,1,1) );
       EXPECT_FLOAT_EQ(  0.820335412418090, eigen_vector_host(p,2,1) );

       EXPECT_FLOAT_EQ( -0.758692986068544, eigen_vector_host(p,0,2) );
       EXPECT_FLOAT_EQ(  0.605272011786890, eigen_vector_host(p,1,2) );
       EXPECT_FLOAT_EQ(  0.240895713199397, eigen_vector_host(p,2,2) );
    }

    // compare to  Matlab result
    for(int p=0; p<num_particle; p++)
    {
       EXPECT_FLOAT_EQ( 0.627750144535749 , U_host(p,0,0) );
       EXPECT_FLOAT_EQ( 0.580440082644498 , U_host(p,1,0) );
       EXPECT_FLOAT_EQ( 0.518670479683388 , U_host(p,2,0) );

       EXPECT_FLOAT_EQ( -0.174111197014272 , U_host(p,0,1) );
       EXPECT_FLOAT_EQ( -0.544733973795557 , U_host(p,1,1) );
       EXPECT_FLOAT_EQ(  0.820335412418090 , U_host(p,2,1) );

       EXPECT_FLOAT_EQ( -0.758692986068544 , U_host(p,0,2) );
       EXPECT_FLOAT_EQ(  0.605272011786890 , U_host(p,1,2) );
       EXPECT_FLOAT_EQ(  0.240895713199397 , U_host(p,2,2) );

       EXPECT_FLOAT_EQ( 0.627750144535748 , V_host(p,0,0) );
       EXPECT_FLOAT_EQ( 0.580440082644498 , V_host(p,1,0) );
       EXPECT_FLOAT_EQ( 0.518670479683387 , V_host(p,2,0) );

       EXPECT_FLOAT_EQ( -0.174111197014272 , V_host(p,0,1) );
       EXPECT_FLOAT_EQ( -0.544733973795557 , V_host(p,1,1) );
       EXPECT_FLOAT_EQ(  0.820335412418090 , V_host(p,2,1) );

       EXPECT_FLOAT_EQ( -0.758692986068544 , V_host(p,0,2) );
       EXPECT_FLOAT_EQ(  0.605272011786891 , V_host(p,1,2) );
       EXPECT_FLOAT_EQ(  0.240895713199397 , V_host(p,2,2) );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, svd_test )
{
    svdTest();
}
//---------------------------------------------------------------------------//

} // end namespace Test
