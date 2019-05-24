#include <Harlow_DeformationGradient.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void updateTest()
{
    // Create a particle velocity gradients.
    int num_particle = 10;
    Kokkos::View<double*[4][3],Kokkos::HostSpace> velocity_host(
        "velocity", num_particle );
    for ( int p = 0; p < num_particle; ++p )
    {
        velocity_host(p,1,0) = 3.2*p - 1.3;
        velocity_host(p,1,1) = -12.3*p - 2.22;
        velocity_host(p,1,2) = 8.33*p + 12.3;

        velocity_host(p,2,0) = -9.2*p - 1.9;
        velocity_host(p,2,1) = 1.3*p - 6.22;
        velocity_host(p,2,2) = 3.33*p + 1.3;

        velocity_host(p,3,0) = 4.9*p - 7.6;
        velocity_host(p,3,1) = -2.33*p - 1.5;
        velocity_host(p,3,2) = -6.13*p + 3.2;
    }
    auto velocity = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), velocity_host );

    // Create deformation gradients. Set them to be the identity.
    Kokkos::View<double*[3][3],Kokkos::HostSpace> def_grad_host(
        "def_grad", num_particle );
    Kokkos::deep_copy( def_grad_host, 0.0 );
    for ( int p = 0; p < num_particle; ++p )
    {
        def_grad_host(p,0,0) = 1.0;
        def_grad_host(p,1,1) = 1.0;
        def_grad_host(p,2,2) = 1.0;
    }
    auto def_grad = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), def_grad_host );

    // Update the gradient.
    double dt = 9.9332e-5;
    DeformationGradient::update( dt, velocity, def_grad );

    // Check the resulting deformation_gradient. Because the initial
    // deformation gradient was the identity would should get back the
    // velocity gradient times the timestep.
    Kokkos::deep_copy( def_grad_host, def_grad );
    for ( int p = 0; p < num_particle; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
            {
                if ( i==j )
                    EXPECT_DOUBLE_EQ( def_grad_host(p,i,j),
                                      dt * velocity_host(p,i+1,j) + 1.0 );
                else
                    EXPECT_DOUBLE_EQ( def_grad_host(p,i,j),
                                      dt * velocity_host(p,i+1,j) );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, update_test )
{
    updateTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
