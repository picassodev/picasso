#include <Harlow_ParticleAdvection.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void moveTest()
{
    // Create a particle velocites.
    int num_particle = 132;
    Kokkos::View<double*[1][3],Kokkos::HostSpace> velocity_host(
        "velocity", num_particle );
    for ( int p = 0; p < num_particle; ++p )
    {
        velocity_host(p,0,0) = 3.2*p - 1.3;
        velocity_host(p,0,1) = -12.3*p - 2.22;
        velocity_host(p,0,2) = 8.33*p + 12.3;
    }
    auto velocity = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), velocity_host );

    // Create particle positions
    Kokkos::View<double*[3],TEST_MEMSPACE> position(
        "position", num_particle );
    Kokkos::deep_copy( position, 3.2 );

    // Move the particles.
    double dt = 9.9332;
    ParticleAdvection::move( dt, velocity, position );

    // Check the resulting positions.
    auto position_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), position );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_DOUBLE_EQ( position_host(p,0), dt * velocity_host(p,0,0) + 3.2 );
        EXPECT_DOUBLE_EQ( position_host(p,1), dt * velocity_host(p,0,1) + 3.2 );
        EXPECT_DOUBLE_EQ( position_host(p,2), dt * velocity_host(p,0,2) + 3.2 );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, move_test )
{
    moveTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
