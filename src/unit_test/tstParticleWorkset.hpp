#include <Cajita_GridBlock.hpp>
#include <Cajita_Types.hpp>

#include <Harlow_Types.hpp>
#include <Harlow_ParticleWorkset.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void worksetTest()
{
    // Make a cartesian grid.
    std::vector<int> num_cell = { 13, 21, 10 };
    std::vector<double> low_corner = { -1.1, 3.3, -5.3 };
    std::vector<bool> boundary_location = { false, false, false, false, false, false};
    std::vector<bool> periodic = {false,false,false};
    double cell_size = 0.53;
    int halo_width = 4;
    Cajita::GridBlock grid( low_corner, num_cell, boundary_location,
                            periodic, cell_size, halo_width );

    // Calculate the low corners of the node primal grid. This includes the halo.
    std::vector<double> node_low_corner =
        { low_corner[Dim::I] - halo_width * cell_size,
          low_corner[Dim::J] - halo_width * cell_size,
          low_corner[Dim::K] - halo_width * cell_size };

    // Put a particle in the lower left center of each local cell.
    int num_particle = num_cell[0] * num_cell[1] * num_cell[2];
    using MemberTypes = Cabana::MemberTypes<double[3]>;
    Cabana::AoSoA<MemberTypes,TEST_DEVICE> particles( "particles", num_particle );
    auto particles_mirror = Cabana::Experimental::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particles );
    auto position_mirror = Cabana::slice<0>( particles_mirror );
    int pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
            {
                position_mirror( pid, Dim::I ) =
                    low_corner[Dim::I] + (i+0.25) * cell_size;
                position_mirror( pid, Dim::J ) =
                    low_corner[Dim::J] + (j+0.25) * cell_size;
                position_mirror( pid, Dim::K ) =
                    low_corner[Dim::K] + (k+0.25) * cell_size;
            }
    Cabana::deep_copy( particles, particles_mirror );
    auto position = Cabana::slice<0>( particles );

    // Create a workset.
    auto workset =
        createParticleWorkset<FunctionOrder::Quadratic,TEST_DEVICE>(
            grid, num_particle );

    // Update the workset.
    double dt = 33.2;
    updateParticleWorkset<FunctionOrder::Quadratic>(
        position, *workset, dt );

    // Check the sizes.
    EXPECT_EQ( workset->_num_particle, num_particle );
    EXPECT_EQ( workset->_ns, 3 );
    EXPECT_EQ( workset->_dx, cell_size );
    EXPECT_EQ( workset->_rdx, 1.0/cell_size );
    EXPECT_EQ( workset->_low_x, node_low_corner[Dim::I] );
    EXPECT_EQ( workset->_low_y, node_low_corner[Dim::J] );
    EXPECT_EQ( workset->_low_z, node_low_corner[Dim::K] );
    EXPECT_EQ( workset->_dt, dt );

    // Check the allocation.
    EXPECT_EQ( workset->_logical.extent(0), num_particle );
    EXPECT_EQ( workset->_nodes.extent(0), num_particle );
    EXPECT_EQ( workset->_nodes.extent(1), 3 );
    EXPECT_EQ( workset->_distance.extent(0), num_particle );
    EXPECT_EQ( workset->_distance.extent(1), 3 );
    EXPECT_EQ( workset->_basis.extent(0), num_particle );
    EXPECT_EQ( workset->_basis.extent(1), 3 );
    EXPECT_EQ( workset->_basis_grad.extent(0), num_particle );
    EXPECT_EQ( workset->_basis_grad.extent(1), 3 );
    EXPECT_EQ( workset->_basis_grad.extent(2), 3 );
    EXPECT_EQ( workset->_basis_grad.extent(3), 3 );

    // Check the data.
    auto workset_logical = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workset->_logical );
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
            {
                EXPECT_DOUBLE_EQ( workset_logical(pid,Dim::I),
                                  i + 0.75 + halo_width );
                EXPECT_DOUBLE_EQ( workset_logical(pid,Dim::J),
                                  j + 0.75 + halo_width );
                EXPECT_DOUBLE_EQ( workset_logical(pid,Dim::K),
                                  k + 0.75 + halo_width );
            }

    auto workset_nodes = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workset->_nodes );
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
                for ( int n = 0; n < 3; ++n )
                {
                    EXPECT_EQ( workset_nodes(pid,n,Dim::I), i + n + halo_width - 1);
                    EXPECT_EQ( workset_nodes(pid,n,Dim::J), j + n + halo_width - 1);
                    EXPECT_EQ( workset_nodes(pid,n,Dim::K), k + n + halo_width - 1);
                }

    auto workset_distance = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workset->_distance );
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
                for ( int n = 0; n < 3; ++n )
                    for ( int d = 0; d < 3; ++ d )
                        EXPECT_FLOAT_EQ( workset_distance(pid,n,d),
                                         cell_size*(n-1.25) );

    auto workset_basis = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workset->_basis );
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
                for ( int d = 0; d < 3; ++d )
                {
                    double xn = workset_logical(pid,d) -
                                int(workset_logical(pid,d)) + 0.5;
                    EXPECT_FLOAT_EQ( workset_basis(pid,0,d),
                                     0.5 * xn * xn - 1.5 * xn + 9.0 / 8.0 );
                    xn -= 1.0;
                    EXPECT_FLOAT_EQ( workset_basis(pid,1,d),
                                     -xn * xn + 0.75 );
                    xn -= 1.0;
                    EXPECT_FLOAT_EQ( workset_basis(pid,2,d),
                                     0.5 * xn * xn + 1.5 * xn + 9.0 / 8.0 );
                }

    auto workset_basis_grad = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), workset->_basis_grad );
    pid = 0;
    for ( int i = 0; i < num_cell[Dim::I]; ++i )
        for ( int j = 0; j < num_cell[Dim::J]; ++j )
            for ( int k = 0; k < num_cell[Dim::K]; ++k, ++pid )
                for ( int ni = 0; ni < 3; ++ni )
                    for ( int nj = 0; nj < 3; ++nj )
                        for ( int nk = 0; nk < 3; ++nk )
                        {
                            double weight = workset_basis(pid,ni,Dim::I) *
                                            workset_basis(pid,nj,Dim::J) *
                                            workset_basis(pid,nk,Dim::K) * 4.0 *
                                            workset->_rdx * workset->_rdx;
                            EXPECT_FLOAT_EQ( workset_basis_grad(pid,ni,nj,nk,Dim::I),
                                             weight * workset_distance(pid,ni,Dim::I) );
                            EXPECT_FLOAT_EQ( workset_basis_grad(pid,ni,nj,nk,Dim::J),
                                             weight * workset_distance(pid,nj,Dim::J) );
                            EXPECT_FLOAT_EQ( workset_basis_grad(pid,ni,nj,nk,Dim::K),
                                             weight * workset_distance(pid,nk,Dim::K) );
                        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, workset_test )
{
    worksetTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
