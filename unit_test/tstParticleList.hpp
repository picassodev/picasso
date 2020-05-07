#include <Harlow_ParticleList.hpp>
#include <Harlow_FieldTypes.hpp>
#include <Harlow_InputParser.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
void sliceTest()
{
    // Get inputs for mesh.
    InputParser parser( "uniform_mesh_test_1.json", "json" );
    Kokkos::Array<double,6> global_box = { -10.0, -10.0, -10.0,
                                           10.0, 10.0, 10.0 };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(),
        global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    using list_type = ParticleList<UniformMesh<TEST_MEMSPACE>,
                                   Field::LogicalPosition,
                                   Field::Mass,
                                   Field::Color,
                                   Field::DeformationGradient>;

    list_type particles( "test_particles", mesh );

    // Resize the aosoa.
    auto& aosoa = particles.aosoa();
    std::size_t num_p = 10;
    aosoa.resize( num_p );
    EXPECT_EQ( particles.aosoa().size(), 10 );

    // Populate fields.
    auto px = particles.slice( Field::LogicalPosition() );
    auto pm = particles.slice( Field::Mass() );
    auto pc = particles.slice( Field::Color() );
    auto pf = particles.slice( Field::DeformationGradient() );

    Cabana::deep_copy( px, 1.23 );
    Cabana::deep_copy( pm, 3.3 );
    Cabana::deep_copy( pc, 5 );
    Cabana::deep_copy( pf, -1.2 );

    // Check the slices.
    EXPECT_EQ( px.label(), "logical_position" );
    EXPECT_EQ( pm.label(), "mass" );
    EXPECT_EQ( pc.label(), "color" );
    EXPECT_EQ( pf.label(), "deformation_gradient" );

    auto aosoa_host = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), aosoa );
    for ( std::size_t p = 0; p < num_p; ++p )
    {
        typename list_type::particle_type particle( aosoa_host.getTuple(p) );

        // Check the deep copy.
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::LogicalPosition(),d), 1.23 );

        EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::Mass()), 3.3 );

        EXPECT_EQ( ParticleAccess::get(particle,Field::Color()), 5 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::DeformationGradient(),i,j), -1.2 );

        // Locally modify.
        for ( int d = 0; d < 3; ++d )
            ParticleAccess::get(particle,Field::LogicalPosition(),d) += 1.0;

        ParticleAccess::get(particle,Field::Mass()) += 1.0;

        ParticleAccess::get(particle,Field::Color()) += 1;

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                ParticleAccess::get(particle,Field::DeformationGradient(),i,j) += 1.0;

        // Check the modification.
        for ( int d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::LogicalPosition(),d), 2.23 );

        EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::Mass()), 4.3 );

        EXPECT_EQ( ParticleAccess::get(particle,Field::Color()), 6 );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_DOUBLE_EQ( ParticleAccess::get(particle,Field::DeformationGradient(),i,j), -0.2 );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, slice_test )
{
    sliceTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
