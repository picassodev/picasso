#include <Harlow_FacetGeometry.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_InputParser.hpp>
#include <Harlow_ParticleList.hpp>
#include <Harlow_UniformMesh.hpp>
#include <Harlow_ParticleLevelSet.hpp>
#include <Harlow_FieldManager.hpp>

#include <Harlow_ParticleInit.hpp>
#include <Harlow_SiloParticleWriter.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Harlow;

namespace Test
{
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct LocateFunctor
{
    FacetGeometryData<MemorySpace> geom;

    template<class ParticleType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( const double x[3], ParticleType& p ) const
    {
        float xf[3] = {float(x[0]),float(x[1]),float(x[2])};
        for ( int d = 0; d < 3; ++d )
        {
            ParticleAccess::get( p, Field::PhysicalPosition(), d ) = x[d];
            ParticleAccess::get( p, Field::LogicalPosition(), d ) = x[d];
        }
        auto volume_id = FacetGeometryOps::locatePoint(xf,geom);
        ParticleAccess::get( p, Field::Color() ) = volume_id;
        return (volume_id >= 0);
    }
};

//---------------------------------------------------------------------------//
void zalesaksDiskTest()
{
    // Get inputs.
    InputParser parser( "particle_level_set_zalesaks_disk.json", "json" );

    // Get the geometry.
    FacetGeometry<TEST_MEMSPACE> geometry(
        parser.propertyTree(), TEST_EXECSPACE() );

    // Make mesh.
    int minimum_halo_size = 2;
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(),
        geometry.globalBoundingBox(), minimum_halo_size, MPI_COMM_WORLD );

    // Create particles.
    using list_type = ParticleList<UniformMesh<TEST_MEMSPACE>,
                                   Field::PhysicalPosition,
                                   Field::LogicalPosition,
                                   Field::Color>;
    auto particles = createParticleList(
        "particles", mesh,
        ParticleTraits<Field::PhysicalPosition,
        Field::LogicalPosition,
        Field::Color>() );

    // Assign particles a color equal to the volume id in which they are
    // located. The implicit complement is not constructed.
    int ppc = 3;
    LocateFunctor<TEST_MEMSPACE> init_func;
    init_func.geom = geometry.data();
    initializeParticles(
        InitUniform(), TEST_EXECSPACE(), ppc, init_func, *particles );

    // Write the initial particle state.
    double time = 0.0;
    SiloParticleWriter::writeTimeStep(
        mesh->localGrid()->globalGrid(),
        0,
        time,
        particles->slice(Field::PhysicalPosition()),
        particles->slice(Field::Color()) );

    // Build a level set for disk.
    int disk_color = 0;
    ParticleLevelSet<list_type,FieldLocation::Node> level_set(
        parser.propertyTree(), particles, disk_color );
    level_set.updateParticleIndices( TEST_EXECSPACE() );

    // Compute the initial level set.
    level_set.updateSignedDistance( TEST_EXECSPACE() );

    // Write the initial level set.
    Cajita::BovWriter::writeTimeStep( 0, time, *(level_set.getSignedDistance()) );

    // Advect the disk one full rotation.
    int num_particle = particles->size();
    auto xp = particles->slice( Field::PhysicalPosition() );
    auto xl = particles->slice( Field::LogicalPosition() );
    double pi_scale = 4.0 * atan(1.0) / 314;
    int num_step = 628;
    for ( int t = 0; t < num_step; ++t )
    {
        // Move the particles.
        Kokkos::parallel_for(
            "move_particles",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,num_particle),
            KOKKOS_LAMBDA( const int p ){
                double vx = pi_scale * ( 0.5 - xp(p,Dim::J) );
                double vy = pi_scale * ( xp(p,Dim::I) - 0.5 );
                xp(p,Dim::I) += vx;
                xp(p,Dim::J) += vy;
                xl(p,Dim::I) += vx;
                xl(p,Dim::J) += vy;
            });

        // Write the particle state.
        time += 1.0;
        SiloParticleWriter::writeTimeStep(
            mesh->localGrid()->globalGrid(),
            t+1,
            time,
            particles->slice(Field::PhysicalPosition()),
            particles->slice(Field::Color()) );

        // Compute the level set.
        level_set.updateSignedDistance( TEST_EXECSPACE() );

        // Write the level set.
        Cajita::BovWriter::writeTimeStep(
            t+1, time, *(level_set.getSignedDistance()) );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, zalesaks_disk_test )
{
    zalesaksDiskTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
