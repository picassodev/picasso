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
    // Get the communicator rank.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Get inputs.
    InputParser parser( "particle_level_set_zalesaks_disk.json", "json" );

    // Get the geometry.
    FacetGeometry<TEST_MEMSPACE> geometry(
        parser.propertyTree(), TEST_EXECSPACE() );

    // Make mesh.
    int minimum_halo_size = 4;
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(),
        geometry.globalBoundingBox(), minimum_halo_size, MPI_COMM_WORLD );

    // Create particles.
    using list_type = ParticleList<UniformMesh<TEST_MEMSPACE>,
                                   Field::PhysicalPosition,
                                   Field::LogicalPosition,
                                   Field::Color,
                                   Field::PartId>;
    auto particles = createParticleList(
        "particles", mesh,
        ParticleTraits<Field::PhysicalPosition,
        Field::LogicalPosition,
        Field::Color,
        Field::PartId>() );

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
        particles->slice(Field::Color()),
        particles->slice(Field::PartId()) );

    // Build a level set for disk.
    int disk_color = 0;
    ParticleLevelSet<list_type,FieldLocation::Node> level_set(
        parser.propertyTree(), particles, disk_color );
    level_set.updateParticleIndices( TEST_EXECSPACE() );

    // Compute the initial level set.
    level_set.updateSignedDistance( TEST_EXECSPACE() );

    // Write the initial level set.
    Cajita::BovWriter::writeTimeStep(
        0, time, *(level_set.getDistanceEstimate()) );
    Cajita::BovWriter::writeTimeStep(
        0, time, *(level_set.getSignedDistance()) );

    // Advect the disk one full rotation.
    double pi = 4.0 * atan(1.0);
    int num_step = 628;
    double delta_phi = 2.0 * pi / num_step;
    for ( int t = 0; t < num_step; ++t )
    {
        // Get slices.
        auto xp = particles->slice( Field::PhysicalPosition() );
        auto xl = particles->slice( Field::LogicalPosition() );
        auto xr = particles->slice( Field::PartId() );

        // Move the particles around the circle.
        Kokkos::parallel_for(
            "move_particles",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,particles->size()),
            KOKKOS_LAMBDA( const int p ){
                // Get the particle location relative to the origin of
                // rotation.
                double x = xp(p,Dim::I) - 0.5;
                double y = xp(p,Dim::J) - 0.5;

                // Compute the radius of the circle on which the particle is
                // rotating.
                double r = sqrt( x*x + y*y );

                // Compute the angle relative to the origin.
                double phi = ( y >= 0.0 ) ? acos( x / r ) : -acos( x / r );

                // Increment the angle.
                phi += delta_phi;

                // Compute new particle location.
                x = r * cos(phi) + 0.5;
                y = r * sin(phi) + 0.5;

                // Update.
                xp(p,Dim::I) = x;
                xp(p,Dim::J) = y;
                xl(p,Dim::I) = x;
                xl(p,Dim::J) = y;
                xr(p) = comm_rank;
            });

        // Move particles to new ranks if needed if needed.
        bool did_redistribute = particles->redistribute();

        // If they went to new ranks, update the list of colors which write to
        // the level set.
        if ( did_redistribute )
            level_set.updateParticleIndices( TEST_EXECSPACE() );

        // Write the particle state.
        time += 1.0;
        SiloParticleWriter::writeTimeStep(
            mesh->localGrid()->globalGrid(),
            t+1,
            time,
            particles->slice(Field::PhysicalPosition()),
            particles->slice(Field::Color()),
            particles->slice(Field::PartId()) );

        // Compute the level set.
        level_set.updateSignedDistance( TEST_EXECSPACE() );

        // Write the level set.
        Cajita::BovWriter::writeTimeStep(
            t+1, time, *(level_set.getDistanceEstimate()) );
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
