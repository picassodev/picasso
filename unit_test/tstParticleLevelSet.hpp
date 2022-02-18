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

#include <Picasso_FacetGeometry.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleLevelSet.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Picasso_ParticleInit.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
template <class MemorySpace>
struct LocateFunctor
{
    FacetGeometryData<MemorySpace> geom;

    template <class ParticleType>
    KOKKOS_INLINE_FUNCTION bool operator()( const double x[3], const double,
                                            const int, ParticleType& p ) const
    {
        float xf[3] = { float( x[0] ), float( x[1] ), float( x[2] ) };
        for ( int d = 0; d < 3; ++d )
        {
            get( p, Field::PhysicalPosition<3>(), d ) = x[d];
            get( p, Field::LogicalPosition<3>(), d ) = x[d];
        }
        auto volume_id = FacetGeometryOps::locatePoint( xf, geom );
        get( p, Field::Color() ) = volume_id;
        return ( volume_id >= 0 );
    }
};

//---------------------------------------------------------------------------//
void zalesaksTest( const std::string& filename )
{
    // Get the communicator rank.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Get inputs.
    InputParser parser( filename, "json" );

    // Get the geometry.
    FacetGeometry<TEST_MEMSPACE> geometry( parser.propertyTree(),
                                           TEST_EXECSPACE() );

    // Make mesh.
    int minimum_halo_size = 4;
    auto mesh = std::make_shared<UniformMesh<TEST_MEMSPACE>>(
        parser.propertyTree(), geometry.globalBoundingBox(), minimum_halo_size,
        MPI_COMM_WORLD );

    // Create particles.
    auto particles = createParticleList(
        "particles", mesh,
        ParticleTraits<Field::PhysicalPosition<3>, Field::LogicalPosition<3>,
                       Field::Color, Field::CommRank>() );

    // Assign particles a color equal to the volume id in which they are
    // located. The implicit complement is not constructed.
    int ppc = 3;
    LocateFunctor<TEST_MEMSPACE> init_func;
    init_func.geom = geometry.data();
    initializeParticles( InitUniform(), TEST_EXECSPACE(), ppc, init_func,
                         *particles );

    // Write the initial particle state.
    double time = 0.0;
#ifdef Cabana_ENABLE_SILO
    Cajita::Experimental::SiloParticleOutput::writeTimeStep(
        mesh->localGrid()->globalGrid(), 0, time,
        particles->slice( Field::PhysicalPosition<3>() ),
        particles->slice( Field::Color() ),
        particles->slice( Field::CommRank() ) );
#endif

    // Build a level set for disk.
    int disk_color = 0;
    auto level_set = createParticleLevelSet<FieldLocation::Node>(
        parser.propertyTree(), mesh, disk_color );
    level_set->updateParticleColors( TEST_EXECSPACE(),
                                     particles->slice( Field::Color() ) );

    // Compute the initial level set.
    level_set->estimateSignedDistance(
        TEST_EXECSPACE(), particles->slice( Field::LogicalPosition<3>() ) );
    level_set->levelSet()->redistance( TEST_EXECSPACE() );

    // Write the initial level set.
    Cajita::Experimental::BovWriter::writeTimeStep(
        0, time, *( level_set->levelSet()->getDistanceEstimate() ) );
    Cajita::Experimental::BovWriter::writeTimeStep(
        0, time, *( level_set->levelSet()->getSignedDistance() ) );

    // Advect the disk one full rotation.
    double pi = 4.0 * atan( 1.0 );
    int num_step = 1; // change to 628 to go one revolution.
    double delta_phi = 2.0 * pi / num_step;
    for ( int t = 0; t < num_step; ++t )
    {
        // Get slices.
        auto xp = particles->slice( Field::PhysicalPosition<3>() );
        auto xl = particles->slice( Field::LogicalPosition<3>() );
        auto xr = particles->slice( Field::CommRank() );

        // Move the particles around the circle.
        Kokkos::parallel_for(
            "move_particles",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, particles->size() ),
            KOKKOS_LAMBDA( const int p ) {
                // Get the particle location relative to the origin of
                // rotation.
                double x = xp( p, Dim::I ) - 0.5;
                double y = xp( p, Dim::J ) - 0.5;

                // Compute the radius of the circle on which the particle is
                // rotating.
                double r = sqrt( x * x + y * y );

                // Compute the angle relative to the origin.
                double phi = ( y >= 0.0 ) ? acos( x / r ) : -acos( x / r );

                // Increment the angle.
                phi += delta_phi;

                // Compute new particle location.
                x = r * cos( phi ) + 0.5;
                y = r * sin( phi ) + 0.5;

                // Update.
                xp( p, Dim::I ) = x;
                xp( p, Dim::J ) = y;
                xl( p, Dim::I ) = x;
                xl( p, Dim::J ) = y;
                xr( p ) = comm_rank;
            } );

        // Move particles to new ranks if needed if needed.
        bool did_redistribute = particles->redistribute();

        // If they went to new ranks, update the list of colors which write to
        // the level set.
        if ( did_redistribute )
            level_set->updateParticleColors(
                TEST_EXECSPACE(), particles->slice( Field::Color() ) );

        // Write the particle state.
        time += 1.0;
#ifdef Cabana_ENABLE_SILO
        Cajita::Experimental::SiloParticleOutput::writeTimeStep(
            mesh->localGrid()->globalGrid(), t + 1, time,
            particles->slice( Field::PhysicalPosition<3>() ),
            particles->slice( Field::Color() ),
            particles->slice( Field::CommRank() ) );
#endif

        // Compute the level set.
        level_set->estimateSignedDistance(
            TEST_EXECSPACE(), particles->slice( Field::LogicalPosition<3>() ) );
        level_set->levelSet()->redistance( TEST_EXECSPACE() );

        // Write the level set.
        Cajita::Experimental::BovWriter::writeTimeStep(
            t + 1, time, *( level_set->levelSet()->getDistanceEstimate() ) );
        Cajita::Experimental::BovWriter::writeTimeStep(
            t + 1, time, *( level_set->levelSet()->getSignedDistance() ) );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// TEST( TEST_CATEGORY, zalesaks_disk_test )
// {
//     zalesaksTest( "particle_level_set_zalesaks_disk.json" );
// }

// TEST( TEST_CATEGORY, zalesaks_sphere_test )
// {
//     zalesaksTest( "particle_level_set_zalesaks_sphere.json" );
// }

//---------------------------------------------------------------------------//

} // end namespace Test
