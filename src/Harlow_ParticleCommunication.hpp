#ifndef HARLOW_PARTICLECOMMUNICATION_HPP
#define HARLOW_PARTICLECOMMUNICATION_HPP

#include <Cajita_GridBlock.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_MpiTraits.hpp>

#include <Harlow_Types.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <mpi.h>

#include <vector>
#include <numeric>
#include <type_traits>
#include <algorithm>

namespace Harlow
{
namespace ParticleCommunication
{
//---------------------------------------------------------------------------//
// Particle destination
//---------------------------------------------------------------------------//
template<class CoordSliceType,
         class NeighborRankView,
         class DestinationRankView>
void computeParticleDestinations(
    const Cajita::GlobalGrid& global_grid,
    const CoordSliceType& coords,
    const NeighborRankView& neighbor_ranks,
    DestinationRankView& destinations )
{
    using execution_space = typename CoordSliceType::execution_space;

    // Get the grid block without a halo.
    const auto& local_block = global_grid.block();

    // Locate the particles in the global grid and get their destination
    // rank. The particle halo should be constructed such that particles will
    // only move to a location in the 26 neighbor halo or stay on this rank.
    auto num_particle_send = coords.size();
    double low_x = local_block.lowCorner( Dim::I );
    double low_y = local_block.lowCorner( Dim::J );
    double low_z = local_block.lowCorner( Dim::K );
    double high_x = low_x +
                    local_block.numEntity( MeshEntity::Cell, Dim::I ) *
                    local_block.cellSize();
    double high_y = low_y +
                    local_block.numEntity( MeshEntity::Cell, Dim::J ) *
                    local_block.cellSize();
    double high_z = low_z +
                    local_block.numEntity( MeshEntity::Cell, Dim::K ) *
                    local_block.cellSize();
    Kokkos::parallel_for(
        "redistribute_locate",
        Kokkos::RangePolicy<execution_space>(0,num_particle_send),
        KOKKOS_LAMBDA( const int p ){
            // Compute the logical index of the neighbor we are sending to.
            int di = 1;
            if ( coords(p,Dim::I) < low_x ) di = 0;
            else if ( coords(p,Dim::I) > high_x ) di = 2;

            int dj = 1;
            if ( coords(p,Dim::J) < low_y ) dj = 0;
            else if ( coords(p,Dim::J) > high_y ) dj = 2;

            int dk = 1;
            if ( coords(p,Dim::K) < low_z ) dk = 0;
            else if ( coords(p,Dim::K) > high_z ) dk = 2;

            // Compute the MPI rank.
            destinations( p ) = neighbor_ranks( di + 3*(dj + 3*dk) );
        });
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Neighbor selection for redistribution.
//---------------------------------------------------------------------------//
// Determine if each of the 26 adjacent blocks in logical space is a rank we
// should send to.
void getNeighbors( const Cajita::GlobalGrid& global_grid,
                   std::vector<int>& topology )
{
    auto grid = global_grid.block();

    auto has_halo =
        [&]( const int dim, const int logical_index ){
            bool halo_check = true;
            if ( -1 == logical_index )
                halo_check = grid.hasHalo(2*dim);
            else if ( 1 == logical_index )
                halo_check = grid.hasHalo(2*dim+1);
            return halo_check;
        };

    int nr = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i, ++nr )
            {
                // If this is an adjacent rank see if we need to send to
                // it. Also always send to ourselves.
                if ( (has_halo(Dim::I,i) &&
                      has_halo(Dim::J,j) &&
                      has_halo(Dim::K,k)) ||
                     (i==0 && j==0 && k==0) )
                {
                    topology[nr] =
                        global_grid.neighborCommRank(i,j,k);
                }
            }
}

//---------------------------------------------------------------------------//
// Periodic coordinate shift for communication across a periodic boundary.
//---------------------------------------------------------------------------//
// When particles cross a periodic boundary their coordinates must be shifted
// to represent a new physical location.
template<class CoordSliceType>
void shiftPeriodicCoordinates( const Cajita::GlobalGrid& global_grid,
                               CoordSliceType& coords )
{
    using execution_space = typename CoordSliceType::execution_space;

    // Get the global grid bounds.
    double global_low_x = global_grid.lowCorner( Dim::I );
    double global_low_y = global_grid.lowCorner( Dim::J );
    double global_low_z = global_grid.lowCorner( Dim::K );
    double global_high_x = global_low_x +
                           global_grid.numEntity( MeshEntity::Cell, Dim::I ) *
                           global_grid.cellSize();
    double global_high_y = global_low_y +
                           global_grid.numEntity( MeshEntity::Cell, Dim::J ) *
                           global_grid.cellSize();
    double global_high_z = global_low_z +
                           global_grid.numEntity( MeshEntity::Cell, Dim::K ) *
                           global_grid.cellSize();
    double global_span_x = global_high_x - global_low_x;
    double global_span_y = global_high_y - global_low_y;
    double global_span_z = global_high_z - global_low_z;

    // Low X boundary
    if ( global_grid.block().onBoundary(DomainBoundary::LowX) &&
         global_grid.block().isPeriodic(Dim::I) )
    {
        Kokkos::parallel_for(
            "redistribute_low_x_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::I) > global_high_x )
                    coords(p,Dim::I) -= global_span_x;
            } );
    }

    // High X boundary
    if ( global_grid.block().onBoundary(DomainBoundary::HighX) &&
         global_grid.block().isPeriodic(Dim::I) )
    {
        Kokkos::parallel_for(
            "redistribute_high_x_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::I) < global_low_x )
                    coords(p,Dim::I) += global_span_x;
            } );
    }

    // Low Y boundary
    if ( global_grid.block().onBoundary(DomainBoundary::LowY) &&
         global_grid.block().isPeriodic(Dim::J) )
    {
        Kokkos::parallel_for(
            "redistribute_low_y_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::J) > global_high_y )
                    coords(p,Dim::J) -= global_span_y;
            } );
    }

    // High Y boundary
    if ( global_grid.block().onBoundary(DomainBoundary::HighY) &&
         global_grid.block().isPeriodic(Dim::J) )
    {
        Kokkos::parallel_for(
            "redistribute_high_y_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::J) < global_low_y )
                    coords(p,Dim::J) += global_span_y;
            } );
    }

    // Low Z boundary
    if ( global_grid.block().onBoundary(DomainBoundary::LowZ) &&
         global_grid.block().isPeriodic(Dim::K) )
    {
        Kokkos::parallel_for(
            "redistribute_low_z_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::K) > global_high_z )
                    coords(p,Dim::K) -= global_span_z;
            } );
    }

    // High Z boundary
    if ( global_grid.block().onBoundary(DomainBoundary::HighZ) &&
         global_grid.block().isPeriodic(Dim::K) )
    {
        Kokkos::parallel_for(
            "redistribute_high_z_periodic_shift",
            Kokkos::RangePolicy<execution_space>(0,coords.size()),
            KOKKOS_LAMBDA( const int p ){
                if ( coords(p,Dim::K) < global_low_z )
                    coords(p,Dim::K) += global_span_z;
            } );
    }
}

//---------------------------------------------------------------------------//
// Particle redistribution
//---------------------------------------------------------------------------//
/*!
  \brief Redistribute particles to new owning ranks based on their location.

  \param global_grid The grid over which to redistribute the particles.

  \param particles The particles to redistribute.

  \param Member index in the AoSoA of the particle coordinates.
 */
template<class ParticleContainer, std::size_t CoordIndex>
void redistribute( const Cajita::GlobalGrid& global_grid,
                   ParticleContainer& particles,
                   std::integral_constant<std::size_t,CoordIndex> )
{
    using device_type = typename ParticleContainer::device_type;

    // Of the 27 potential blocks figure out which are in our topology. Some
    // of the ranks in this list may be invalid. We will update this list
    // after we compute destination ranks so it is unique and valid.
    std::vector<int> topology( 27, -1 );
    getNeighbors( global_grid, topology );

    // Get the coordinates.
    auto coords = Cabana::slice<CoordIndex>( particles );

    // Locate the particles in the global grid and get their destination
    // rank.
    Kokkos::View<int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
        neighbor_ranks( topology.data(), topology.size() );
    auto nr_mirror = Kokkos::create_mirror_view_and_copy(
        device_type(), neighbor_ranks );
    Kokkos::View<int*,device_type> destinations(
        Kokkos::ViewAllocateWithoutInitializing("destinations"),
        particles.size() );
    computeParticleDestinations( global_grid, coords, nr_mirror, destinations );

    // Make the topology a list of unique and valid ranks.
    auto remove_end = std::remove( topology.begin(), topology.end(), -1 );
    std::sort( topology.begin(), remove_end );
    auto unique_end = std::unique( topology.begin(), remove_end );
    topology.resize( std::distance(topology.begin(),unique_end) );

    // Create the Cabana distributor.
    Cabana::Distributor<device_type> distributor( global_grid.cartesianComm(),
                                                  destinations,
                                                  topology );

    // Redistribute the particles.
    Cabana::migrate( distributor, particles );

    // Get a new slice of the coordinates.
    coords = Cabana::slice<CoordIndex>( particles );

    // Shift the particle coordinates for movement across periodic boundaries.
    shiftPeriodicCoordinates( global_grid, coords );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleCommunication
} // end namespace Harlow

#endif // end HARLOW_PARTICLECOMMUNICATION_HPP
