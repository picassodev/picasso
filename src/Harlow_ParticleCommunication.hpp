#ifndef HARLOW_PARTICLECOMMUNICATION_HPP
#define HARLOW_PARTICLECOMMUNICATION_HPP

#include <Cajita.hpp>

#include <Harlow_Types.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

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
    const Cajita::Block& block,
    const CoordSliceType& coords,
    const NeighborRankView& neighbor_ranks,
    DestinationRankView& destinations )
{
    using execution_space = typename CoordSliceType::execution_space;

    // Locate the particles in the global grid and get their destination
    // rank. The particle halo should be constructed such that particles will
    // only move to a location in the 26 neighbor halo or stay on this rank.
    auto num_particle_send = coords.size();
    auto low_corner_i = block.lowCorner(Cajita::Own(),Dim::I);
    auto low_corner_j = block.lowCorner(Cajita::Own(),Dim::J);
    auto low_corner_k = block.lowCorner(Cajita::Own(),Dim::K);
    auto high_corner_i = block.highCorner(Cajita::Own(),Dim::I);
    auto high_corner_j = block.highCorner(Cajita::Own(),Dim::J);
    auto high_corner_k = block.highCorner(Cajita::Own(),Dim::K);
    Kokkos::parallel_for(
        "redistribute_locate",
        Kokkos::RangePolicy<execution_space>(0,num_particle_send),
        KOKKOS_LAMBDA( const int p ){
            // Compute the logical index of the neighbor we are sending to.
            int ni = 1;
            if ( coords(p,Dim::I) < low_corner_i ) ni = 0;
            else if ( coords(p,Dim::I) > high_corner_i ) ni = 2;

            int nj = 1;
            if ( coords(p,Dim::J) < low_corner_j ) nj = 0;
            else if ( coords(p,Dim::J) > high_corner_j ) nj = 2;

            int nk = 1;
            if ( coords(p,Dim::K) < low_corner_k ) nk = 0;
            else if ( coords(p,Dim::K) > high_corner_k ) nk = 2;

            // Compute the MPI rank.
            destinations( p ) = neighbor_ranks( ni + 3*(nj + 3*nk) );
        });
}

//---------------------------------------------------------------------------//
// Periodic coordinate shift for communication across a periodic boundary.
//---------------------------------------------------------------------------//
// When particles cross a periodic boundary their coordinates must be shifted
// to represent a new physical location.
template<class CoordSliceType>
void shiftPeriodicCoordinates( const Cajita::Domain& domain,
                               CoordSliceType& coords )
{
    using execution_space = typename CoordSliceType::execution_space;

    for ( int d = 0; d < 3; ++d )
    {
        if ( domain.isPeriodic(d) )
        {
            double global_low = domain.lowCorner( d );
            double global_high = domain.highCorner( d );
            double global_span = domain.extent( d );

            Kokkos::parallel_for(
                "redistribute_periodic_shift",
                Kokkos::RangePolicy<execution_space>(0,coords.size()),
                KOKKOS_LAMBDA( const int p ){
                    if ( coords(p,d) > global_high )
                        coords(p,d) -= global_span;
                    else if ( coords(p,d) < global_low )
                        coords(p,d) += global_span;
                } );
        }
    }
}

//---------------------------------------------------------------------------//
// Particle redistribution
//---------------------------------------------------------------------------//
/*!
  \brief Redistribute particles to new owning ranks based on their location.

  \param block The block in which the particles are currently located.

  \param particles The particles to redistribute.

  \param Member index in the AoSoA of the particle coordinates.
 */
template<class ParticleContainer, std::size_t CoordIndex>
void redistribute( const Cajita::Block& block,
                   ParticleContainer& particles,
                   std::integral_constant<std::size_t,CoordIndex> )
{
    using device_type = typename ParticleContainer::device_type;

    // Of the 27 potential blocks figure out which are in our topology. Some
    // of the ranks in this list may be invalid. We will update this list
    // after we compute destination ranks so it is unique and valid.
    std::vector<int> topology( 27, -1 );
    int nr = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i, ++nr )
                topology[nr] = block.neighborRank(i,j,k);

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
    computeParticleDestinations( block, coords, nr_mirror, destinations );

    // Make the topology a list of unique and valid ranks.
    auto remove_end = std::remove( topology.begin(), topology.end(), -1 );
    std::sort( topology.begin(), remove_end );
    auto unique_end = std::unique( topology.begin(), remove_end );
    topology.resize( std::distance(topology.begin(),unique_end) );

    // Create the Cabana distributor.
    Cabana::Distributor<device_type> distributor( block.globalGrid().comm(),
                                                  destinations,
                                                  topology );

    // Redistribute the particles.
    Cabana::migrate( distributor, particles );

    // Get a new slice of the coordinates.
    coords = Cabana::slice<CoordIndex>( particles );

    // Shift the particle coordinates for movement across periodic boundaries.
    shiftPeriodicCoordinates( block.globalGrid().domain(), coords );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleCommunication
} // end namespace Harlow

#endif // end HARLOW_PARTICLECOMMUNICATION_HPP
