#ifndef HARLOW_PARTICLECOMMUNICATION_HPP
#define HARLOW_PARTICLECOMMUNICATION_HPP

#include <Harlow_GlobalGrid.hpp>
#include <Harlow_MpiTraits.hpp>
#include <Harlow_ParticleFieldOps.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <mpi.h>

#include <vector>
#include <numeric>
#include <type_traits>

namespace Harlow
{
namespace ParticleCommunication
{
//---------------------------------------------------------------------------//
// Particle byte size.
//---------------------------------------------------------------------------//
// Compute the byte size of a given particle field.
template<class ViewType>
KOKKOS_INLINE_FUNCTION
std::size_t fieldByteSize( const ViewType& view )
{
    int count = 1;
    for ( int d = 1; d < view.Rank; ++d )
        count *= view.extent(d);
    return count * sizeof( typename ViewType::value_type );
}

//---------------------------------------------------------------------------//
// Compute the byte size of a particle.
template<class FieldType>
std::size_t particleByteSize( const FieldType& field )
{
    return fieldByteSize( field );
}

template<class FieldType, class ... FieldViewTypes>
std::size_t particleByteSize( const FieldType& field, FieldViewTypes&&... next )
{
    return fieldByteSize( field ) + particleByteSize( next... );
}

//---------------------------------------------------------------------------//
// Particle destination
//---------------------------------------------------------------------------//
template<class CoordViewType>
void computeParticleDestinations(
    const GlobalGrid& global_grid,
    const CoordViewType& coords,
    Kokkos::View<int*,typename CoordViewType::memory_space>& destinations,
    Kokkos::View<std::size_t[27],typename CoordViewType::memory_space>& send_counts,
    Kokkos::View<std::size_t[27],typename CoordViewType::memory_space>& send_offsets )
{
    using execution_space = typename CoordViewType::execution_space;

    // Get the grid block without a halo.
    const GridBlock& local_block = global_grid.block();

    // Locate the particles in the global grid and get their destination
    // rank. The particle halo should be constructed such that particles will
    // only move to a location in the 26 neighbor halo.
    auto num_particle_send = coords.extent(0);
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
            int di = 0;
            if ( coords(p,Dim::I) < low_x ) di = -1;
            else if ( coords(p,Dim::I) > high_x ) di = 1;

            int dj = 0;
            if ( coords(p,Dim::J) < low_y ) dj = -1;
            else if ( coords(p,Dim::J) > high_y ) dj = 1;

            int dk = 0;
            if ( coords(p,Dim::K) < low_z ) dk = -1;
            else if ( coords(p,Dim::K) > high_z ) dk = 1;

            // If we are sending to ourselves assign a destination value to
            // be one past the last neighbor so it is last in the list.
            if ( di==0 && dj==0 && dk==0 )
            {
                destinations(p) = 26;
                return;
            }

            // Otherwise compute the local neighbor id.
            int nid = 0;
            for ( int k = -1; k < 2; ++k )
                for ( int j = -1; j < 2; ++j )
                    for ( int i = -1; i < 2; ++i )
                        if ( !(i==0 && j==0 && k==0) )
                        {
                            if (i==di && j==dj && k==dk)
                            {
                                destinations(p) = nid;
                                break;
                            }
                            else
                            {
                                ++nid;
                            }
                        }
        } );


    // Count the number of particles going to each neighbor.
    auto send_counts_sv =
        Kokkos::Experimental::create_scatter_view( send_counts );
    Kokkos::parallel_for(
        "redistribute_count",
        Kokkos::RangePolicy<execution_space>(0,num_particle_send),
        KOKKOS_LAMBDA( const int p ){
            auto send_counts_sv_data = send_counts_sv.access();
            send_counts_sv_data( destinations(p) ) += 1;
        } );
    Kokkos::Experimental::contribute( send_counts, send_counts_sv );

    // Compute the count offset for each neighbor via exclusive scan.
    Kokkos::parallel_scan(
        "redistribute_offset_scan",
        Kokkos::RangePolicy<execution_space>(0,27),
        KOKKOS_LAMBDA( const int i, int& update, const bool final_pass ){
            if ( final_pass ) send_offsets(i) = update;
            update += send_counts(i);
        } );
}

//---------------------------------------------------------------------------//
// Neighbor selection
//---------------------------------------------------------------------------//
// Determine if each of the 26 adjacent blocks in logical space is a rank we
// should send to.
std::vector<bool> getNeighbors( const GridBlock& grid )
{
    std::vector<bool> is_neighbor;
    is_neighbor.reserve( 26 );

    auto has_halo =
        [&]( const int dim, const int logical_index ){
            bool halo_check = true;
            if ( -1 == logical_index )
                halo_check = grid.hasHalo(2*dim);
            else if ( 1 == logical_index )
                halo_check = grid.hasHalo(2*dim+1);
            return halo_check;
        };

    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 && k==0) )
                    is_neighbor.push_back( has_halo(Dim::I,i) &&
                                           has_halo(Dim::J,j) &&
                                           has_halo(Dim::K,k) );

    return is_neighbor;
}

//---------------------------------------------------------------------------//
// Serialization - PACKING
//---------------------------------------------------------------------------//
// Pack a rank 0 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void packData( char* send_buffer,
               const int particle_id,
               const ViewType& view,
               typename std::enable_if<
               1==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(send_buffer);
    *ptr = view( particle_id );
}

// Pack a rank 1 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void packData( char* send_buffer,
               const int particle_id,
               const ViewType& view,
               typename std::enable_if<
               2==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(send_buffer);
    for ( std::size_t d0 = 0; d0 < view.extent(1); ++d0, ++ptr )
        *ptr = view( particle_id, d0 );
}

// Pack a rank 2 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void packData( char* send_buffer,
               const int particle_id,
               const ViewType& view,
               typename std::enable_if<
               3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(send_buffer);
    for ( std::size_t d0 = 0; d0 < view.extent(1); ++d0 )
        for ( std::size_t d1 = 0; d1 < view.extent(2); ++d1, ++ptr )
            *ptr = view( particle_id, d0 , d1 );
}

// Pack a field into the send buffer.
template<typename FieldView,
         typename DestinationView,
         typename OffsetView,
         typename SendBuffer>
void packField( const FieldView& field,
                const DestinationView& destinations,
                const OffsetView& offsets,
                const std::size_t particle_byte_size,
                const std::size_t element_byte_offset,
                SendBuffer& send_buffer )
{
    using memory_space = typename FieldView::memory_space;
    using execution_space = typename FieldView::execution_space;

    // Get the number of particles.
    auto num_particle = field.extent(0);

    // Create a view of counts to track where we are writing.
    Kokkos::View<std::size_t[27],memory_space> counts( "pack_counts" );

    // Pack the field into the send buffer. This includes the data staying on
    // this rank.
    //
    // We offset into the send buffer by first computing the particle id of
    // the destination rank that we are packing (lid). We then calculate the
    // byte offset to that particle's location and then the byte offset to the
    // field we are packing.
    Kokkos::parallel_for(
        "redistribute_pack_send_buffer",
        Kokkos::RangePolicy<execution_space>(0,num_particle),
        KOKKOS_LAMBDA( const int p ){
            auto lid = Kokkos::atomic_fetch_add(
                &counts(destinations(p)), 1 );
            std::size_t local_offset =
                (offsets(destinations(p)) + lid) * particle_byte_size +
                element_byte_offset;
            packData( &send_buffer(local_offset), p, field );
        } );
}

// Pack particles into a buffer.
template<typename DestinationView,
         typename OffsetView,
         typename SendBuffer,
         typename ViewType>
void packParticlesImpl( SendBuffer& send_buffer,
                        const DestinationView& destinations,
                        const OffsetView& offsets,
                        const std::size_t particle_byte_size,
                        const std::size_t element_byte_offset,
                        const ViewType& view )
{
    // Pack the field.
    packField( view, destinations, offsets, particle_byte_size,
               element_byte_offset, send_buffer );
}

// Pack particles into a buffer.
template<typename DestinationView,
         typename OffsetView,
         typename SendBuffer,
         typename ViewType,
         typename ... FieldViews>
void packParticlesImpl( SendBuffer& send_buffer,
                        const DestinationView& destinations,
                        const OffsetView& offsets,
                        const std::size_t particle_byte_size,
                        const std::size_t element_byte_offset,
                        const ViewType& view,
                        FieldViews&&... fields )
{
    // Pack the field.
    packField( view, destinations, offsets, particle_byte_size,
               element_byte_offset, send_buffer );

    // Pack the next fields. Increment the element byte offset.
    packParticlesImpl( send_buffer,
                       destinations,
                       offsets,
                       particle_byte_size,
                       element_byte_offset + fieldByteSize(view),
                       fields... );
}

// Pack an arbitrary number of particle fields into a buffer. The
// implementation functions unroll the variadic list of fields.
template<typename DestinationView,
         typename OffsetView,
         typename SendBuffer,
         typename ... FieldViews>
void packParticles( SendBuffer& send_buffer,
                    const DestinationView& destinations,
                    const OffsetView& offsets,
                    const std::size_t particle_byte_size,
                    FieldViews&&... fields )
{
    packParticlesImpl( send_buffer, destinations, offsets, particle_byte_size,
                       0, fields... );
}

//---------------------------------------------------------------------------//
// Serialization - UNPACKING
//---------------------------------------------------------------------------//
// Unpack a rank 0 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void unpackData( char* receive_buffer,
                 const int particle_id,
                 ViewType& view,
                 typename std::enable_if<
                 1==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(receive_buffer);
    view( particle_id ) = *ptr;
}

// Unpack a rank 1 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void unpackData( char* receive_buffer,
                 const int particle_id,
                 ViewType& view,
                 typename std::enable_if<
                 2==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(receive_buffer);
    for ( std::size_t d0 = 0; d0 < view.extent(1); ++d0, ++ptr )
        view( particle_id, d0 ) = *ptr;
}

// Unpack a rank 2 field.
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void unpackData( char* receive_buffer,
                 const int particle_id,
                 ViewType& view,
                 typename std::enable_if<
                 3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    auto ptr = reinterpret_cast<typename ViewType::value_type*>(receive_buffer);
    for ( std::size_t d0 = 0; d0 < view.extent(1); ++d0 )
        for ( std::size_t d1 = 0; d1 < view.extent(2); ++d1, ++ptr )
            view( particle_id, d0 , d1 ) = *ptr;
}

// Unpack a field into the receive buffer.
template<typename FieldView, typename ReceiveBuffer>
void unpackField( FieldView& field,
                  const std::size_t num_unpack,
                  const std::size_t write_offset,
                  const std::size_t particle_byte_size,
                  const std::size_t element_byte_offset,
                  const ReceiveBuffer& receive_buffer )
{
    using execution_space = typename FieldView::execution_space;

    // Unpack the field into the receive buffer.
    //
    // For the receive buffer we are reading from, we start reading at the
    // number of bytes between the particle we are operating on and the
    // beginning on the buffer plus the byte offset for the field we are
    // writing.
    //
    // For the field we are writing into we use write_offset to calculate an
    // index offset into that field as we may not be writing from the
    // beginning. We need this because we unpack data in two passes: one to
    // unpack the communicated particles and one to unpack the particles that
    // stayed on a given rank.
    Kokkos::parallel_for(
        "redistribute_unpack_receive_buffer",
        Kokkos::RangePolicy<execution_space>(0,num_unpack),
        KOKKOS_LAMBDA( const int p ){
            std::size_t local_offset =
                p * particle_byte_size + element_byte_offset;
            unpackData( &receive_buffer( local_offset ),
                        p + write_offset,
                        field );
        } );
}

// Unpack particles from a buffer.
template<typename ReceiveBuffer, typename ViewType>
void unpackParticlesImpl( const ReceiveBuffer& receive_buffer,
                          const std::size_t num_unpack,
                          const std::size_t write_offset,
                          const std::size_t particle_byte_size,
                          const std::size_t element_byte_offset,
                          const ViewType& view )
{
    // Unpack the field.
    unpackField( view, num_unpack, write_offset, particle_byte_size,
                 element_byte_offset, receive_buffer );
}

// Unpack particles from a buffer.
template<typename ReceiveBuffer, typename ViewType, typename ... FieldViews>
void unpackParticlesImpl( const ReceiveBuffer& receive_buffer,
                          const std::size_t num_unpack,
                          const std::size_t write_offset,
                          const std::size_t particle_byte_size,
                          const std::size_t element_byte_offset,
                          const ViewType& view,
                          FieldViews&&... fields )
{
    // Unpack the field.
    unpackField( view, num_unpack, write_offset, particle_byte_size,
                 element_byte_offset, receive_buffer );

    // Unpack the next fields. Increment the element byte offset.
    unpackParticlesImpl( receive_buffer,
                         num_unpack,
                         write_offset,
                         particle_byte_size,
                         element_byte_offset + fieldByteSize(view),
                         fields... );
}

// Unpack an arbitrary number of particle fields from a buffer. The
// implementation functions unroll the variadic list of fields.
template<typename ReceiveBuffer, typename ... FieldViews>
void unpackParticles( const ReceiveBuffer& receive_buffer,
                      const std::size_t num_unpack,
                      const std::size_t write_offset,
                      const std::size_t particle_byte_size,
                      FieldViews&&... fields )
{
    unpackParticlesImpl( receive_buffer, num_unpack, write_offset,
                         particle_byte_size, 0, fields... );
}

//---------------------------------------------------------------------------//
// Periodic coordinate shift for communication across a periodic boundary.
//---------------------------------------------------------------------------//
// When particles cross a periodic boundary their coordinates must be shifted
// to represent a new physical location.
template<class CoordViewType>
void shiftPeriodicCoordinates( const GlobalGrid& global_grid,
                               CoordViewType& coords )
{
    using execution_space = typename CoordViewType::execution_space;

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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
            Kokkos::RangePolicy<execution_space>(0,coords.extent(0)),
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
  \brief Redistribute particles to new owning ranks based on their location
  using the graph communication pattern.

  Communication in this pattern occurs over the entire halo of a specified
  width using the full 26 neighbor communication pattern. Neighbors are
  ordered in the 3x3 grid about the calling rank with the I index moving the
  fastest and the K index moving the slowest.

  \param global_grid The grid over which to redistribute the particles.

  \param coords The particle coordinates. These will also be redistributed.

  \param fields Tuple of particle fields to be redistributed.
 */
template<class CoordViewType, class ... FieldViewTypes>
void redistribute( const GlobalGrid& global_grid,
                   CoordViewType& coords,
                   FieldViewTypes&&... fields )
{
    using memory_space = typename CoordViewType::memory_space;

    // Get the grid block without a halo.
    const GridBlock& local_block = global_grid.block();

    // Get the neighbor filter. This will tell us if we actually send to any
    // of the 26 adjacent logical neighbors.
    auto is_neighbor = getNeighbors( local_block );

    // Check for the case where we have no logical neighbors. This will occur
    // only in serial when there are no periodic boundaries. In that case we
    // do not need to communicate and can just return.
    bool has_neighbor = false;
    for ( auto b : is_neighbor ) if (b) has_neighbor = true;
    if ( !has_neighbor ) return;

    // Locate the particles in the global grid and get their destination
    // rank. The particle halo should be constructed such that particles will
    // only move to a location in the 26 neighbor halo. Also get the counts
    // and offsets.
    auto num_particle_send = coords.extent(0);
    Kokkos::View<int*,memory_space> destinations(
        Kokkos::ViewAllocateWithoutInitializing("destinations"),
        num_particle_send );
    Kokkos::View<std::size_t[27],memory_space> send_counts( "send_counts" );
    Kokkos::View<std::size_t[27],memory_space> send_offsets( "send_offsets" );
    computeParticleDestinations(
        global_grid, coords, destinations, send_counts, send_offsets );

    // Get counts and offsets on the host.
    auto send_counts_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), send_counts );
    auto send_offsets_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), send_offsets );

    // Compute the total number of bytes needed to represent each particle.
    auto particle_byte_size = particleByteSize( coords, fields... );

    // Compute the send byte counts and offsets for each neighbor. We compute
    // these because we send/receive in bytes due to the possibility of
    // arbitrary field types.
    std::vector<int> send_byte_counts;
    send_byte_counts.reserve(26);
    std::vector<int> send_byte_offsets;;
    send_byte_offsets.reserve(26);
    for ( int n = 0; n < 26; ++n )
    {
        if ( is_neighbor[n] )
        {
            send_byte_counts.push_back(
                send_counts_host(n) * particle_byte_size );
            send_byte_offsets.push_back(
                send_offsets_host(n) * particle_byte_size );
        }
    }

    // Communicate the amount of data each neighbor will receive. This only
    // includes the data leaving this rank.
    std::vector<int> receive_byte_counts( send_byte_counts.size() );
    MPI_Neighbor_alltoall( send_byte_counts.data(), 1, MPI_INT,
                           receive_byte_counts.data(), 1, MPI_INT,
                           global_grid.graphComm() );

    // Compute the receive byte offsets for each neighbor via exclusive scan.
    std::vector<int> receive_byte_offsets( receive_byte_counts.size(), 0 );
    std::partial_sum( receive_byte_counts.begin(),
                      receive_byte_counts.end() - 1,
                      receive_byte_offsets.begin() + 1 );

    // Allocate a send buffer. This includes both the data staying on this
    // rank and the data going to other ranks.
    std::size_t total_send_bytes = particle_byte_size * num_particle_send;
    Kokkos::View<char*,memory_space>
        send_buffer( "send_buffer", total_send_bytes );

    // Pack the send buffer. The data staying on this rank is packed into the
    // end of the buffer after all of the other ranks.
    packParticles( send_buffer, destinations, send_offsets, particle_byte_size,
                   coords, fields... );

    // Allocate a receive buffer. We only receive data coming from other
    // ranks.
    auto total_receive_bytes = std::accumulate( receive_byte_counts.begin(),
                                                receive_byte_counts.end(), 0 );
    Kokkos::View<char*,memory_space>
        receive_buffer( "receive_buffer", total_receive_bytes );

    // Communicate the particles that are going to other ranks.
    MPI_Neighbor_alltoallv(
        send_buffer.data(), send_byte_counts.data(),
        send_byte_offsets.data(), MPI_BYTE,
        receive_buffer.data(), receive_byte_counts.data(),
        receive_byte_offsets.data(), MPI_BYTE,
        global_grid.graphComm() );

    // Compute the total number of particles we received via communication.
    std::size_t num_particle_receive =
        total_receive_bytes / particle_byte_size;

    // Resize the fields. This includes the data we received and the data
    // staying on this rank.
    ParticleFieldOps::resize(
        num_particle_receive + send_counts_host(26), coords, fields... );

    // Unpack the receive buffer into the first 26 positions.
    unpackParticles( receive_buffer, num_particle_receive, 0,
                     particle_byte_size, coords, fields... );

    // Get a view of the particle data that stayed on this rank.
    auto particles_that_stay = Kokkos::subview(
        send_buffer, Kokkos::pair<int,int>(
            send_offsets_host(26) * particle_byte_size,
            (send_offsets_host(26)+send_counts_host(26)) * particle_byte_size) );

    // Unpack the data staying on this rank into the last position.
    unpackParticles( particles_that_stay, send_counts_host(26),
                     send_offsets_host(26), particle_byte_size,
                     coords, fields... );

    // Shift the particle coordinates for movement across periodic boundaries.
    shiftPeriodicCoordinates( global_grid, coords );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleCommunication
} // end namespace Harlow

#endif // end HARLOW_PARTICLECOMMUNICATION_HPP
