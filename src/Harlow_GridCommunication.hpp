#ifndef HARLOW_GRIDCOMMUNICATION_HPP
#define HARLOW_GRIDCOMMUNICATION_HPP

#include <Harlow_GridBlock.hpp>
#include <Harlow_GridField.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_MpiTraits.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <array>

namespace Harlow
{
namespace GridCommunication
{
//---------------------------------------------------------------------------//
// Communication pattern tags.
//---------------------------------------------------------------------------//
// Tag for doing the 6-neighbor Cartesian topology communication.
struct CartesianTag {};

// Tag for doing the 26-neighbor graph topology communication.
struct GraphTag {};

//---------------------------------------------------------------------------//
// Neighbor counts.
//---------------------------------------------------------------------------//
// Given a grid block calculate the counts to send/receive from each Cartesian
// halo neighbor. Neighbors are ordered as {-i,+i,-j,+j,-k,+k}. Blocks on
// physical boundaries that are not periodic have a count of 0.
std::vector<int> neighborCounts( const GridBlock& grid,
                                 const int halo_size,
                                 const int entity_type,
                                 CartesianTag )
{
    if ( halo_size > grid.haloSize() )
        throw std::invalid_argument("Requested halo size larger than grid halo");

    std::vector<int> counts( 6, 0 );

    // Compute the size of a neighbor face in a given dimension.
    auto face_size =
        [&]( const int dim ){
            int size = -1;
            if ( Dim::I == dim )
                size = grid.localNumEntity(entity_type,Dim::J) *
                       grid.localNumEntity(entity_type,Dim::K);
            else if ( Dim::J == dim )
                size = grid.localNumEntity(entity_type,Dim::I) *
                       grid.localNumEntity(entity_type,Dim::K);
            else if ( Dim::K == dim )
                size = grid.localNumEntity(entity_type,Dim::I) *
                       grid.localNumEntity(entity_type,Dim::J);
            return size;
        };

    // If we are not on a physical boundary or if that boundary is periodic
    // then we send data.
    for ( int n = 0; n < 6; ++n )
        if ( grid.hasHalo(n) )
            counts[n] = halo_size * face_size(n/2);

    return counts;
}

//---------------------------------------------------------------------------//
// Given a grid block calculate the counts to send/receive from each graph
// halo neighbor. Neighbors are ordered as with I moving the fastest and K
// moving the slowest in the 3x3 grid about the local rank. Blocks on physical
// boundaries that are not periodic have a count of 0.
std::vector<int> neighborCounts( const GridBlock& grid,
                                 const int halo_size,
                                 const int entity_type,
                                 GraphTag )
{
    if ( halo_size > grid.haloSize() )
        throw std::invalid_argument("Requested halo size larger than grid halo");

    std::vector<int> counts;
    counts.reserve(26);

    // Compute the number of entities in a given dimension for a neighbor at
    // the given logical index in the 3x3 grid. If we are not on a physical
    // boundary or if that boundary is periodic then we send data.
    auto num_entity =
        [&]( const int dim, const int logical_index ){
            int nc = -1;
            if ( -1 == logical_index )
                nc = grid.hasHalo(2*dim) ? halo_size : 0;
            else if ( 0 == logical_index )
                nc = grid.localNumEntity(entity_type,dim);
            else if ( 1 == logical_index )
                nc = grid.hasHalo(2*dim+1) ? halo_size : 0;
            return nc;
        };

    // Compute the size of each neighbor. Note that we do not send to
    // ourselves (when each index is 0).
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 && k==0) )
                    counts.push_back( num_entity( Dim::I, i ) *
                                      num_entity( Dim::J, j ) *
                                      num_entity( Dim::K, k ) );

    return counts;
}

//---------------------------------------------------------------------------//
// Serialization
//---------------------------------------------------------------------------//
// Pack a neighbor into a send buffer.

// Rank 0 fields
template<typename ViewType,typename BufferType, typename PackRange>
void packNeighbor(
    const PackRange& pack_range,
    const ViewType field,
    const int offset,
    BufferType send_buffer,
    typename std::enable_if<3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the send buffer for the neighbor.
    NeighborBuffer send_buffer_subview(
        send_buffer.data() + offset,
        pack_range[Dim::I].second - pack_range[Dim::I].first,
        pack_range[Dim::J].second - pack_range[Dim::J].first,
        pack_range[Dim::K].second - pack_range[Dim::K].first );

    // Get a view of the layer we are sending.
    auto field_subview = Kokkos::subview(
        field,
        pack_range[Dim::I], pack_range[Dim::J], pack_range[Dim::K] );

    // Copy to the send buffer.
    Kokkos::deep_copy( send_buffer_subview, field_subview );
}

// Rank 1 fields
template<typename ViewType,typename BufferType, typename PackRange>
void packNeighbor(
    const PackRange& pack_range,
    const ViewType field,
    const int offset,
    BufferType send_buffer,
    typename std::enable_if<4==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the send buffer for the neighbor.
    NeighborBuffer send_buffer_subview(
        send_buffer.data() + offset,
        pack_range[Dim::I].second - pack_range[Dim::I].first,
        pack_range[Dim::J].second - pack_range[Dim::J].first,
        pack_range[Dim::K].second - pack_range[Dim::K].first,
        field.extent(3) );

    // Get a view of the layer we are sending.
    auto field_subview = Kokkos::subview(
        field,
        pack_range[Dim::I], pack_range[Dim::J], pack_range[Dim::K], Kokkos::ALL );

    // Copy to the send buffer.
    Kokkos::deep_copy( send_buffer_subview, field_subview );
}

//---------------------------------------------------------------------------//
// Gather a neighbor from a receive buffer.

// Rank 0 fields.
template<typename ViewType,typename BufferType, typename PackRange>
void gatherNeighbor(
    const PackRange& unpack_range,
    const BufferType receive_buffer,
    const int offset,
    ViewType field,
    typename std::enable_if<3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the receive buffer for the neighbor.
    NeighborBuffer receive_buffer_subview(
        receive_buffer.data() + offset,
        unpack_range[Dim::I].second - unpack_range[Dim::I].first,
        unpack_range[Dim::J].second - unpack_range[Dim::J].first,
        unpack_range[Dim::K].second - unpack_range[Dim::K].first );

    // Get a view of the layer we are receiving.
    auto field_subview = Kokkos::subview( field,
                                          unpack_range[Dim::I],
                                          unpack_range[Dim::J],
                                          unpack_range[Dim::K] );

    // Copy into the halo.
    Kokkos::deep_copy( field_subview, receive_buffer_subview );
}

// Rank 1 fields.
template<typename ViewType,typename BufferType, typename PackRange>
void gatherNeighbor(
    const PackRange& unpack_range,
    const BufferType receive_buffer,
    const int offset,
    ViewType field,
    typename std::enable_if<4==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the receive buffer for the neighbor.
    NeighborBuffer receive_buffer_subview(
        receive_buffer.data() + offset,
        unpack_range[Dim::I].second - unpack_range[Dim::I].first,
        unpack_range[Dim::J].second - unpack_range[Dim::J].first,
        unpack_range[Dim::K].second - unpack_range[Dim::K].first,
        field.extent(3) );

    // Get a view of the layer we are receiving.
    auto field_subview = Kokkos::subview( field,
                                          unpack_range[Dim::I],
                                          unpack_range[Dim::J],
                                          unpack_range[Dim::K],
                                          Kokkos::ALL );

    // Copy into the halo.
    Kokkos::deep_copy( field_subview, receive_buffer_subview );
}

//---------------------------------------------------------------------------//
// Scatter a neighbor from a receive buffer.

// Rank 0 fields.
template<typename ViewType,typename BufferType, typename PackRange>
void scatterNeighbor(
    const PackRange& unpack_range,
    const BufferType receive_buffer,
    const int offset,
    ViewType field,
    typename std::enable_if<3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Declare an execution policy for the scatter update.
    using ExecPolicy =
        Kokkos::MDRangePolicy<typename ViewType::execution_space,
                              Kokkos::Rank<3> >;

    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the receive buffer for the neighbor.
    NeighborBuffer receive_buffer_subview(
        receive_buffer.data() + offset,
        unpack_range[Dim::I].second - unpack_range[Dim::I].first,
        unpack_range[Dim::J].second - unpack_range[Dim::J].first,
        unpack_range[Dim::K].second - unpack_range[Dim::K].first );

    // Get a view of the layer we are receiving.
    auto field_subview = Kokkos::subview( field,
                                          unpack_range[Dim::I],
                                          unpack_range[Dim::J],
                                          unpack_range[Dim::K] );

    // Add the halo contribution into the local entities.
    Kokkos::parallel_for(
        "Scatter negative neighbor update",
        ExecPolicy(
            {0,0,0},
            {unpack_range[Dim::I].second - unpack_range[Dim::I].first,
             unpack_range[Dim::J].second - unpack_range[Dim::J].first,
             unpack_range[Dim::K].second - unpack_range[Dim::K].first} ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            field_subview(i,j,k) += receive_buffer_subview(i,j,k);
        } );
}

// Rank 1 fields.
template<typename ViewType,typename BufferType, typename PackRange>
void scatterNeighbor(
    const PackRange& unpack_range,
    const BufferType receive_buffer,
    const int offset,
    ViewType field,
    typename std::enable_if<4==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Declare an execution policy for the scatter update.
    using ExecPolicy =
        Kokkos::MDRangePolicy<typename ViewType::execution_space,
                              Kokkos::Rank<4> >;

    // Define an offset view type.
    using NeighborBuffer =
        Kokkos::View<typename ViewType::data_type,
                     typename ViewType::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    // Get the offset view into the receive buffer for the neighbor.
    NeighborBuffer receive_buffer_subview(
        receive_buffer.data() + offset,
        unpack_range[Dim::I].second - unpack_range[Dim::I].first,
        unpack_range[Dim::J].second - unpack_range[Dim::J].first,
        unpack_range[Dim::K].second - unpack_range[Dim::K].first,
        field.extent(3) );

    // Get a view of the layer we are receiving.
    auto field_subview = Kokkos::subview( field,
                                          unpack_range[Dim::I],
                                          unpack_range[Dim::J],
                                          unpack_range[Dim::K],
                                          Kokkos::ALL );

    // Add the halo contribution into the local entities.
    Kokkos::parallel_for(
        "Scatter negative neighbor update",
        ExecPolicy(
            {{0,0,0,0}},
            {{unpack_range[Dim::I].second - unpack_range[Dim::I].first,
              unpack_range[Dim::J].second - unpack_range[Dim::J].first,
              unpack_range[Dim::K].second - unpack_range[Dim::K].first,
              field.extent(3) }} ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int d0 ) {
            field_subview(i,j,k,d0) += receive_buffer_subview(i,j,k,d0);
        } );
}

//---------------------------------------------------------------------------//
// Gather
//---------------------------------------------------------------------------//
/*!
  \brief Gather field data from owning ranks into the halo using a Cartesian
  communication pattern.

  Communication in this pattern occurs between only the locally-owned elements
  of the 6 neighbors in the cartesian topology in a halo of specified
  width. Neighbors are ordered as {-i,+i,-j,+j,-k,+k}.

  \param grid_field The field to gather.

  \param halo_size The size of the halo over which to gather the data.

  \param tag tag Algorithm tag indicating a Cartesian topology communication.
*/
template<class GridFieldType>
void gather( GridFieldType& grid_field, const int halo_size, CartesianTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid.
    const auto& block = grid_field.block();
    if ( halo_size > block.haloSize() )
        throw std::invalid_argument("Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor.
    std::vector<int> counts =
        neighborCounts( block, halo_size, location, tag );

    // Compute the number of multidimensional elements in at each entity.
    int md_size = 1;
    for ( int d = 3; d < field.Rank; ++d )
        md_size *= field.extent(d);

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets view exclusive scan.
    int total_elements = 0;
    std::vector<int> offsets( 6, 0 );
    for ( int n = 0; n < 6; ++n )
    {
        offsets[n] = total_elements;
        total_elements += counts[n];
    }

    // Allocate a send buffer.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        send_buffer( "gather_send", total_elements );

    // Pack the send buffer.
    for ( int d = 0; d < 3; ++d )
    {
        std::array<Kokkos::pair<int,int>,3> pack_range;
        pack_range[Dim::I] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::I),
                                  block.localEntityEnd(location,Dim::I));
        pack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::J),
                                  block.localEntityEnd(location,Dim::J));
        pack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::K),
                                  block.localEntityEnd(location,Dim::K));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our layer of local entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d),
                block.localEntityBegin(location,d) + halo_size );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our layer of local entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d) - halo_size,
                block.localEntityEnd(location,d) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d+1], send_buffer );
        }
    }

    // Allocate a receive buffer. We receive as much as we send from our
    // opposite neighbors.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        receive_buffer( "gather_receive", total_elements );

    // Do the communciation.
    MPI_Neighbor_alltoallv(
        send_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        receive_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        grid_field.cartesianComm() );

    // Unpack the receive buffer.
    for ( int d = 0; d < 3; ++d )
    {
        std::array<Kokkos::pair<int,int>,3> unpack_range;
        unpack_range[Dim::I] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::I),
                                  block.localEntityEnd(location,Dim::I));
        unpack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::J),
                                  block.localEntityEnd(location,Dim::J));
        unpack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::K),
                                  block.localEntityEnd(location,Dim::K));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our halo entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d) - halo_size,
                block.localEntityBegin(location,d) );

            // Unpack
            gatherNeighbor( unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our halo entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d),
                block.localEntityEnd(location,d) + halo_size );

            // Unpack
            gatherNeighbor(
                unpack_range, receive_buffer, offsets[2*d+1], field );
        }
    }
}

//---------------------------------------------------------------------------//
// Scatter
//---------------------------------------------------------------------------//
/*!
  \brief Scatter the data from the halo back to their owning rank using a
  Cartesian communication pattern.

  Communication in this pattern occurs between only the locally-owned elements
  of the 6 neighbors in the cartesian topology in a halo of specified
  width. Neighbors are ordered as {-i,+i,-j,+j,-k,+k}.

  \param grid_field The field to scatter.

  \param halo_size The size of the halo over which to scatter the data.

  \param tag tag Algorithm tag indicating a Cartesian topology communication.
*/
template<class GridFieldType>
void scatter( GridFieldType& grid_field, const int halo_size, CartesianTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid block.
    const auto& block = grid_field.block();
    if ( halo_size > block.haloSize() )
        throw std::invalid_argument("Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor.
    std::vector<int> counts =
        neighborCounts( block, halo_size, location, tag );

    // Compute the number of multidimensional elements in at each entity.
    int md_size = 1;
    for ( int d = 3; d < field.Rank; ++d )
        md_size *= field.extent(d);

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets view exclusive scan.
    int total_elements = 0;
    std::vector<int> offsets( 6, 0 );
    for ( int n = 0; n < 6; ++n )
    {
        offsets[n] = total_elements;
        total_elements += counts[n];
    }

    // Allocate a send buffer.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        send_buffer( "scatter_send", total_elements );

    // Pack the send buffer.
    for ( int d = 0; d < 3; ++d )
    {
        std::array<Kokkos::pair<int,int>,3> pack_range;
        pack_range[Dim::I] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::I),
                                  block.localEntityEnd(location,Dim::I));
        pack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::J),
                                  block.localEntityEnd(location,Dim::J));
        pack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::K),
                                  block.localEntityEnd(location,Dim::K));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our halo entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d) - halo_size,
                block.localEntityBegin(location,d) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our halo entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d),
                block.localEntityEnd(location,d) + halo_size );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d+1], send_buffer );
        }
    }

    // Allocate a receive buffer. We receive as much as we send from our
    // opposite neighbors.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        receive_buffer( "scatter_receive", total_elements );

    // Do the communciation.
    MPI_Neighbor_alltoallv(
        send_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        receive_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        grid_field.cartesianComm() );

    // Unpack the receive buffer.
    for ( int d = 0; d < 3; ++d )
    {
        std::array<Kokkos::pair<int,int>,3> unpack_range;
        unpack_range[Dim::I] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::I),
                                  block.localEntityEnd(location,Dim::I));
        unpack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::J),
                                  block.localEntityEnd(location,Dim::J));
        unpack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localEntityBegin(location,Dim::K),
                                  block.localEntityEnd(location,Dim::K));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our local entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d),
                block.localEntityBegin(location,d) + halo_size );

            // Unpack
            scatterNeighbor(
                unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our local entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d) - halo_size,
                block.localEntityEnd(location,d) );

            // Unpack
            scatterNeighbor(
                unpack_range, receive_buffer, offsets[2*d+1], field );
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace GridCommunication
} // end namespace Harlow

#endif // end HARLOW_GRIDCOMMUNICATION_HPP
