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
std::vector<int> neighborCounts( const GridBlock& grid, CartesianTag )
{
    std::vector<int> counts( 6, 0 );

    // I- neighbor
    if ( !grid.onBoundary(DomainBoundary::LowX) || grid.isPeriodic(Dim::I) )
         counts[0] = grid.haloSize() *
                     grid.localNumCell(Dim::J) * grid.localNumCell(Dim::K);

    // I+ neighbor
    if ( !grid.onBoundary(DomainBoundary::HighX) || grid.isPeriodic(Dim::I) )
        counts[1] = grid.haloSize() *
                    grid.localNumCell(Dim::J) * grid.localNumCell(Dim::K);

    // J- neighbor
    if ( !grid.onBoundary(DomainBoundary::LowY) || grid.isPeriodic(Dim::J) )
        counts[2] = grid.haloSize() *
                    grid.localNumCell(Dim::I) * grid.localNumCell(Dim::K);

    // J+ neighbor
    if ( !grid.onBoundary(DomainBoundary::HighY) || grid.isPeriodic(Dim::J) )
        counts[3] = grid.haloSize() *
                    grid.localNumCell(Dim::I) * grid.localNumCell(Dim::K);

    // K- neighbor
    if ( !grid.onBoundary(DomainBoundary::LowZ) || grid.isPeriodic(Dim::K) )
        counts[4] = grid.haloSize() *
                    grid.localNumCell(Dim::I) * grid.localNumCell(Dim::J);

    // K+ neighbor
    if ( !grid.onBoundary(DomainBoundary::HighZ) || grid.isPeriodic(Dim::K) )
        counts[5] = grid.haloSize() *
                    grid.localNumCell(Dim::I) * grid.localNumCell(Dim::J);

    return counts;
}

//---------------------------------------------------------------------------//
// Serialization
//---------------------------------------------------------------------------//
// Pack a neighbor into a send buffer.
template<typename ViewType,typename BufferType, typename PackRange>
void packNeighbor( const PackRange& pack_range,
                   const ViewType field,
                   const int offset,
                   BufferType send_buffer )
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

//---------------------------------------------------------------------------//
// Gather a neighbor from a receive buffer.
template<typename ViewType,typename BufferType, typename PackRange>
void gatherNeighbor( const PackRange& unpack_range,
                     const BufferType receive_buffer,
                     const int offset,
                     ViewType field )
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

//---------------------------------------------------------------------------//
// Scatter a neighbor from a receive buffer.
template<typename ViewType,typename BufferType, typename PackRange>
void scatterNeighbor( const PackRange& unpack_range,
                      const BufferType receive_buffer,
                      const int offset,
                      ViewType field )
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

    // Add the halo contribution into the local cells.
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

//---------------------------------------------------------------------------//
// Gather
//---------------------------------------------------------------------------//
/*!
  \brief Gather field data from owning ranks into the halo using a Cartesian
  communication pattern.

  This operation is specifically for implementing grid operators on scalar
  cell fields for the solve during the projection phase of the algorithm. The
  differential operator stencils we use are with a face/cell staggered scheme
  and therefore we do not need to communicate the corners of the halo - we
  only need to send to our 6 neighbors.
*/
template<class GridFieldType>
void gather( GridFieldType& grid_field, CartesianTag tag )
{
    if ( grid_field.location() != FieldLocation::Cell )
        throw std::invalid_argument("Gather only for cell fields");

    // Get the field.
    auto field = grid_field.data();
    if ( field.Rank != 3 )
        throw std::invalid_argument("Gather only for scalar fields");
    using view_type = decltype(field);

    // Get the local grid.
    const auto& block = grid_field.block();
    if ( block.haloSize() != 1 )
        throw std::invalid_argument(
            "Gather only for fields with halo size of 1");

    // Count the number of cells we are sending to each neighbor.
    std::vector<int> counts = neighborCounts( block, tag );

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
            Kokkos::pair<int,int>(block.localCellBegin(Dim::I),
                                  block.localCellEnd(Dim::I));
        pack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::J),
                                  block.localCellEnd(Dim::J));
        pack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::K),
                                  block.localCellEnd(Dim::K));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our layer of local cells.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localCellBegin(d),
                block.localCellBegin(d) + block.haloSize() );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our layer of local cells.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localCellEnd(d) - block.haloSize(),
                block.localCellEnd(d) );

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
            Kokkos::pair<int,int>(block.localCellBegin(Dim::I),
                                  block.localCellEnd(Dim::I));
        unpack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::J),
                                  block.localCellEnd(Dim::J));
        unpack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::K),
                                  block.localCellEnd(Dim::K));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our halo cells.
            unpack_range[d] = Kokkos::pair<int,int>(
                0, block.localCellBegin(d) );

            // Unpack
            gatherNeighbor( unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our halo cells.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localCellEnd(d), block.numCell(d) );

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

  This operation is specifically for implementing grid operators on scalar
  cell fields for the solve during the projection phase of the algorithm. The
  differential operator stencils we use are with a face/cell staggered scheme
  and therefore we do not need to communicate the corners of the halo - we
  only need to send to our 6 neighbors.
*/
template<class GridFieldType>
void scatter( GridFieldType& grid_field, CartesianTag tag )
{
    if ( grid_field.location() != FieldLocation::Cell )
        throw std::invalid_argument("Scatter only for cell fields");

    // Get the field.
    auto field = grid_field.data();
    if ( field.Rank != 3 )
        throw std::invalid_argument("Scatter only for scalar fields");
    using view_type = decltype(field);

    // Get the local grid block.
    const auto& block = grid_field.block();
    if ( block.haloSize() != 1 )
        throw std::invalid_argument(
            "Scatter only for fields with halo size of 1");

    // Count the number of cells we are sending to each neighbor.
    std::vector<int> counts = neighborCounts( block, tag );

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
            Kokkos::pair<int,int>(block.localCellBegin(Dim::I),
                                  block.localCellEnd(Dim::I));
        pack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::J),
                                  block.localCellEnd(Dim::J));
        pack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::K),
                                  block.localCellEnd(Dim::K));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our ghost cells.
            pack_range[d] = Kokkos::pair<int,int>(
                0, block.localCellBegin(d) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our ghost cells.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localCellEnd(d), block.numCell(d) );

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
            Kokkos::pair<int,int>(block.localCellBegin(Dim::I),
                                  block.localCellEnd(Dim::I));
        unpack_range[Dim::J] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::J),
                                  block.localCellEnd(Dim::J));
        unpack_range[Dim::K] =
            Kokkos::pair<int,int>(block.localCellBegin(Dim::K),
                                  block.localCellEnd(Dim::K));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our local cells.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localCellBegin(d),
                block.localCellBegin(d) + block.haloSize() );

            // Unpack
            scatterNeighbor(
                unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our local cells.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localCellEnd(d) - block.haloSize(),
                block.localCellEnd(d) );

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
