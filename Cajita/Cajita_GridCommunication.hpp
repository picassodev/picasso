#ifndef CAJITA_GRIDCOMMUNICATION_HPP
#define CAJITA_GRIDCOMMUNICATION_HPP

#include <Cajita_GridBlock.hpp>
#include <Cajita_GridField.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_MpiTraits.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <array>

namespace Cajita
{
namespace GridCommunication
{
//---------------------------------------------------------------------------//
// Communication pattern tags.
//---------------------------------------------------------------------------//
// Tag for communicating with the 6 face-adjacent neighbors with the Cartesian
// topology.
struct CartCommFaceTag {};

// Tag for communicating with the 26 node-adjacent neighbors with the graph
// topology.
struct GraphCommNodeTag {};

// Note - we could also explore a 26 node-adjacent communication
// implementation using 2-passes with the Cartesian communication.

//---------------------------------------------------------------------------//
// Neighbor counts.
//---------------------------------------------------------------------------//
// Given a grid block calculate the counts to send/receive from each Cartesian
// halo neighbor. Neighbors are ordered as {-i,+i,-j,+j,-k,+k}. Blocks on
// physical boundaries that are not periodic have a count of 0.
std::vector<int> neighborCounts( const GridBlock& grid,
                                 const int halo_num_cell,
                                 const int entity_type,
                                 CartCommFaceTag )
{
    if ( halo_num_cell > grid.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than grid halo");

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
        {
            int dim = n / 2;
            int logical_index = (n % 2) ? 1 : -1;
            counts[n] =
                grid.haloNumEntity(entity_type,dim,logical_index,halo_num_cell) *
                face_size(dim);
        };

    return counts;
}

//---------------------------------------------------------------------------//
// Given a grid block calculate the counts to send/receive from each graph
// halo neighbor. Neighbors are ordered as with I moving the fastest and K
// moving the slowest in the 3x3 grid about the local rank. Blocks on physical
// boundaries that are not periodic have a count of 0.
std::vector<int> neighborCounts( const GridBlock& grid,
                                 const int halo_num_cell,
                                 const int entity_type,
                                 GraphCommNodeTag )
{
    if ( halo_num_cell > grid.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than grid halo");

    std::vector<int> counts;
    counts.reserve( 26 );

    // Compute the number of entities in a given dimension for a neighbor at
    // the given logical index in the 3x3 grid. If we are not on a physical
    // boundary or if that boundary is periodic then we send data.
    auto num_entity =
        [&]( const int dim, const int logical_index ){
            int nc = -1;
            if ( -1 == logical_index )
                nc = grid.hasHalo(2*dim)
                     ? grid.haloNumEntity(
                         entity_type,dim,logical_index,halo_num_cell)
                     : 0;
            else if ( 0 == logical_index )
                nc = grid.haloNumEntity(
                    entity_type,dim,logical_index,halo_num_cell);
            else if ( 1 == logical_index )
                nc = grid.hasHalo(2*dim+1)
                     ? grid.haloNumEntity(
                         entity_type,dim,logical_index,halo_num_cell)
                     : 0;
            return nc;
        };

    // Compute the size of each neighbor. Note that we do not send to
    // ourselves (when each index is 0).
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 && k==0) )
                    counts.push_back( num_entity(Dim::I,i) *
                                      num_entity(Dim::J,j) *
                                      num_entity(Dim::K,k) );

    return counts;
}

//---------------------------------------------------------------------------//
// Multidimensional element count.
//---------------------------------------------------------------------------//
// Count the product of the extents at an individual entity in a grid field
template<class ViewType>
unsigned elementsPerEntity( const ViewType& view )
{
    return view.extent(3);
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
        pack_range[Dim::K].second - pack_range[Dim::K].first,
        field.extent(3) );

    // Get a view of the layer we are sending.
    auto field_subview = Kokkos::subview(
        field,
        pack_range[Dim::I], pack_range[Dim::J], pack_range[Dim::K],
        Kokkos::ALL );

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
template<typename ViewType,typename BufferType, typename PackRange>
void scatterNeighbor( const PackRange& unpack_range,
                      const BufferType receive_buffer,
                      const int offset,
                      ViewType field )
{
    // Declare an execution policy for the scatter update.
    using ExecPolicy =
        Kokkos::MDRangePolicy<typename ViewType::execution_space,
                              Kokkos::Rank<4> >;
    using point_type = typename ExecPolicy::point_type;

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
    point_type begin = {{0,0,0,0}};
    point_type end = {{unpack_range[Dim::I].second - unpack_range[Dim::I].first,
                       unpack_range[Dim::J].second - unpack_range[Dim::J].first,
                       unpack_range[Dim::K].second - unpack_range[Dim::K].first,
                       typename point_type::value_type (field.extent(3)) }};
    Kokkos::parallel_for(
        "Scatter negative neighbor update",
        ExecPolicy( begin, end ),
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

  \param halo_num_cell The number of halo cells over which to gather the data.

  \param tag tag Algorithm tag indicating a Cartesian topology communication.
*/
template<class GridFieldType>
void gather(
    GridFieldType& grid_field, const int halo_num_cell, CartCommFaceTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid.
    const auto& block = grid_field.block();
    if ( halo_num_cell > block.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor.
    std::vector<int> counts =
        neighborCounts( block, halo_num_cell, location, tag );

    // Compute the number of multidimensional elements in each entity.
    int md_size = elementsPerEntity( field );

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets via exclusive scan.
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
        // Initialize the pack range.
        std::array<Kokkos::pair<int,int>,3> pack_range;
        for ( int dim = 0; dim < 3; ++dim )
            pack_range[dim] =
                Kokkos::pair<int,int>(block.localEntityBegin(location,dim),
                                      block.localEntityEnd(location,dim));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our layer of local entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d),
                block.localEntityBegin(location,d) +
                block.haloNumEntity(location,d,-1,halo_num_cell) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our layer of local entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d) -
                block.haloNumEntity(location,d,1,halo_num_cell),
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

    // Do the communication.
    MPI_Neighbor_alltoallv(
        send_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        receive_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        grid_field.cartesianComm() );

    // Unpack the receive buffer.
    for ( int d = 0; d < 3; ++d )
    {
        // Initialize the unpack range.
        std::array<Kokkos::pair<int,int>,3> unpack_range;
        for ( int dim = 0; dim < 3; ++dim )
            unpack_range[dim] =
                Kokkos::pair<int,int>(block.localEntityBegin(location,dim),
                                      block.localEntityEnd(location,dim));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our halo entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.haloEntityBegin(location,d,-1,halo_num_cell),
                block.haloEntityEnd(location,d,-1,halo_num_cell) );

            // Unpack
            gatherNeighbor( unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our halo entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.haloEntityBegin(location,d,1,halo_num_cell),
                block.haloEntityEnd(location,d,1,halo_num_cell) );

            // Unpack
            gatherNeighbor(
                unpack_range, receive_buffer, offsets[2*d+1], field );
        }
    }
}

//---------------------------------------------------------------------------//
/*!
  \brief Gather field data from owning ranks into the halo using a 26-neighbor
  graph communication pattern.

  Communication in this pattern occurs over the entire halo of a specified
  width using the full 26 neighbor communication pattern. Neighbors are
  ordered in the 3x3 grid about the calling rank with the I index moving the
  fastest and the K index moving the slowest.

  \param grid_field The field to gather.

  \param halo_num_cell The number of halo cells over which to gather the data.

  \param tag tag Algorithm tag indicating a Graph topology communication.
*/
template<class GridFieldType>
void gather(
    GridFieldType& grid_field, const int halo_num_cell, GraphCommNodeTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid.
    const auto& block = grid_field.block();
    if ( halo_num_cell > block.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor. Note that
    // this includes neighbors we are not sending to. We will filter those out.
    std::vector<int> counts =
        neighborCounts( block, halo_num_cell, location, tag );

    // Compute the number of multidimensional elements in each entity.
    int md_size = elementsPerEntity( field );

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets via exclusive scan.
    int total_elements = 0;
    std::vector<int> offsets( 26, 0 );
    for ( int n = 0; n < 26; ++n )
    {
        offsets[n] = total_elements;
        total_elements += counts[n];
    }

    // Get a set of counts without zeros. We need to do this because the MPI
    // distributed graph implementation does not support null neighbor procs.
    std::vector<int> send_counts = counts;
    auto send_counts_end =
        std::remove( send_counts.begin(), send_counts.end(), 0 );
    send_counts.resize( std::distance(send_counts.begin(),send_counts_end) );

    // Compute send offsets via exclusive scan.
    int num_neighbor = send_counts.size();
    std::vector<int> send_offsets( num_neighbor, 0 );
    for ( int n = 1; n < num_neighbor; ++n )
    {
        send_offsets[n] = send_offsets[n-1] + send_counts[n-1];
    }

    // Allocate a send buffer.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        send_buffer( "gather_send", total_elements );

    // Set the pack range. We pack our local elements.
    auto dim_pack_range =
        [&]( const int dim, const int logical_index ){
            Kokkos::pair<int,int> range;
            if ( -1 == logical_index )
            {
                range.first = block.localEntityBegin( location, dim );
                range.second =
                    block.localEntityBegin( location, dim ) +
                    block.haloNumEntity(location,dim,logical_index,halo_num_cell);
            }
            else if ( 0 == logical_index )
            {
                range.first = block.localEntityBegin( location, dim );
                range.second = block.localEntityEnd( location, dim );
            }
            else if ( 1 == logical_index )
            {
                range.first =
                    block.localEntityEnd( location, dim ) -
                    block.haloNumEntity(location,dim,logical_index,halo_num_cell);
                range.second = block.localEntityEnd( location, dim );
            }
            return range;
        };

    // Pack the send buffer.
    int nid = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 & k==0) )
                {
                    if ( counts[nid] > 0 )
                    {
                        // Get the pack range.
                        std::array<Kokkos::pair<int,int>,3> pack_range =
                            {{ dim_pack_range(Dim::I,i),
                               dim_pack_range(Dim::J,j),
                               dim_pack_range(Dim::K,k) }};

                        // Pack.
                        packNeighbor(
                            pack_range, field, offsets[nid], send_buffer );
                    }

                    // Increment the neighbor count.
                    ++nid;
                }

    // Allocate a receive buffer. We receive as much as we send from our
    // opposite neighbors.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        receive_buffer( "gather_receive", total_elements );

    // Do the communication. Only do this if we actually have data to
    // send. The only time we will get a rank where this is the case is when
    // we have a comm size of 1 and there are no periodic boundaries. Again,
    // we need to do this here because the distributed graph communicator does
    // not handle null neighbor procs.
    if ( total_elements > 0 )
        MPI_Neighbor_alltoallv(
            send_buffer.data(), send_counts.data(), send_offsets.data(),
            MpiTraits<typename view_type::value_type>::type(),
            receive_buffer.data(), send_counts.data(), send_offsets.data(),
            MpiTraits<typename view_type::value_type>::type(),
            grid_field.graphComm() );

    // Set the unpack range. We unpack in the halo.
    auto dim_unpack_range =
        [&]( const int dim, const int logical_index ){
            Kokkos::pair<int,int> range(
                block.haloEntityBegin(location,dim,logical_index,halo_num_cell),
                block.haloEntityEnd(location,dim,logical_index,halo_num_cell) );
            return range;
        };

    // Unpack the receive buffer.
    nid = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 & k==0) )
                {
                    if ( counts[nid] > 0 )
                    {
                        // Get the unpack range.
                        std::array<Kokkos::pair<int,int>,3> unpack_range =
                            {{ dim_unpack_range(Dim::I,i),
                               dim_unpack_range(Dim::J,j),
                               dim_unpack_range(Dim::K,k) }};

                        // Unpack.
                        gatherNeighbor(
                            unpack_range, receive_buffer, offsets[nid], field );
                    }

                    // Increment the neighbor count.
                    ++nid;
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

  \param halo_num_cell The number of halo cells over which to scatter the
  data.

  \param tag tag Algorithm tag indicating a Cartesian topology communication.
*/
template<class GridFieldType>
void scatter(
    GridFieldType& grid_field, const int halo_num_cell, CartCommFaceTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid block.
    const auto& block = grid_field.block();
    if ( halo_num_cell > block.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor.
    std::vector<int> counts =
        neighborCounts( block, halo_num_cell, location, tag );

    // Compute the number of multidimensional elements in each entity.
    int md_size = elementsPerEntity( field );

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets via exclusive scan.
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

    // Pack the send buffer. We send our halo.
    for ( int d = 0; d < 3; ++d )
    {
        // Initialize the pack range.
        std::array<Kokkos::pair<int,int>,3> pack_range;
        for ( int dim = 0; dim < 3; ++dim )
            pack_range[dim] =
                Kokkos::pair<int,int>(block.localEntityBegin(location,dim),
                                      block.localEntityEnd(location,dim));

        // Negative Neighbor
        if ( counts[2*d] > 0 )
        {
            // We send our halo entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.haloEntityBegin(location,d,-1,halo_num_cell),
                block.haloEntityEnd(location,d,-1,halo_num_cell) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d], send_buffer );
        }

        // Positive Neighbor
        if ( counts[2*d+1] > 0 )
        {
            // We send our halo entities.
            pack_range[d] = Kokkos::pair<int,int>(
                block.haloEntityBegin(location,d,1,halo_num_cell),
                block.haloEntityEnd(location,d,1,halo_num_cell) );

            // Pack
            packNeighbor( pack_range, field, offsets[2*d+1], send_buffer );
        }
    }

    // Allocate a receive buffer. We receive as much as we send from our
    // opposite neighbors.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        receive_buffer( "scatter_receive", total_elements );

    // Do the communication.
    MPI_Neighbor_alltoallv(
        send_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        receive_buffer.data(), counts.data(), offsets.data(),
        MpiTraits<typename view_type::value_type>::type(),
        grid_field.cartesianComm() );

    // Unpack the receive buffer. We receive in our local.
    for ( int d = 0; d < 3; ++d )
    {
        std::array<Kokkos::pair<int,int>,3> unpack_range;
        for ( int dim = 0; dim < 3; ++dim )
            unpack_range[dim] =
                Kokkos::pair<int,int>(block.localEntityBegin(location,dim),
                                      block.localEntityEnd(location,dim));

        // Negative Neighbor.
        if ( counts[2*d] > 0 )
        {
            // We receive in our local entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityBegin(location,d),
                block.localEntityBegin(location,d) +
                block.haloNumEntity(location,d,-1,halo_num_cell) );

            // Unpack
            scatterNeighbor(
                unpack_range, receive_buffer, offsets[2*d], field );
        }

        // Positive Neighbor.
        if ( counts[2*d+1] > 0 )
        {
            // We receive in our local entities.
            unpack_range[d] = Kokkos::pair<int,int>(
                block.localEntityEnd(location,d) -
                block.haloNumEntity(location,d,1,halo_num_cell),
                block.localEntityEnd(location,d) );

            // Unpack
            scatterNeighbor(
                unpack_range, receive_buffer, offsets[2*d+1], field );
        }
    }
}

//---------------------------------------------------------------------------//
/*!
  \brief Scatter the data from the halo back to their owning rank using a
  26-neighbor graph communication pattern.

  Communication in this pattern occurs over the entire halo of a specified
  width using the full 26 neighbor communication pattern. Neighbors are
  ordered in the 3x3 grid about the calling rank with the I index moving the
  fastest and the K index moving the slowest.

  \param grid_field The field to scatter.

  \param halo_num_cell The number of halo cells over which to scatter the
  data.

  \param tag tag Algorithm tag indicating a Graph topology communication.
*/
template<class GridFieldType>
void scatter(
    GridFieldType& grid_field, const int halo_num_cell, GraphCommNodeTag tag )
{
    // Get the field.
    auto field = grid_field.data();
    using view_type = decltype(field);

    // Get the location of the field.
    auto location = grid_field.location();

    // Get the local grid.
    const auto& block = grid_field.block();
    if ( halo_num_cell > block.haloSize() )
        throw std::invalid_argument(
            "Requested halo size larger than block halo");

    // Count the number of entities we are sending to each neighbor. Note that
    // this includes neighbors we are not sending to. We will filter those
    // out.
    std::vector<int> counts =
        neighborCounts( block, halo_num_cell, location, tag );

    // Compute the number of multidimensional elements in each entity.
    int md_size = elementsPerEntity( field );

    // Scale the counts by the number of elements in each entity.
    for ( auto& c : counts ) c *= md_size;

    // Compute the total sends and offsets via exclusive scan.
    int total_elements = 0;
    std::vector<int> offsets( 26, 0 );
    for ( int n = 0; n < 26; ++n )
    {
        offsets[n] = total_elements;
        total_elements += counts[n];
    }

    // Get a set of counts without zeros. We need to do this because the MPI
    // distributed graph implementation does not support null neighbor procs.
    std::vector<int> send_counts = counts;
    auto send_counts_end =
        std::remove( send_counts.begin(), send_counts.end(), 0 );
    send_counts.resize( std::distance(send_counts.begin(),send_counts_end) );

    // Compute send offsets via exclusive scan.
    int num_neighbor = send_counts.size();
    std::vector<int> send_offsets( num_neighbor, 0 );
    for ( int n = 1; n < num_neighbor; ++n )
    {
        send_offsets[n] = send_offsets[n-1] + send_counts[n-1];
    }

    // Allocate a send buffer.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        send_buffer( "scatter_send", total_elements );

    // Set the pack range. We pack in the halo.
    auto dim_pack_range =
        [&]( const int dim, const int logical_index ){
            Kokkos::pair<int,int> range(
                block.haloEntityBegin(location,dim,logical_index,halo_num_cell),
                block.haloEntityEnd(location,dim,logical_index,halo_num_cell) );
            return range;
        };

    // Pack the send buffer.
    int nid = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 & k==0) )
                {
                    if ( counts[nid] > 0 )
                    {
                        // Get the pack range.
                        std::array<Kokkos::pair<int,int>,3> pack_range =
                            {{ dim_pack_range(Dim::I,i),
                               dim_pack_range(Dim::J,j),
                               dim_pack_range(Dim::K,k) }};

                        // Pack.
                        packNeighbor(
                            pack_range, field, offsets[nid], send_buffer );
                    }

                    // Increment the neighbor count.
                    ++nid;
                }

    // Allocate a receive buffer. We receive as much as we send from our
    // opposite neighbors.
    Kokkos::View<typename view_type::value_type*,
                 typename view_type::memory_space>
        receive_buffer( "scatter_receive", total_elements );

    // Do the communication. Only do this if we actually have data to
    // send. The only time we will get a rank where this is the case is when
    // we have a comm size of 1 and there are no periodic boundaries. Again,
    // we need to do this here because the distributed graph communicator does
    // not handle null neighbor procs.
    if ( total_elements > 0 )
        MPI_Neighbor_alltoallv(
            send_buffer.data(), send_counts.data(), send_offsets.data(),
            MpiTraits<typename view_type::value_type>::type(),
            receive_buffer.data(), send_counts.data(), send_offsets.data(),
            MpiTraits<typename view_type::value_type>::type(),
            grid_field.graphComm() );

    // Set the unpack range. We unpack our local elements.
    auto dim_unpack_range =
        [&]( const int dim, const int logical_index ){
            Kokkos::pair<int,int> range;
            if ( -1 == logical_index )
            {
                range.first = block.localEntityBegin( location, dim );
                range.second =
                    block.localEntityBegin( location, dim ) +
                    block.haloNumEntity(location,dim,logical_index,halo_num_cell);
            }
            else if ( 0 == logical_index )
            {
                range.first = block.localEntityBegin( location, dim );
                range.second = block.localEntityEnd( location, dim );
            }
            else if ( 1 == logical_index )
            {
                range.first =
                    block.localEntityEnd( location, dim ) -
                    block.haloNumEntity(location,dim,logical_index,halo_num_cell);
                range.second = block.localEntityEnd( location, dim );
            }
            return range;
        };

    // Unpack the receive buffer into the local data.
    nid = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !(i==0 && j==0 & k==0) )
                {
                    if ( counts[nid] > 0 )
                    {
                        // Get the unpack range.
                        std::array<Kokkos::pair<int,int>,3> unpack_range =
                            {{ dim_unpack_range(Dim::I,i),
                               dim_unpack_range(Dim::J,j),
                               dim_unpack_range(Dim::K,k) }};

                        // Unpack.
                        scatterNeighbor(
                            unpack_range, receive_buffer, offsets[nid], field );
                    }

                    // Increment the neighbor count.
                    ++nid;
                }
}

//---------------------------------------------------------------------------//

} // end namespace GridCommunication
} // end namespace Cajita

#endif // end CAJITA_GRIDCOMMUNICATION_HPP
