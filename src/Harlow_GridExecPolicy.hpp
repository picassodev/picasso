#ifndef HARLOW_GRIDEXECPOLICY_HPP
#define HARLOW_GRIDEXECPOLICY_HPP

#include <Harlow_GridBlock.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <vector>
#include <exception>

namespace Harlow
{
namespace GridExecution
{
//---------------------------------------------------------------------------//
// Execution policy creators.
//---------------------------------------------------------------------------//
// Create a grid execution policy over all of the cells including the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createCellPolicy( const GridBlock& grid )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(MeshEntity::Cell,Dim::I),
                        grid.numEntity(MeshEntity::Cell,Dim::J),
                        grid.numEntity(MeshEntity::Cell,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over all of the nodes including the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createNodePolicy( const GridBlock& grid )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(MeshEntity::Node,Dim::I),
                        grid.numEntity(MeshEntity::Node,Dim::J),
                        grid.numEntity(MeshEntity::Node,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over the local cells (does not include the
// halo).
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalCellPolicy( const GridBlock& grid )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{ grid.localEntityBegin(MeshEntity::Cell,Dim::I),
                          grid.localEntityBegin(MeshEntity::Cell,Dim::J),
                          grid.localEntityBegin(MeshEntity::Cell,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(MeshEntity::Cell,Dim::I),
                        grid.localEntityEnd(MeshEntity::Cell,Dim::J),
                        grid.localEntityEnd(MeshEntity::Cell,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over the local nodes (does not include the
// halo).
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalNodePolicy( const GridBlock& grid )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{ grid.localEntityBegin(MeshEntity::Node,Dim::I),
                          grid.localEntityBegin(MeshEntity::Node,Dim::J),
                          grid.localEntityBegin(MeshEntity::Node,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(MeshEntity::Node,Dim::I),
                        grid.localEntityEnd(MeshEntity::Node,Dim::J),
                        grid.localEntityEnd(MeshEntity::Node,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over cells on a physical boundary including
// cells in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createBoundaryCellPolicy( const GridBlock& grid,
                          const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(MeshEntity::Cell,Dim::I),
                        grid.numEntity(MeshEntity::Cell,Dim::J),
                        grid.numEntity(MeshEntity::Cell,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(MeshEntity::Cell,Dim::I);
        end[0] = grid.localEntityBegin(MeshEntity::Cell,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(MeshEntity::Cell,Dim::I) - 1;
        end[0] = grid.localEntityEnd(MeshEntity::Cell,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(MeshEntity::Cell,Dim::J);
        end[1] = grid.localEntityBegin(MeshEntity::Cell,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(MeshEntity::Cell,Dim::J) - 1;
        end[1] = grid.localEntityEnd(MeshEntity::Cell,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(MeshEntity::Cell,Dim::K);
        end[2] = grid.localEntityBegin(MeshEntity::Cell,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(MeshEntity::Cell,Dim::K) - 1;
        end[2] = grid.localEntityEnd(MeshEntity::Cell,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over nodes on a physical boundary including
// nodes in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createBoundaryNodePolicy( const GridBlock& grid,
                          const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(MeshEntity::Node,Dim::I),
                        grid.numEntity(MeshEntity::Node,Dim::J),
                        grid.numEntity(MeshEntity::Node,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(MeshEntity::Node,Dim::I);
        end[0] = grid.localEntityBegin(MeshEntity::Node,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(MeshEntity::Node,Dim::I) - 1;
        end[0] = grid.localEntityEnd(MeshEntity::Node,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(MeshEntity::Node,Dim::J);
        end[1] = grid.localEntityBegin(MeshEntity::Node,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(MeshEntity::Node,Dim::J) - 1;
        end[1] = grid.localEntityEnd(MeshEntity::Node,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(MeshEntity::Node,Dim::K);
        end[2] = grid.localEntityBegin(MeshEntity::Node,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(MeshEntity::Node,Dim::K) - 1;
        end[2] = grid.localEntityEnd(MeshEntity::Node,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over cells on a physical boundary. Does not
// include the cells in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalBoundaryCellPolicy( const GridBlock& grid,
                               const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{ grid.localEntityBegin(MeshEntity::Cell,Dim::I),
                          grid.localEntityBegin(MeshEntity::Cell,Dim::J),
                          grid.localEntityBegin(MeshEntity::Cell,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(MeshEntity::Cell,Dim::I),
                        grid.localEntityEnd(MeshEntity::Cell,Dim::J),
                        grid.localEntityEnd(MeshEntity::Cell,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(MeshEntity::Cell,Dim::I);
        end[0] = grid.localEntityBegin(MeshEntity::Cell,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(MeshEntity::Cell,Dim::I) - 1;
        end[0] = grid.localEntityEnd(MeshEntity::Cell,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(MeshEntity::Cell,Dim::J);
        end[1] = grid.localEntityBegin(MeshEntity::Cell,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(MeshEntity::Cell,Dim::J) - 1;
        end[1] = grid.localEntityEnd(MeshEntity::Cell,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(MeshEntity::Cell,Dim::K);
        end[2] = grid.localEntityBegin(MeshEntity::Cell,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(MeshEntity::Cell,Dim::K) - 1;
        end[2] = grid.localEntityEnd(MeshEntity::Cell,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over nodes on a physical boundary. Does not
// include the nodes in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalBoundaryNodePolicy( const GridBlock& grid,
                               const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{ grid.localEntityBegin(MeshEntity::Node,Dim::I),
                          grid.localEntityBegin(MeshEntity::Node,Dim::J),
                          grid.localEntityBegin(MeshEntity::Node,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(MeshEntity::Node,Dim::I),
                        grid.localEntityEnd(MeshEntity::Node,Dim::J),
                        grid.localEntityEnd(MeshEntity::Node,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(MeshEntity::Node,Dim::I);
        end[0] = grid.localEntityBegin(MeshEntity::Node,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(MeshEntity::Node,Dim::I) - 1;
        end[0] = grid.localEntityEnd(MeshEntity::Node,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(MeshEntity::Node,Dim::J);
        end[1] = grid.localEntityBegin(MeshEntity::Node,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(MeshEntity::Node,Dim::J) - 1;
        end[1] = grid.localEntityEnd(MeshEntity::Node,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(MeshEntity::Node,Dim::K);
        end[2] = grid.localEntityBegin(MeshEntity::Node,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(MeshEntity::Node,Dim::K) - 1;
        end[2] = grid.localEntityEnd(MeshEntity::Node,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//

} // end namespace GridExecution
} // end namespace Harlow

#endif // end HARLOW_GRIDEXECPOLICY_HPP
