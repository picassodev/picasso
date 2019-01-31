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
    point_type end = {{ grid.numCell(Dim::I),
                        grid.numCell(Dim::J),
                        grid.numCell(Dim::K) }};
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
    point_type end = {{ grid.numNode(Dim::I),
                        grid.numNode(Dim::J),
                        grid.numNode(Dim::K) }};
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
    point_type begin = {{ grid.localCellBegin(Dim::I),
                          grid.localCellBegin(Dim::J),
                          grid.localCellBegin(Dim::K) }};
    point_type end = {{ grid.localCellEnd(Dim::I),
                        grid.localCellEnd(Dim::J),
                        grid.localCellEnd(Dim::K) }};
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
    point_type begin = {{ grid.localNodeBegin(Dim::I),
                          grid.localNodeBegin(Dim::J),
                          grid.localNodeBegin(Dim::K) }};
    point_type end = {{ grid.localNodeEnd(Dim::I),
                        grid.localNodeEnd(Dim::J),
                        grid.localNodeEnd(Dim::K) }};
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
    point_type end = {{ grid.numCell(Dim::I),
                        grid.numCell(Dim::J),
                        grid.numCell(Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localCellBegin(Dim::I);
        end[0] = grid.localCellBegin(Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localCellEnd(Dim::I) - 1;
        end[0] = grid.localCellEnd(Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localCellBegin(Dim::J);
        end[1] = grid.localCellBegin(Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localCellEnd(Dim::J) - 1;
        end[1] = grid.localCellEnd(Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localCellBegin(Dim::K);
        end[2] = grid.localCellBegin(Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localCellEnd(Dim::K) - 1;
        end[2] = grid.localCellEnd(Dim::K);
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
    point_type end = {{ grid.numNode(Dim::I),
                        grid.numNode(Dim::J),
                        grid.numNode(Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localNodeBegin(Dim::I);
        end[0] = grid.localNodeBegin(Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localNodeEnd(Dim::I) - 1;
        end[0] = grid.localNodeEnd(Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localNodeBegin(Dim::J);
        end[1] = grid.localNodeBegin(Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localNodeEnd(Dim::J) - 1;
        end[1] = grid.localNodeEnd(Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localNodeBegin(Dim::K);
        end[2] = grid.localNodeBegin(Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localNodeEnd(Dim::K) - 1;
        end[2] = grid.localNodeEnd(Dim::K);
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

    point_type begin = {{ grid.localCellBegin(Dim::I),
                          grid.localCellBegin(Dim::J),
                          grid.localCellBegin(Dim::K) }};
    point_type end = {{ grid.localCellEnd(Dim::I),
                        grid.localCellEnd(Dim::J),
                        grid.localCellEnd(Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localCellBegin(Dim::I);
        end[0] = grid.localCellBegin(Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localCellEnd(Dim::I) - 1;
        end[0] = grid.localCellEnd(Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localCellBegin(Dim::J);
        end[1] = grid.localCellBegin(Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localCellEnd(Dim::J) - 1;
        end[1] = grid.localCellEnd(Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localCellBegin(Dim::K);
        end[2] = grid.localCellBegin(Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localCellEnd(Dim::K) - 1;
        end[2] = grid.localCellEnd(Dim::K);
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

    point_type begin = {{ grid.localNodeBegin(Dim::I),
                          grid.localNodeBegin(Dim::J),
                          grid.localNodeBegin(Dim::K) }};
    point_type end = {{ grid.localNodeEnd(Dim::I),
                        grid.localNodeEnd(Dim::J),
                        grid.localNodeEnd(Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localNodeBegin(Dim::I);
        end[0] = grid.localNodeBegin(Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localNodeEnd(Dim::I) - 1;
        end[0] = grid.localNodeEnd(Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localNodeBegin(Dim::J);
        end[1] = grid.localNodeBegin(Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localNodeEnd(Dim::J) - 1;
        end[1] = grid.localNodeEnd(Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localNodeBegin(Dim::K);
        end[2] = grid.localNodeBegin(Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localNodeEnd(Dim::K) - 1;
        end[2] = grid.localNodeEnd(Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//

} // end namespace GridExecution
} // end namespace Harlow

#endif // end HARLOW_GRIDEXECPOLICY_HPP
