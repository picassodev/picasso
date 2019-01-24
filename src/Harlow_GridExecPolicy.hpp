#ifndef HARLOW_GRIDEXECPOLICY_HPP
#define HARLOW_GRIDEXECPOLICY_HPP

#include <Harlow_GridBlock.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <vector>
#include <exception>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Execution policy creators.
//---------------------------------------------------------------------------//
// Create a grid execution policy over all of the cells including the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createCellExecPolicy( const GridBlock& grid )
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
createNodeExecPolicy( const GridBlock& grid )
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
createLocalCellExecPolicy( const GridBlock& grid )
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
createLocalNodeExecPolicy( const GridBlock& grid )
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
createCellBoundaryExecPolicy( const GridBlock& grid,
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

    if ( PhysicalBoundary::LowX == boundary_id )
    {
        end[0] = 1;
    }

    else if ( PhysicalBoundary::HighX == boundary_id )
    {
        begin[0] = grid.numCell(Dim::I) - 1;
    }

    else if ( PhysicalBoundary::LowY == boundary_id )
    {
        end[1] = 1;
    }

    else if ( PhysicalBoundary::HighY == boundary_id )
    {
        begin[1] = grid.numCell(Dim::J) - 1;
    }

    else if ( PhysicalBoundary::LowZ == boundary_id )
    {
        end[2] = 1;
    }

    else if ( PhysicalBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.numCell(Dim::K) - 1;
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over nodes on a physical boundary including
// nodes in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createNodeBoundaryExecPolicy( const GridBlock& grid,
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

    if ( PhysicalBoundary::LowX == boundary_id )
    {
        end[0] = 1;
    }

    else if ( PhysicalBoundary::HighX == boundary_id )
    {
        begin[0] = grid.numNode(Dim::I) - 1;
    }

    else if ( PhysicalBoundary::LowY == boundary_id )
    {
        end[1] = 1;
    }

    else if ( PhysicalBoundary::HighY == boundary_id )
    {
        begin[1] = grid.numNode(Dim::J) - 1;
    }

    else if ( PhysicalBoundary::LowZ == boundary_id )
    {
        end[2] = 1;
    }

    else if ( PhysicalBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.numNode(Dim::K) - 1;
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over cells on a physical boundary. Does not
// include the cells in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalCellBoundaryExecPolicy( const GridBlock& grid,
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

    if ( PhysicalBoundary::LowX == boundary_id )
    {
        begin[0] = 0;
        end[0] = 1;
    }

    else if ( PhysicalBoundary::HighX == boundary_id )
    {
        begin[0] = grid.numCell(Dim::I) - 1;
        end[0] = grid.numCell(Dim::I);
    }

    else if ( PhysicalBoundary::LowY == boundary_id )
    {
        begin[1] = 0;
        end[1] = 1;
    }

    else if ( PhysicalBoundary::HighY == boundary_id )
    {
        begin[1] = grid.numCell(Dim::J) - 1;
        end[1] = grid.numCell(Dim::J);
    }

    else if ( PhysicalBoundary::LowZ == boundary_id )
    {
        begin[2] = 0;
        end[2] = 1;
    }

    else if ( PhysicalBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.numCell(Dim::K) - 1;
        end[2] = grid.numCell(Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over nodes on a physical boundary. Does not
// include the nodes in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalNodeBoundaryExecPolicy( const GridBlock& grid,
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

    if ( PhysicalBoundary::LowX == boundary_id )
    {
        begin[0] = 0;
        end[0] = 1;
    }

    else if ( PhysicalBoundary::HighX == boundary_id )
    {
        begin[0] = grid.numNode(Dim::I) - 1;
        end[0] = grid.numNode(Dim::I);
    }

    else if ( PhysicalBoundary::LowY == boundary_id )
    {
        begin[1] = 0;
        end[1] = 1;
    }

    else if ( PhysicalBoundary::HighY == boundary_id )
    {
        begin[1] = grid.numNode(Dim::J) - 1;
        end[1] = grid.numNode(Dim::J);
    }

    else if ( PhysicalBoundary::LowZ == boundary_id )
    {
        begin[2] = 0;
        end[2] = 1;
    }

    else if ( PhysicalBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.numNode(Dim::K) - 1;
        end[2] = grid.numNode(Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over cells in the halo of a given neighbor
// in the logical space of the global grid.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createHaloCellExecPolicy( const GridBlock& grid,
                          const std::vector<int>& neighbor_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    point_type begin;
    point_type end;

    for ( int d = 0; d < 3; ++d )
    {
        if ( LogicalBoundary::Negative == neighbor_id[d] )
        {
            begin[d] = 0;
            end[d] = grid.localCellBegin(d);
        }
        else if ( LogicalBoundary::Zero == neighbor_id[d] )
        {
            begin[d] = grid.localCellBegin(d);
            end[d] = grid.localCellEnd(d);
        }
        else if ( LogicalBoundary::Positive == neighbor_id[d] )
        {
            begin[d] = grid.localCellEnd(d);
            end[d] = grid.numCell(d);
        }
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over nodes in the halo of a given neighbor
// in logical space of the global grid.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createHaloNodeExecPolicy( const GridBlock& grid,
                          const std::vector<int>& neighbor_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    point_type begin;
    point_type end;

    for ( int d = 0; d < 3; ++d )
    {
        if ( LogicalBoundary::Negative == neighbor_id[d] )
        {
            begin[d] = 0;
            end[d] = grid.localNodeBegin(d);
        }
        else if ( LogicalBoundary::Zero == neighbor_id[d] )
        {
            begin[d] = grid.localNodeBegin(d);
            end[d] = grid.localNodeEnd(d);
        }
        else if ( LogicalBoundary::Positive == neighbor_id[d] )
        {
            begin[d] = grid.localNodeEnd(d);
            end[d] = grid.numNode(d);
        }
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GRIDEXECPOLICY_HPP
