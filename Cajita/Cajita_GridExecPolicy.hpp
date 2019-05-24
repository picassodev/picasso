#ifndef CAJITA_GRIDEXECPOLICY_HPP
#define CAJITA_GRIDEXECPOLICY_HPP

#include <Cajita_GridBlock.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <vector>
#include <exception>

namespace Cajita
{
namespace GridExecution
{
//---------------------------------------------------------------------------//
// Execution policy creators.
// ---------------------------------------------------------------------------//
// Create a grid execution policy over all of the entities of the given type
// including the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createEntityPolicy( const GridBlock& grid, const int entity_type )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(entity_type,Dim::I),
                        grid.numEntity(entity_type,Dim::J),
                        grid.numEntity(entity_type,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over the local entities (does not include
// the halo).
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalEntityPolicy( const GridBlock& grid, const int entity_type )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;
    point_type begin = {{ grid.localEntityBegin(entity_type,Dim::I),
                          grid.localEntityBegin(entity_type,Dim::J),
                          grid.localEntityBegin(entity_type,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(entity_type,Dim::I),
                        grid.localEntityEnd(entity_type,Dim::J),
                        grid.localEntityEnd(entity_type,Dim::K) }};
    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over entities on a physical boundary
// including cells in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createBoundaryEntityPolicy( const GridBlock& grid,
                            const int entity_type,
                            const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{0,0,0}};
    point_type end = {{ grid.numEntity(entity_type,Dim::I),
                        grid.numEntity(entity_type,Dim::J),
                        grid.numEntity(entity_type,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(entity_type,Dim::I);
        end[0] = grid.localEntityBegin(entity_type,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(entity_type,Dim::I) - 1;
        end[0] = grid.localEntityEnd(entity_type,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(entity_type,Dim::J);
        end[1] = grid.localEntityBegin(entity_type,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(entity_type,Dim::J) - 1;
        end[1] = grid.localEntityEnd(entity_type,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(entity_type,Dim::K);
        end[2] = grid.localEntityBegin(entity_type,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(entity_type,Dim::K) - 1;
        end[2] = grid.localEntityEnd(entity_type,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//
// Create a grid execution policy over entities on a physical boundary. Does
// not include the cells in the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createLocalBoundaryEntityPolicy( const GridBlock& grid,
                                 const int entity_type,
                                 const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument("Block not on given physical boundary");

    point_type begin = {{ grid.localEntityBegin(entity_type,Dim::I),
                          grid.localEntityBegin(entity_type,Dim::J),
                          grid.localEntityBegin(entity_type,Dim::K) }};
    point_type end = {{ grid.localEntityEnd(entity_type,Dim::I),
                        grid.localEntityEnd(entity_type,Dim::J),
                        grid.localEntityEnd(entity_type,Dim::K) }};

    if ( DomainBoundary::LowX == boundary_id )
    {
        begin[0] = grid.localEntityBegin(entity_type,Dim::I);
        end[0] = grid.localEntityBegin(entity_type,Dim::I) + 1;
    }

    else if ( DomainBoundary::HighX == boundary_id )
    {
        begin[0] = grid.localEntityEnd(entity_type,Dim::I) - 1;
        end[0] = grid.localEntityEnd(entity_type,Dim::I);
    }

    else if ( DomainBoundary::LowY == boundary_id )
    {
        begin[1] = grid.localEntityBegin(entity_type,Dim::J);
        end[1] = grid.localEntityBegin(entity_type,Dim::J) + 1;
    }

    else if ( DomainBoundary::HighY == boundary_id )
    {
        begin[1] = grid.localEntityEnd(entity_type,Dim::J) - 1;
        end[1] = grid.localEntityEnd(entity_type,Dim::J);
    }

    else if ( DomainBoundary::LowZ == boundary_id )
    {
        begin[2] = grid.localEntityBegin(entity_type,Dim::K);
        end[2] = grid.localEntityBegin(entity_type,Dim::K) + 1;
    }

    else if ( DomainBoundary::HighZ == boundary_id )
    {
        begin[2] = grid.localEntityEnd(entity_type,Dim::K) - 1;
        end[2] = grid.localEntityEnd(entity_type,Dim::K);
    }

    return Policy( begin, end );
}

//---------------------------------------------------------------------------//

} // end namespace GridExecution
} // end namespace Cajita

#endif // end CAJITA_GRIDEXECPOLICY_HPP
