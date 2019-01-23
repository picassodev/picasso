#ifndef HARLOW_STRUCTUREDGRIDBLOCK_HPP
#define HARLOW_STRUCTUREDGRIDBLOCK_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <vector>
#include <type_traits>
#include <exception>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Local structured grid interface.
//---------------------------------------------------------------------------//
class StructuredGridBlock
{
  public:

    // Get the physical coordinates of the low corner of the grid in a given
    // dimension. This low corner includes the halo region.
    virtual double lowCorner( const int dim ) const = 0;

    // Given a physical boundary id return if this grid is on that boundary.
    virtual bool onBoundary( const int boundary_id ) const = 0;

    // Get the cell size.
    virtual double cellSize() const = 0;

    // Get the inverse cell size.
    virtual double inverseCellSize() const = 0;

    // Get the halo size.
    virtual int haloSize() const = 0;

    // Get the number of nodes in a given dimension including the halo.
    virtual int numCell( const int dim ) const = 0;

    // Get the number of nodes in a given dimension including the halo.
    virtual int numNode( const int dim ) const = 0;

    // Get the beginning local cell index in a given direction. The local
    // cells do not include the halo.
    virtual int localCellBegin( const int dim ) const = 0;

    // Get the ending local cell index in a given direction. The local cells
    // do not include the halo.
    virtual int localCellEnd( const int dim ) const = 0;

    // Get the beginning local node index in a given direction. The local
    // nodes do not include the halo.
    virtual int localNodeBegin( const int dim ) const = 0;

    // Get the ending local node index in a given direction. The local nodes
    // do not include the halo.
    virtual int localNodeEnd( const int dim ) const = 0;
};

//---------------------------------------------------------------------------//
// Structured grid field type traits.
//---------------------------------------------------------------------------//
template<typename DataType,int Rank>
struct StructuredGridBlockFieldDataTypeImpl;

template<typename DataType>
struct StructuredGridBlockFieldDataTypeImpl<DataType,0>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    using type = value_type***;
};

template<typename DataType>
struct StructuredGridBlockFieldDataTypeImpl<DataType,1>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    using type = value_type***[extent_0];
};

template<typename DataType>
struct StructuredGridBlockFieldDataTypeImpl<DataType,2>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    static constexpr unsigned extent_1 = std::extent<DataType,1>::value;
    using type = value_type***[extent_0][extent_1];
};

template<typename DataType>
struct StructuredGridBlockFieldDataTypeImpl<DataType,3>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    static constexpr unsigned extent_1 = std::extent<DataType,1>::value;
    static constexpr unsigned extent_2 = std::extent<DataType,2>::value;
    using type = value_type***[extent_0][extent_1][extent_2];
};

template<typename DataType>
struct StructuredGridBlockFieldDataType
{
    static constexpr unsigned rank = std::rank<DataType>::value;
    using type =
        typename StructuredGridBlockFieldDataTypeImpl<DataType,rank>::type;
};

//---------------------------------------------------------------------------//
// Field creators
//---------------------------------------------------------------------------//
// Given a grid create a view of cell data with ijk indexing.
template<class DataType, class DeviceType>
Kokkos::View<typename StructuredGridBlockFieldDataType<DataType>::type,DeviceType>
createCellField( const StructuredGridBlock& grid,
                 const std::string& field_name = "" )
{
    return Kokkos::View<
        typename StructuredGridBlockFieldDataType<DataType>::type,DeviceType>(
            field_name,
            grid.numCell(Dim::I),
            grid.numCell(Dim::J),
            grid.numCell(Dim::K) );
}

//---------------------------------------------------------------------------//
// Given a grid create a view of node data with ijk indexing.
template<class DataType, class DeviceType>
Kokkos::View<typename StructuredGridBlockFieldDataType<DataType>::type,DeviceType>
createNodeField( const StructuredGridBlock& grid,
                 const std::string& field_name = "" )
{
    return Kokkos::View<
        typename StructuredGridBlockFieldDataType<DataType>::type,DeviceType>(
            field_name,
            grid.numNode(Dim::I),
            grid.numNode(Dim::J),
            grid.numNode(Dim::K) );
}

//---------------------------------------------------------------------------//
// Execution policy creators.
//---------------------------------------------------------------------------//
// Create a grid execution policy over all of the cells including the halo.
template<class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >
createCellExecPolicy( const StructuredGridBlock& grid )
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
createNodeExecPolicy( const StructuredGridBlock& grid )
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
createLocalCellExecPolicy( const StructuredGridBlock& grid )
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
createLocalNodeExecPolicy( const StructuredGridBlock& grid )
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
createCellBoundaryExecPolicy( const StructuredGridBlock& grid,
                              const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument(" not on given physical boundary");

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
createNodeBoundaryExecPolicy( const StructuredGridBlock& grid,
                              const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument(" not on given physical boundary");

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
createLocalCellBoundaryExecPolicy( const StructuredGridBlock& grid,
                                   const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument(" not on given physical boundary");

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
createLocalNodeBoundaryExecPolicy( const StructuredGridBlock& grid,
                                   const int boundary_id )
{
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3> >;
    using point_type = typename Policy::point_type;

    if ( !grid.onBoundary(boundary_id) )
        throw std::invalid_argument(" not on given physical boundary");

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
createHaloCellExecPolicy( const StructuredGridBlock& grid,
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
createHaloNodeExecPolicy( const StructuredGridBlock& grid,
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

#endif // end HARLOW_STRUCTUREDGRIDBLOCK_HPP
