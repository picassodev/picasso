#ifndef HARLOW_GRIDFIELD_HPP
#define HARLOW_GRIDFIELD_HPP

#include <Harlow_GridBlock.hpp>
#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Cartesian grid field type traits.
//---------------------------------------------------------------------------//
template<typename DataType,int Rank>
struct GridBlockFieldDataTypeImpl;

template<typename DataType>
struct GridBlockFieldDataTypeImpl<DataType,0>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    using type = value_type***;
};

template<typename DataType>
struct GridBlockFieldDataTypeImpl<DataType,1>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    using type = value_type***[extent_0];
};

template<typename DataType>
struct GridBlockFieldDataTypeImpl<DataType,2>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    static constexpr unsigned extent_1 = std::extent<DataType,1>::value;
    using type = value_type***[extent_0][extent_1];
};

template<typename DataType>
struct GridBlockFieldDataTypeImpl<DataType,3>
{
    using value_type = typename std::remove_all_extents<DataType>::type;
    static constexpr unsigned extent_0 = std::extent<DataType,0>::value;
    static constexpr unsigned extent_1 = std::extent<DataType,1>::value;
    static constexpr unsigned extent_2 = std::extent<DataType,2>::value;
    using type = value_type***[extent_0][extent_1][extent_2];
};

template<typename DataType>
struct GridBlockFieldDataType
{
    static constexpr unsigned rank = std::rank<DataType>::value;
    using type =
        typename GridBlockFieldDataTypeImpl<DataType,rank>::type;
};

//---------------------------------------------------------------------------//
// Field creators
//---------------------------------------------------------------------------//
// Given a grid create a view of cell data with ijk indexing.
template<class DataType, class DeviceType>
Kokkos::View<typename GridBlockFieldDataType<DataType>::type,DeviceType>
createCellField( const GridBlock& grid,
                 const std::string& field_name = "" )
{
    return Kokkos::View<
        typename GridBlockFieldDataType<DataType>::type,DeviceType>(
            field_name,
            grid.numCell(Dim::I),
            grid.numCell(Dim::J),
            grid.numCell(Dim::K) );
}

//---------------------------------------------------------------------------//
// Given a grid create a view of node data with ijk indexing.
template<class DataType, class DeviceType>
Kokkos::View<typename GridBlockFieldDataType<DataType>::type,DeviceType>
createNodeField( const GridBlock& grid,
                 const std::string& field_name = "" )
{
    return Kokkos::View<
        typename GridBlockFieldDataType<DataType>::type,DeviceType>(
            field_name,
            grid.numNode(Dim::I),
            grid.numNode(Dim::J),
            grid.numNode(Dim::K) );
}

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GRIDFIELD_HPP
