#ifndef HARLOW_GRIDFIELD_HPP
#define HARLOW_GRIDFIELD_HPP

#include <Harlow_GlobalGrid.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <memory>

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
// Field creator
// ---------------------------------------------------------------------------//
// Given a grid create a view of data with ijk indexing over the given entity
// type. The view is uninitialized.
template<class DataType, class DeviceType>
Kokkos::View<typename GridBlockFieldDataType<DataType>::type,DeviceType>
createField( const GridBlock& grid,
             const int field_location,
             const std::string& field_name = "" )
{
    return Kokkos::View<
        typename GridBlockFieldDataType<DataType>::type,DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing(field_name),
            grid.numEntity(field_location,Dim::I),
            grid.numEntity(field_location,Dim::J),
            grid.numEntity(field_location,Dim::K) );
}

//---------------------------------------------------------------------------//
// Parallel grid field container.
//---------------------------------------------------------------------------//
template<class DataType, class DeviceType>
class GridField
{
  public:

    using data_type = DataType;
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;
    using view_data_type = typename GridBlockFieldDataType<DataType>::type;
    using view_type = Kokkos::View<view_data_type,device_type>;
    using value_type = typename view_type::value_type;

    GridField( const std::shared_ptr<GlobalGrid>& global_grid,
               const int field_location,
               const int halo_cell_width,
               const std::string& field_name = "" )
        : _global_grid( global_grid )
        , _field_location( field_location )
    {
        _block.assign( _global_grid->block(), halo_cell_width );

        _data = createField<DataType,DeviceType>( _block, field_location, field_name );
    }

    // Get the grid communicator.
    MPI_Comm comm() const
    { return _global_grid->comm(); }

    // Get the grid communicator with a 6-neighbor Cartesian
    // topology. Neighbors are ordered in this topology as
    // {-I,+I,-J,+J,-K,+K}.
    MPI_Comm cartesianComm() const
    { return _global_grid->cartesianComm(); }

    // Get the grid communicator with a 26-neighbor graph topology. Neighbors
    // are logically ordered in the 3x3 grid about centered on the local rank
    // with the I index moving the fastest and the K index moving the slowest.
    MPI_Comm graphComm() const
    { return _global_grid->graphComm(); }

    // Get the grid.
    const GridBlock& block() const
    { return _block; }

    // Get the local field data.
    view_type data() const
    { return _data; }

    // Get the field location.
    int location() const
    { return _field_location; }

  private:

    std::shared_ptr<GlobalGrid> _global_grid;
    GridBlock _block;
    view_type _data;
    int _field_location;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
class is_grid_field : public std::false_type {};

template<class DataType, class DeviceType>
class is_grid_field<GridField<DataType,DeviceType> >
    : public std::true_type {};

template<class DataType, class DeviceType>
class is_grid_field<GridField<const DataType,DeviceType> >
    : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_GRIDFIELD_HPP
