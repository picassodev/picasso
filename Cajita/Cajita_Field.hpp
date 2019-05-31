#ifndef CAJITA_FIELD_HPP
#define CAJITA_FIELD_HPP

#include <Cajita_Types.hpp>
#include <Cajita_GlobalGrid.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <memory>
#include <string>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Field creator
// ---------------------------------------------------------------------------//
// Given a grid create a view of data with ijk indexing over the given entity
// type. The view is uninitialized.
template<class ValueType, class DeviceType>
Kokkos::View<ValueType****,DeviceType>
createField( const GridBlock& grid,
             const int num_component,
             const int field_location,
             const std::string& field_name = "" )
{
    return Kokkos::View<ValueType****,DeviceType>(
            Kokkos::ViewAllocateWithoutInitializing(field_name),
            grid.numEntity(field_location,Dim::I),
            grid.numEntity(field_location,Dim::J),
            grid.numEntity(field_location,Dim::K),
            num_component );
}

//---------------------------------------------------------------------------//
// Parallel grid field container.
//---------------------------------------------------------------------------//
template<class ValueType, class DeviceType>
class Field
{
  public:

    using value_type = ValueType;
    using device_type = DeviceType;
    using memory_space = typename device_type::memory_space;
    using execution_space = typename device_type::execution_space;
    using view_type = Kokkos::View<ValueType****,device_type>;

    Field( const std::shared_ptr<GlobalGrid>& global_grid,
               const int num_component,
               const int field_location,
               const int halo_cell_width,
               const std::string& field_name = "" )
        : _global_grid( global_grid )
        , _num_comp( num_component )
        , _field_location( field_location )
        , _name( field_name )
    {
        _block.assign( _global_grid->block(), halo_cell_width );

        _data = createField<ValueType,DeviceType>(
            _block, num_component, field_location, field_name );
    }

    // Get the global grid.
    const GlobalGrid& globalGrid() const
    { return *_global_grid; }

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

    // Get the number of field components.
    int numComp() const
    { return _num_comp; }

    // Get the field location.
    int location() const
    { return _field_location; }

    // Get the field name.
    const std::string& name() const
    { return _name; }

  private:

    std::shared_ptr<GlobalGrid> _global_grid;
    GridBlock _block;
    view_type _data;
    int _num_comp;
    int _field_location;
    std::string _name;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
class is_field : public std::false_type {};

template<class ValueType, class DeviceType>
class is_field<Field<ValueType,DeviceType> >
    : public std::true_type {};

template<class ValueType, class DeviceType>
class is_field<Field<const ValueType,DeviceType> >
    : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_FIELD_HPP
