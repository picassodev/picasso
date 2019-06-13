#ifndef CAJITA_ARRAY_HPP
#define CAJITA_ARRAY_HPP

#include <Cajita_Block.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Array layout.
//---------------------------------------------------------------------------//
template<class EntityType>
class ArrayLayout
{
  public:

    // Entity type.
    using entity_type = EntityType;

    /*!
      \brief Constructor.
      \param block The grid block over which the layout will be constructed.
      \param dofs_per_entity The number of degrees-of-freedom per grid entity.
    */
    ArrayLayout( const std::shared_ptr<Block>& block,
                 const int dofs_per_entity )
        : _block( block )
        , _dofs_per_entity( dofs_per_entity )
    {}

    // Get the grid block over which this layout is defined.
    const Block& block() const
    { return *_block; }

    // Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const
    { return _dofs_per_entity; }

    // Get the local index space of the owned array elements.
    IndexSpace<4> ownedIndexSpace() const
    {
        return appendDimension( _block->ownedIndexSpace(EntityType()),
                                _dofs_per_entity );
    }

    // Get the local index space of the owned and ghosted array elements.
    IndexSpace<4> ghostedIndexSpace() const
    {
        return appendDimension( _block->ghostedIndexSpace(EntityType()),
                                _dofs_per_entity );
    }

    // Given a relative set of indices of a neighbor get the set of local
    // indices we own that we share with that neighbor to use as ghosts.
    IndexSpace<4> sharedOwnedIndexSpace( const int off_i,
                                         const int off_j,
                                         const int off_k ) const
    {
        return appendDimension(
            _block->sharedOwnedIndexSpace(EntityType(),off_i,off_j,off_k),
            _dofs_per_entity );
    }

    // Given a relative set of indices of a neighbor get set of local indices
    // owned by that neighbor that are shared with us to use as ghosts.
    IndexSpace<4> sharedGhostedIndexSpace( const int off_i,
                                           const int off_j,
                                           const int off_k ) const
    {
        return appendDimension(
            _block->sharedGhostedIndexSpace(EntityType(),off_i,off_j,off_k),
            _dofs_per_entity );
    }

  private:

    std::shared_ptr<Block> _block;
    int _dofs_per_entity;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a block.
  \param block The grid block over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template<class EntityType>
std::shared_ptr<ArrayLayout<EntityType>>
createArrayLayout( const std::shared_ptr<Block>& block,
                   const int dofs_per_entity,
                   EntityType )
{
    return std::make_shared<ArrayLayout<EntityType>>( block, dofs_per_entity );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a grid given block
  parameters. An intermediate block will be created and assigned to the
  layout.
  \param block The grid block over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template<class EntityType>
std::shared_ptr<ArrayLayout<EntityType>>
createArrayLayout( const std::shared_ptr<GlobalGrid>& global_grid,
                   const int halo_cell_width,
                   const int dofs_per_entity,
                   EntityType )
{
    return std::make_shared<ArrayLayout<EntityType>>(
        createBlock(global_grid,halo_cell_width), dofs_per_entity );
}

//---------------------------------------------------------------------------//
// Array
//---------------------------------------------------------------------------//
template<class Scalar, class EntityType, class DeviceType>
class Array
{
  public:

    // Value type.
    using value_type = Scalar;

    // Entity type.
    using entity_type = EntityType;

    // Device type.
    using device_type = DeviceType;

    // Array layout type.
    using array_layout = ArrayLayout<entity_type>;

    // View type.
    using view_type = Kokkos::View<value_type****,device_type>;

    /*!
      \brief Create an array with the given layout. Arrays are constructed
      over the ghosted index space of the layout.
      \param label A label for the array.
      \param layout The array layout over which to construct the view.
    */
    Array( const std::string& label,
           const std::shared_ptr<array_layout>& layout )
        : _layout( layout )
        , _data(
            createView<value_type,device_type>(label,layout->ghostedIndexSpace()) )
    {}

    //! Get the layout of the array.
    const array_layout& layout() const
    { return *_layout; }

    //! Get a view of the array data.
    view_type view() const
    { return _data; }

    //! Get the array label.
    std::string label() const
    { return _data.label(); }

  private:

    std::shared_ptr<array_layout> _layout;
    view_type _data;
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
 */
template<class Scalar, class DeviceType, class EntityType>
std::shared_ptr<Array<Scalar,EntityType,DeviceType>>
createArray( const std::string& label,
             const std::shared_ptr<ArrayLayout<EntityType>>& layout )
{
    return std::make_shared<Array<Scalar,EntityType,DeviceType>>(
        label, layout );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita.

#endif // end CAJITA_ARRAY_HPP
