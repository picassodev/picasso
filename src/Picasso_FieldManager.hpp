/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_FIELDMANAGER_HPP
#define PICASSO_FIELDMANAGER_HPP

#include <Picasso_AdaptiveMesh.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Create an array from a mesh and a field layout.
template <class Location, class FieldTag, class Mesh>
auto createArray( const Mesh& mesh, Location, FieldTag )
{
    auto array_layout = Cajita::createArrayLayout(
        mesh.localGrid(), FieldTag::size, typename Location::entity_type() );
    return Cajita::createArray<typename FieldTag::value_type,
                               typename Mesh::memory_space>( FieldTag::label(),
                                                             array_layout );
}

//---------------------------------------------------------------------------//
// Base field handle.
struct FieldHandleBase
{
    virtual ~FieldHandleBase() = default;
};

//---------------------------------------------------------------------------//
// Field handle.
template <class Location, class FieldTag, class Mesh>
struct FieldHandle : public FieldHandleBase
{
    std::shared_ptr<Cajita::Array<
        typename FieldTag::value_type, typename Location::entity_type,
        typename Mesh::cajita_mesh, typename Mesh::memory_space>>
        array;

    std::shared_ptr<Cajita::Halo<typename Mesh::memory_space>> halo;
};

//---------------------------------------------------------------------------//
// Field manager.
template <class Mesh>
class FieldManager
{
  public:
    using mesh_type = Mesh;

  public:
    // Non-adaptive mesh constructor.
    template <class M = Mesh>
    FieldManager(
        const std::shared_ptr<M>& mesh,
        typename std::enable_if<!is_adaptive_mesh<M>::value>::type* = 0 )
        : _mesh( mesh )
    {
    }

    // Adaptive mesh constructor.
    template <class M = Mesh>
    FieldManager(
        const std::shared_ptr<M>& mesh,
        typename std::enable_if<is_adaptive_mesh<M>::value>::type* = 0 )
        : _mesh( mesh )
    {
        // The mesh is adaptive so add the physical position of its nodes as a
        // field.
        auto key =
            createKey( FieldLocation::Node(),
                       Field::PhysicalPosition<mesh_type::num_space_dim>() );
        auto handle = std::make_shared<FieldHandle<
            FieldLocation::Node,
            Field::PhysicalPosition<mesh_type::num_space_dim>, Mesh>>();
        handle->array = _mesh->nodes();
        handle->halo =
            Cajita::createHalo<typename Field::PhysicalPosition<
                                   mesh_type::num_space_dim>::value_type,
                               typename Mesh::memory_space>(
                *( handle->array->layout() ),
                Cajita::NodeHaloPattern<Mesh::num_space_dim>() );
        _fields.emplace( key, handle );
    }

    // Get the mesh.
    std::shared_ptr<Mesh> mesh() const { return _mesh; }

    // Add a field. The field will be allocated and a halo created if it does
    // not already exist.
    template <class Location, class FieldTag>
    void add( const Location& location, const FieldTag& tag )
    {
        auto key = createKey( location, tag );
        if ( !_fields.count( key ) )
        {
            _fields.emplace( key, createFieldHandle( location, tag ) );
        }
    }

    // Add a field by layout.
    template <class Layout>
    void add( const Layout& )
    {
        add( typename Layout::location{}, typename Layout::tag{} );
    }

    // Get a shared pointer to a field array.
    template <class Location, class FieldTag>
    auto array( const Location& location, const FieldTag& tag ) const
    {
        return getFieldHandle( location, tag )->array;
    }

    // Get a shared pointer to a field array by layout.
    template <class Layout>
    auto array( const Layout& ) const
    {
        return array( typename Layout::location{}, typename Layout::tag{} );
    }

    // Get a view of a field.
    template <class Location, class FieldTag>
    auto view( const Location& location, const FieldTag& tag ) const
    {
        return array( location, tag )->view();
    }

    // Get a view of a field by layout.
    template <class Layout>
    auto view( const Layout& ) const
    {
        return view( typename Layout::location{}, typename Layout::tag{} );
    }

    // Scatter a field.
    template <class Location, class FieldTag>
    void scatter( const Location& location, const FieldTag& tag ) const
    {
        auto handle = getFieldHandle( location, tag );
        handle->halo->scatter(
            typename mesh_type::memory_space::execution_space(),
            Cajita::ScatterReduce::Sum(), *( handle->array ) );
    }

    // Scatter a field by layout.
    template <class Layout>
    void scatter( const Layout& ) const
    {
        scatter( typename Layout::location{}, typename Layout::tag{} );
    }

    // Gather a field.
    template <class Location, class FieldTag>
    void gather( const Location& location, const FieldTag& tag ) const
    {
        auto handle = getFieldHandle( location, tag );
        handle->halo->gather(
            typename mesh_type::memory_space::execution_space(),
            *( handle->array ) );
    }

    // Gather a field by layout.
    template <class Layout>
    void gather( const Layout& ) const
    {
        gather( typename Layout::location{}, typename Layout::tag{} );
    }

  private:
    // Create a key from a location and tag.
    template <class Location, class FieldTag>
    std::string createKey( Location, FieldTag ) const
    {
        return std::string( Location::label() + "_" + FieldTag::label() );
    }

    // Create a field handle from a location and tag.
    template <class Location, class FieldTag>
    std::shared_ptr<FieldHandle<Location, FieldTag, Mesh>>
    createFieldHandle( const Location& location, const FieldTag& tag ) const
    {
        auto handle = std::make_shared<FieldHandle<Location, FieldTag, Mesh>>();
        handle->array = createArray( *_mesh, location, tag );
        handle->halo = Cajita::createHalo<typename FieldTag::value_type,
                                          typename Mesh::memory_space>(
            *( handle->array->layout() ),
            Cajita::NodeHaloPattern<Mesh::num_space_dim>() );
        return handle;
    }

    // Get a field handle.
    template <class Location, class FieldTag>
    std::shared_ptr<FieldHandle<Location, FieldTag, Mesh>>
    getFieldHandle( const Location& location, const FieldTag& tag ) const
    {
        auto key = createKey( location, tag );
        if ( !_fields.count( key ) )
            throw std::runtime_error( key + " field doesn't exist" );
        return std::dynamic_pointer_cast<FieldHandle<Location, FieldTag, Mesh>>(
            _fields.find( key )->second );
    }

  private:
    std::shared_ptr<Mesh> _mesh;
    std::unordered_map<std::string, std::shared_ptr<FieldHandleBase>> _fields;
};

//---------------------------------------------------------------------------//
// Creation function.
template <class Mesh>
auto createFieldManager( const std::shared_ptr<Mesh>& mesh )
{
    return std::make_shared<FieldManager<Mesh>>( mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_FIELDMANAGER_HPP
