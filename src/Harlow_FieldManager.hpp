#ifndef HARLOW_FIELDMANAGER_HPP
#define HARLOW_FIELDMANAGER_HPP

#include <Harlow_Types.hpp>
#include <Harlow_ParticleList.hpp>
#include <Harlow_AdaptiveMesh.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <type_traits>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Field locations.
namespace FieldLocation
{
struct Cell
{
    using entity_type = Cajita::Cell;
};

template<int D>
struct Face
{
    using entity_type = Cajita::Face<D>;
};

template<int D>
struct Edge
{
    using entity_type = Cajita::Edge<D>;
};

struct Node
{
    using entity_type = Cajita::Node;
};

}

//---------------------------------------------------------------------------//
// Create an array from a mesh and a field layout.
template<class Location, class FieldTag, class Mesh>
auto createArray( const Mesh& mesh, Location, FieldTag ) ->
    std::shared_ptr<Cajita::Array<typename FieldTag::value_type,
                                  typename Location::entity_type,
                                  Cajita::UniformMesh<double>,
                                  typename Mesh::memory_space>>

{
    auto array_layout = Cajita::createArrayLayout(
        mesh.localGrid(), FieldTag::dim,
        typename Location::entity_type() );
    return Cajita::createArray<double,typename Mesh::memory_space>(
        FieldTag::label(), array_layout );
}

//---------------------------------------------------------------------------//
// Field manager.
template<class Mesh>
class FieldManager
{
  public:

    using mesh_type = Mesh;

    using memory_space = typename Mesh::memory_space;

    using node_array = Cajita::Array<double,
                                     Cajita::Node,
                                     Cajita::UniformMesh<double>,
                                     memory_space>;

    using cell_array = Cajita::Array<double,
                                     Cajita::Cell,
                                     Cajita::UniformMesh<double>,
                                     memory_space>;

    using face_i_array = Cajita::Array<double,
                                       Cajita::Face<Dim::I>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using face_j_array = Cajita::Array<double,
                                       Cajita::Face<Dim::J>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using face_k_array = Cajita::Array<double,
                                       Cajita::Face<Dim::K>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using edge_i_array = Cajita::Array<double,
                                       Cajita::Edge<Dim::I>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using edge_j_array = Cajita::Array<double,
                                       Cajita::Edge<Dim::J>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using edge_k_array = Cajita::Array<double,
                                       Cajita::Edge<Dim::K>,
                                       Cajita::UniformMesh<double>,
                                       memory_space>;

    using view_type = Kokkos::View<double****,memory_space>;

    using halo_type = Cajita::Halo<double,memory_space>;

  public:

    // Uniform Mesh constructor.
    template<class M = Mesh>
    FieldManager(
        const std::shared_ptr<M>& mesh,
        typename std::enable_if<is_uniform_mesh<M>::value>::type* = 0 )
        : _mesh( mesh )
    {}

    // Adaptive mesh constructor.
    template<class M = Mesh>
    FieldManager(
        const std::shared_ptr<M>& mesh,
        typename std::enable_if<is_adaptive_mesh<M>::value>::type* = 0 )
        : _mesh( mesh )
    {
        // If the mesh is adaptive add the physical position of its nodes as a
        // field.
        _node_fields.emplace( Field::PhysicalPosition::label(),
                              _mesh->nodes() );
        _node_halo = Cajita::createHalo<double,memory_space>(
            *(_mesh->nodes()->layout()), Cajita::FullHaloPattern() );
    }

    // Get the mesh.
    const Mesh& mesh() const
    {
        return _mesh;
    }

    // Add a node field.
    template<class FieldTag>
    void add( FieldLocation::Node, FieldTag )
    {
        if ( !_node_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Node(), FieldTag() );
            _node_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _node_halo )
            {
                _node_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a cell field.
    template<class FieldTag>
    void add( FieldLocation::Cell, FieldTag )
    {
        if ( !_cell_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Cell(), FieldTag() );
            _cell_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _cell_halo )
            {
                _cell_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a I-face field.
    template<class FieldTag>
    void add( FieldLocation::Face<Dim::I>, FieldTag )
    {
        if ( !_face_i_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Face<Dim::I>(), FieldTag() );
            _face_i_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _face_i_halo )
            {
                _face_i_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a J-face field.
    template<class FieldTag>
    void add( FieldLocation::Face<Dim::J>, FieldTag )
    {
        if ( !_face_j_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Face<Dim::J>(), FieldTag() );
            _face_j_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _face_j_halo )
            {
                _face_j_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a K-face field.
    template<class FieldTag>
    void add( FieldLocation::Face<Dim::K>, FieldTag )
    {
        if ( !_face_k_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Face<Dim::K>(), FieldTag() );
            _face_k_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _face_k_halo )
            {
                _face_k_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a I-edge field.
    template<class FieldTag>
    void add( FieldLocation::Edge<Dim::I>, FieldTag )
    {
        if ( !_edge_i_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Edge<Dim::I>(), FieldTag() );
            _edge_i_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _edge_i_halo )
            {
                _edge_i_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a J-edge field.
    template<class FieldTag>
    void add( FieldLocation::Edge<Dim::J>, FieldTag )
    {
        if ( !_edge_j_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Edge<Dim::J>(), FieldTag() );
            _edge_j_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _edge_j_halo )
            {
                _edge_j_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Add a K-edge field.
    template<class FieldTag>
    void add( FieldLocation::Edge<Dim::K>, FieldTag )
    {
        if ( !_edge_k_fields.count(FieldTag::label()) )
        {
            auto array = createArray(
                *_mesh, FieldLocation::Edge<Dim::K>(), FieldTag() );
            _edge_k_fields.emplace( FieldTag::label(), array );
            if ( nullptr == _edge_k_halo )
            {
                _edge_k_halo = Cajita::createHalo<double,memory_space>(
                    *(array->layout()), Cajita::FullHaloPattern() );
            }
        }
    }

    // Get a node field.
    template<class FieldTag>
    std::shared_ptr<node_array>
    array( FieldLocation::Node, FieldTag ) const
    {
        if ( !_node_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " node field doesn't exist" );
        return _node_fields.find( FieldTag::label() )->second;
    }

    // Get a cell field.
    template<class FieldTag>
    std::shared_ptr<cell_array>
    array( FieldLocation::Cell, FieldTag ) const
    {
        if ( !_cell_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " cell field doesn't exist" );
        return _cell_fields.find( FieldTag::label() )->second;
    }

    // Get a I-face field.
    template<class FieldTag>
    std::shared_ptr<face_i_array>
    array( FieldLocation::Face<Dim::I>, FieldTag ) const
    {
        if ( !_face_i_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " I-face field doesn't exist" );
        return _face_i_fields.find( FieldTag::label() )->second;
    }

    // Get a J-face field.
    template<class FieldTag>
    std::shared_ptr<face_j_array>
    array( FieldLocation::Face<Dim::J>, FieldTag ) const
    {
        if ( !_face_j_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " J-face field doesn't exist" );
        return _face_j_fields.find( FieldTag::label() )->second;
    }

    // Get a K-face field.
    template<class FieldTag>
    std::shared_ptr<face_k_array>
    array( FieldLocation::Face<Dim::K>, FieldTag ) const
    {
        if ( !_face_k_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " K-face field doesn't exist" );
        return _face_k_fields.find( FieldTag::label() )->second;
    }

    // Get a I-edge field.
    template<class FieldTag>
    std::shared_ptr<edge_i_array>
    array( FieldLocation::Edge<Dim::I>, FieldTag ) const
    {
        if ( !_edge_i_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " I-edge field doesn't exist" );
        return _edge_i_fields.find( FieldTag::label() )->second;
    }

    // Get a J-edge field.
    template<class FieldTag>
    std::shared_ptr<edge_j_array>
    array( FieldLocation::Edge<Dim::J>, FieldTag ) const
    {
        if ( !_edge_j_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " J-edge field doesn't exist" );
        return _edge_j_fields.find( FieldTag::label() )->second;
    }

    // Get a K-edge field.
    template<class FieldTag>
    std::shared_ptr<edge_k_array>
    array( FieldLocation::Edge<Dim::K>, FieldTag ) const
    {
        if ( !_edge_k_fields.count(FieldTag::label()) )
            throw std::runtime_error(
                FieldTag::label() + " K-edge field doesn't exist" );
        return _edge_k_fields.find( FieldTag::label() )->second;
    }

    // Get a view of a field with the given layout.
    template<class Location, class FieldTag>
    view_type view( Location, FieldTag ) const
    {
        return array( Location(), FieldTag() )->view();
    }

    // Scatter a node field.
    template<class FieldTag>
    void scatter( FieldLocation::Node, FieldTag ) const
    {
        _node_halo->scatter( *array(FieldLocation::Node(),FieldTag()) );
    }

    // Scatter a cell field.
    template<class FieldTag>
    void scatter( FieldLocation::Cell, FieldTag ) const
    {
        _cell_halo->scatter( *array(FieldLocation::Cell(),FieldTag()) );
    }

    // Scatter a I-face field.
    template<class FieldTag>
    void scatter( FieldLocation::Face<Dim::I>, FieldTag ) const
    {
        _face_i_halo->scatter(
            *array(FieldLocation::Face<Dim::I>(),FieldTag()) );
    }

    // Scatter a J-face field.
    template<class FieldTag>
    void scatter( FieldLocation::Face<Dim::J>, FieldTag ) const
    {
        _face_j_halo->scatter(
            *array(FieldLocation::Face<Dim::J>(),FieldTag()) );
    }

    // Scatter a K-face field.
    template<class FieldTag>
    void scatter( FieldLocation::Face<Dim::K>, FieldTag ) const
    {
        _face_k_halo->scatter(
            *array(FieldLocation::Face<Dim::K>(),FieldTag()) );
    }

    // Scatter a I-edge field.
    template<class FieldTag>
    void scatter( FieldLocation::Edge<Dim::I>, FieldTag ) const
    {
        _edge_i_halo->scatter(
            *array(FieldLocation::Edge<Dim::I>(),FieldTag()) );
    }

    // Scatter a J-edge field.
    template<class FieldTag>
    void scatter( FieldLocation::Edge<Dim::J>, FieldTag ) const
    {
        _edge_j_halo->scatter(
            *array(FieldLocation::Edge<Dim::J>(),FieldTag()) );
    }

    // Scatter a K-edge field.
    template<class FieldTag>
    void scatter( FieldLocation::Edge<Dim::K>, FieldTag ) const
    {
        _edge_k_halo->scatter(
            *array(FieldLocation::Edge<Dim::K>(),FieldTag()) );
    }

    // Gather a node field.
    template<class FieldTag>
    void gather( FieldLocation::Node, FieldTag ) const
    {
        _node_halo->gather( *array(FieldLocation::Node(),FieldTag()) );
    }

    // Gather a cell field.
    template<class FieldTag>
    void gather( FieldLocation::Cell, FieldTag ) const
    {
        _cell_halo->gather( *array(FieldLocation::Cell(),FieldTag()) );
    }

    // Gather a I-face field.
    template<class FieldTag>
    void gather( FieldLocation::Face<Dim::I>, FieldTag ) const
    {
        _face_i_halo->gather(
            *array(FieldLocation::Face<Dim::I>(),FieldTag()) );
    }

    // Gather a J-face field.
    template<class FieldTag>
    void gather( FieldLocation::Face<Dim::J>, FieldTag ) const
    {
        _face_j_halo->gather(
            *array(FieldLocation::Face<Dim::J>(),FieldTag()) );
    }

    // Gather a K-face field.
    template<class FieldTag>
    void gather( FieldLocation::Face<Dim::K>, FieldTag ) const
    {
        _face_k_halo->gather(
            *array(FieldLocation::Face<Dim::K>(),FieldTag()) );
    }

    // Gather a I-edge field.
    template<class FieldTag>
    void gather( FieldLocation::Edge<Dim::I>, FieldTag ) const
    {
        _edge_i_halo->gather(
            *array(FieldLocation::Edge<Dim::I>(),FieldTag()) );
    }

    // Gather a J-edge field.
    template<class FieldTag>
    void gather( FieldLocation::Edge<Dim::J>, FieldTag ) const
    {
        _edge_j_halo->gather(
            *array(FieldLocation::Edge<Dim::J>(),FieldTag()) );
    }

    // Gather a K-edge field.
    template<class FieldTag>
    void gather( FieldLocation::Edge<Dim::K>, FieldTag ) const
    {
        _edge_k_halo->gather(
            *array(FieldLocation::Edge<Dim::K>(),FieldTag()) );
    }

  private:

    std::shared_ptr<Mesh> _mesh;
    std::unordered_map<
        std::string,std::shared_ptr<node_array>> _node_fields;
    std::unordered_map<
        std::string,std::shared_ptr<cell_array>> _cell_fields;
    std::unordered_map<
        std::string,std::shared_ptr<face_i_array>> _face_i_fields;
    std::unordered_map<
        std::string,std::shared_ptr<face_j_array>> _face_j_fields;
    std::unordered_map<
        std::string,std::shared_ptr<face_k_array>> _face_k_fields;
    std::unordered_map<
        std::string,std::shared_ptr<edge_i_array>> _edge_i_fields;
    std::unordered_map<
        std::string,std::shared_ptr<edge_j_array>> _edge_j_fields;
    std::unordered_map<
        std::string,std::shared_ptr<edge_k_array>> _edge_k_fields;
    std::shared_ptr<halo_type> _node_halo;
    std::shared_ptr<halo_type> _cell_halo;
    std::shared_ptr<halo_type> _face_i_halo;
    std::shared_ptr<halo_type> _face_j_halo;
    std::shared_ptr<halo_type> _face_k_halo;
    std::shared_ptr<halo_type> _edge_i_halo;
    std::shared_ptr<halo_type> _edge_j_halo;
    std::shared_ptr<halo_type> _edge_k_halo;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_FIELDMANAGER_HPP
