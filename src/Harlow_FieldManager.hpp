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
// Field layout.
template<class FieldTag, class Location>
class FieldLayout
{
    using field_tag = FieldTag;
    using field_location = Location;
};

//---------------------------------------------------------------------------//
// Create an array from a mesh and a field layout.
template<class MemorySpace, class Layout, class Mesh>
std::shared_ptr<Cajita::Array<typename Layout::field_tag::value_type,
                              typename Layout::field_location::entity_type,
                              Cajita::UniformMesh<double>,
                              MemorySpace>>
createArray( const Mesh& mesh, Layout )
{
    auto array_layout = Cajita::createArrayLayout(
        mesh.localGrid(), Layout::field_tag::dim,
        typename Layout::field_location::entity_type() );
    return Cajita::createArray<double,MemorySpace>(
        Layout::field_tag::label(), array_layout );
}

//---------------------------------------------------------------------------//
// Field manager.
template<class Mesh>
class FieldManager
{
  public:

    using mesh_type = Mesh;

    using memory_space = typename MeshType::memory_space;

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

    using halo = Cajita::Halo<double,MemorySpace>;

  public:

    // Constructor.
    FieldManager( const std::shared_ptr<Mesh>& mesh )
        : _mesh( mesh )
    {
        // If the mesh is adaptive add the physical position of its nodes as a
        // field.
        if ( is_adaptive_mesh<Mesh>::value )
        {
            _node_fields.emplace( Field::PhysicalPosition::label(),
                                  _mesh->nodes() );
            _node_halo = Cajita::createHalo( *(_mesh->nodes()->layout()),
                                             Cajita::FullHaloPattern() );
        }
    }

    // Get the mesh.
    const Mesh& mesh() const
    {
        return _mesh;
    }

    // Add a node field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Node>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _node_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _node_halo )
        {
            _node_halo = Cajita::createHalo( *(array->layout()),
                                             Cajita::FullHaloPattern() );
        }
    }

    // Add a cell field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Cell>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _cell_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _cell_halo )
        {
            _cell_halo = Cajita::createHalo( *(array->layout()),
                                             Cajita::FullHaloPattern() );
        }
    }

    // Add a I-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::I>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _face_i_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _face_i_halo )
        {
            _face_i_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Add a J-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::J>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _face_j_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _face_j_halo )
        {
            _face_j_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Add a K-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::K>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _face_k_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _face_k_halo )
        {
            _face_k_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Add a I-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::I>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _edge_i_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _edge_i_halo )
        {
            _edge_i_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Add a J-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::J>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _edge_j_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _edge_j_halo )
        {
            _edge_j_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Add a K-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::K>>::value,
                            void>::type
    add( Layout )
    {
        auto array = createArray(_mesh,Layout() );
        _edge_k_fields.emplace( Layout::field_tag::label(), array );
        if ( nullptr == _edge_k_halo )
        {
            _edge_k_halo = Cajita::createHalo( *(array->layout()),
                                               Cajita::FullHaloPattern() );
        }
    }

    // Get a node field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Node>::value,
                            std::shared_ptr<node_array>>::type
    array( Layout ) const
    {
        return _node_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a cell field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Cell>::value,
                            std::shared_ptr<cell_array>>::type
    array( Layout ) const
    {
        return _cell_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a I-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::I>>::value,
                            std::shared_ptr<face_i_array>>::type
    array( Layout ) const
    {
        return _face_i_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a J-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::J>>::value,
                            std::shared_ptr<face_j_array>>::type
    array( Layout ) const
    {
        return _face_j_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a K-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::K>>::value,
                            std::shared_ptr<face_k_array>>::type
    array( Layout ) const
    {
        return _face_k_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a I-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::I>>::value,
                            std::shared_ptr<edge_i_array>>::type
    array( Layout ) const
    {
        return _edge_i_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a J-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::J>>::value,
                            std::shared_ptr<edge_j_array>>::type
    array( Layout ) const
    {
        return _edge_j_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a K-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::K>>::value,
                            std::shared_ptr<edge_k_array>>::type
    array( Layout ) const
    {
        return _edge_k_fields.find(
            Layout::field_tag::label() )->second;
    }

    // Get a view of a node field.
    template<class Layout>
    view_type view( Layout ) const
    {
        return array( Layout() )->view();
    }

    // Scatter a node field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Node>::value,
                            void>::type
    scatter( Layout ) const
    {
        _node_halo->scatter( *array(Layout()) );
    }

    // Scatter a cell field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Cell>::value,
                            void>::type
    scatter( Layout ) const
    {
        _cell_halo->scatter( *array(Layout()) );
    }

    // Scatter a I-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::I>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _face_i_halo->scatter( *array(Layout()) );
    }

    // Scatter a J-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::J>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _face_j_halo->scatter( *array(Layout()) );
    }

    // Scatter a K-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::K>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _face_k_halo->scatter( *array(Layout()) );
    }

    // Scatter a I-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::I>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _edge_i_halo->scatter( *array(Layout()) );
    }

    // Scatter a J-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::J>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _edge_j_halo->scatter( *array(Layout()) );
    }

    // Scatter a K-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::K>>::value,
                            void>::type
    scatter( Layout ) const
    {
        _edge_k_halo->scatter( *array(Layout()) );
    }

    // Gather a node field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Node>::value,
                            void>::type
    gather( Layout ) const
    {
        _node_halo->gather( *array(Layout()) );
    }

    // Gather a cell field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Cell>::value,
                            void>::type
    gather( Layout ) const
    {
        _cell_halo->gather( *array(Layout()) );
    }

    // Gather a I-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::I>>::value,
                            void>::type
    gather( Layout ) const
    {
        _face_i_halo->gather( *array(Layout()) );
    }

    // Gather a J-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::J>>::value,
                            void>::type
    gather( Layout ) const
    {
        _face_j_halo->gather( *array(Layout()) );
    }

    // Gather a K-face field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Face<Dim::K>>::value,
                            void>::type
    gather( Layout ) const
    {
        _face_k_halo->gather( *array(Layout()) );
    }

    // Gather a I-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::I>>::value,
                            void>::type
    gather( Layout ) const
    {
        _edge_i_halo->gather( *array(Layout()) );
    }

    // Gather a J-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::J>>::value,
                            void>::type
    gather( Layout ) const
    {
        _edge_j_halo->gather( *array(Layout()) );
    }

    // Gather a K-edge field.
    template<class Layout>
    typename std::enable_if<std::is_same<
                                typename Layout::field_location::entity_type,
                                Cajita::Edge<Dim::K>>::value,
                            void>::type
    gather( Layout ) const
    {
        _edge_k_halo->gather( *array(Layout()) );
    }

  private:

    std::shared_ptr<Mesh> _mesh;
    std::unordered_map<std::string,std::shared_ptr<node_array>> _node_fields;
    std::unordered_map<std::string,std::shared_ptr<cell_array>> _cell_fields;
    std::unordered_map<std::string,std::shared_ptr<face_i_array>> _face_i_fields;
    std::unordered_map<std::string,std::shared_ptr<face_j_array>> _face_j_fields;
    std::unordered_map<std::string,std::shared_ptr<face_k_array>> _face_k_fields;
    std::unordered_map<std::string,std::shared_ptr<edge_i_array>> _edge_i_fields;
    std::unordered_map<std::string,std::shared_ptr<edge_j_array>> _edge_j_fields;
    std::unordered_map<std::string,std::shared_ptr<edge_k_array>> _edge_k_fields;
    std::shared_ptr<halo> _node_halo;
    std::shared_ptr<halo> _cell_halo;
    std::shared_ptr<halo> _face_i_halo;
    std::shared_ptr<halo> _face_j_halo;
    std::shared_ptr<halo> _face_k_halo;
    std::shared_ptr<halo> _edge_i_halo;
    std::shared_ptr<halo> _edge_j_halo;
    std::shared_ptr<halo> _edge_k_halo;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_FIELDMANAGER_HPP
