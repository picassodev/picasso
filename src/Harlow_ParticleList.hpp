#ifndef HARLOW_PARTICLELIST_HPP
#define HARLOW_PARTICLELIST_HPP

#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_FieldTypes.hpp>
#include <Harlow_AdaptiveMesh.hpp>
#include <Harlow_UniformMesh.hpp>

#include <Cabana_Core.hpp>

#include <memory>
#include <type_traits>
#include <string>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Field tag indexer.
//---------------------------------------------------------------------------//
template<class Tag, class ... FieldTags>
struct FieldTagIndexer;

//---------------------------------------------------------------------------//
// 1-field particle
template<class Tag0>
struct FieldTagIndexer<Tag0,Tag0>
{
    static constexpr std::size_t index = 0;
};

//---------------------------------------------------------------------------//
// 2-field particle
template<class Tag0,
         class Tag1>
struct FieldTagIndexer<Tag0,Tag0,Tag1>
{
    static constexpr std::size_t index = 0;
};

template<class Tag0,
         class Tag1>
struct FieldTagIndexer<Tag1,Tag0,Tag1>
{
    static constexpr std::size_t index = 1;
};

//---------------------------------------------------------------------------//
// 3-field particle
template<class Tag0,
         class Tag1,
         class Tag2>
struct FieldTagIndexer<Tag0,Tag0,Tag1,Tag2>
{
    static constexpr std::size_t index = 0;
};

template<class Tag0,
         class Tag1,
         class Tag2>
struct FieldTagIndexer<Tag1,Tag0,Tag1,Tag2>
{
    static constexpr std::size_t index = 1;
};

template<class Tag0,
         class Tag1,
         class Tag2>
struct FieldTagIndexer<Tag2,Tag0,Tag1,Tag2>
{
    static constexpr std::size_t index = 2;
};

//---------------------------------------------------------------------------//
// 4-field particle
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct FieldTagIndexer<Tag0,Tag0,Tag1,Tag2,Tag3>
{
    static constexpr std::size_t index = 0;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct FieldTagIndexer<Tag1,Tag0,Tag1,Tag2,Tag3>
{
    static constexpr std::size_t index = 1;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct FieldTagIndexer<Tag2,Tag0,Tag1,Tag2,Tag3>
{
    static constexpr std::size_t index = 2;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct FieldTagIndexer<Tag3,Tag0,Tag1,Tag2,Tag3>
{
    static constexpr std::size_t index = 3;
};

//---------------------------------------------------------------------------//
// 5-field particle
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct FieldTagIndexer<Tag0,Tag0,Tag1,Tag2,Tag3,Tag4>
{
    static constexpr std::size_t index = 0;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct FieldTagIndexer<Tag1,Tag0,Tag1,Tag2,Tag3,Tag4>
{
    static constexpr std::size_t index = 1;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct FieldTagIndexer<Tag2,Tag0,Tag1,Tag2,Tag3,Tag4>
{
    static constexpr std::size_t index = 2;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct FieldTagIndexer<Tag3,Tag0,Tag1,Tag2,Tag3,Tag4>
{
    static constexpr std::size_t index = 3;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct FieldTagIndexer<Tag4,Tag0,Tag1,Tag2,Tag3,Tag4>
{
    static constexpr std::size_t index = 4;
};

//---------------------------------------------------------------------------//
// 6-field particle
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag0,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 0;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag1,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 1;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag2,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 2;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag3,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 3;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag4,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 4;
};

template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct FieldTagIndexer<Tag5,Tag0,Tag1,Tag2,Tag3,Tag4,Tag5>
{
    static constexpr std::size_t index = 5;
};

//---------------------------------------------------------------------------//
// Particle Traits
//---------------------------------------------------------------------------//
template<class ... FieldTags>
struct ParticleTraits;

//---------------------------------------------------------------------------//
// 1-Field particle
template<class Tag0>
struct ParticleTraits<Tag0>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type>;
};

//---------------------------------------------------------------------------//
// 2-field particle.
template<class Tag0,
         class Tag1>
struct ParticleTraits<Tag0,
                      Tag1>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type>;
};

//---------------------------------------------------------------------------//
// 3-field particle.
template<class Tag0,
         class Tag1,
         class Tag2>
struct ParticleTraits<Tag0,
                      Tag1,
                      Tag2>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type>;
};

//---------------------------------------------------------------------------//
// 4-field particle.
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct ParticleTraits<Tag0,
                      Tag1,
                      Tag2,
                      Tag3>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type>;
};

//---------------------------------------------------------------------------//
// 5-field particle.
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct ParticleTraits<Tag0,
                      Tag1,
                      Tag2,
                      Tag3,
                      Tag4>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type,
                                             typename Tag4::data_type>;
};

//---------------------------------------------------------------------------//
// 6-field particle.
template<class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct ParticleTraits<Tag0,
                      Tag1,
                      Tag2,
                      Tag3,
                      Tag4,
                      Tag5>
{
    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type,
                                             typename Tag4::data_type,
                                             typename Tag5::data_type>;
};

//---------------------------------------------------------------------------//
// Particle
//---------------------------------------------------------------------------//
template<class ... FieldTags>
struct Particle
{
    using traits = ParticleTraits<FieldTags...>;
    using tuple_type = Cabana::Tuple<typename traits::member_types>;

    // Default constructor.
    KOKKOS_FORCEINLINE_FUNCTION
    Particle() = default;

    // Tuple wrapper constructor.
    KOKKOS_FORCEINLINE_FUNCTION
    Particle( const tuple_type& tuple )
        : _tuple( tuple )
    {}

    // Get the underlying tuple.
    KOKKOS_FORCEINLINE_FUNCTION
    tuple_type& tuple() { return _tuple; }

    KOKKOS_FORCEINLINE_FUNCTION
    const tuple_type& tuple() const { return _tuple; }

    // The tuple this particle wraps.
    tuple_type _tuple;
};

//---------------------------------------------------------------------------//
// Particle accessor.
//---------------------------------------------------------------------------//
namespace ParticleAccess
{

template<class FieldTag, class ... FieldTags, class ... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION
typename Particle<FieldTags...>::tuple_type::template member_value_type<
    FieldTagIndexer<FieldTag,FieldTags...>::index>
get( const Particle<FieldTags...>& particle, FieldTag, IndexTypes... indices )
{
    return Cabana::get<FieldTagIndexer<FieldTag,FieldTags...>::index>(
        particle.tuple(), indices...);
}

template<class FieldTag, class ... FieldTags, class ... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION
typename Particle<FieldTags...>::tuple_type::template member_reference_type<
    FieldTagIndexer<FieldTag,FieldTags...>::index>
get( Particle<FieldTags...>& particle, FieldTag, IndexTypes... indices )
{
    return Cabana::get<FieldTagIndexer<FieldTag,FieldTags...>::index>(
        particle.tuple(), indices...);
}

} // end namespace ParticleAccessor

//---------------------------------------------------------------------------//
// Particle List
//---------------------------------------------------------------------------//
template<class Mesh, class ... FieldTags>
class ParticleList
{
  public:

    using mesh_type = Mesh;

    using memory_space = typename Mesh::memory_space;

    using traits = ParticleTraits<FieldTags...>;

    using aosoa_type = Cabana::AoSoA<typename traits::member_types,memory_space>;

    using tuple_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    using particle_type = Particle<FieldTags...>;

    // Default constructor.
    ParticleList( const std::string& label,
                  const std::shared_ptr<Mesh>& mesh)
        : _aosoa(label)
        , _mesh(mesh)
    {}

    // Get the number of particles in the list.
    std::size_t size() const
    {
        return _aosoa.size();
    }

    // Get the AoSoA
    aosoa_type& aosoa() { return _aosoa; }
    const aosoa_type& aosoa() const { return _aosoa; }

    // Get the mesh.
    const Mesh& mesh() { return *_mesh; }

    // Get a slice of a given field.
    template<class FieldTag>
    slice_type<FieldTagIndexer<FieldTag,FieldTags...>::index>
    slice( FieldTag ) const
    {
        return Cabana::slice<FieldTagIndexer<FieldTag,FieldTags...>::index>(
            _aosoa, FieldTag::label() );
    }

    // Redistribute particles to new owning grids.
    void redistribute( const int minimum_halo_width )
    {
        ParticleCommunication::redistribute(
            *(_mesh->localGrid()),
            minimum_halo_width,
            this->slice(Field::LogicalPosition()),
            _aosoa );
    }

  private:

    aosoa_type _aosoa;
    std::shared_ptr<Mesh> _mesh;
};

//---------------------------------------------------------------------------//
// Creation function.
template<class Mesh, class ... FieldTags>
std::shared_ptr<ParticleList<Mesh,FieldTags...>>
createParticleList( const std::string& label,
                    const std::shared_ptr<Mesh>& mesh,
                    ParticleTraits<FieldTags...> )
{
    return std::make_shared<ParticleList<Mesh,FieldTags...>>( label, mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_PARTICLELIST_HPP
