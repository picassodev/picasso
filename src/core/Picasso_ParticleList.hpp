#ifndef PICASSO_PARTICLELIST_HPP
#define PICASSO_PARTICLELIST_HPP

#include <Picasso_ParticleCommunication.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_AdaptiveMesh.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cabana_Core.hpp>

#include <memory>
#include <type_traits>
#include <string>

namespace Picasso
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
// Particle. Wraps a tuple copy of a particle.
//---------------------------------------------------------------------------//
template<class ... FieldTags>
struct Particle
{
    using traits = ParticleTraits<FieldTags...>;
    using tuple_type = Cabana::Tuple<typename traits::member_types>;

    // Default constructor.
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
// Particle view. Wraps a view of the SoA the particle resides in.
//---------------------------------------------------------------------------//
template<int VectorLength, class ... FieldTags>
struct ParticleView
{
    using traits = ParticleTraits<FieldTags...>;
    using soa_type = Cabana::SoA<typename traits::member_types,VectorLength>;

    static constexpr int vector_length = VectorLength;

    // Default constructor.
    ParticleView() = default;

    // Tuple wrapper constructor.
    KOKKOS_FORCEINLINE_FUNCTION
    ParticleView( soa_type& soa, const int vector_index )
        : _soa( soa )
        , _vector_index( vector_index )
    {}

    // Get the underlying SoA.
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& soa() { return _soa; }

    KOKKOS_FORCEINLINE_FUNCTION
    const soa_type& soa() const { return _soa; }

    // Get the vector index of the particle in the SoA.
    KOKKOS_FORCEINLINE_FUNCTION
    int vectorIndex() const { return _vector_index; }

    // The soa the particle is in.
    soa_type& _soa;

    // The local vector index of the particle.
    int _vector_index;
};

//---------------------------------------------------------------------------//
// Particle accessor.
//---------------------------------------------------------------------------//
namespace ParticleAccess
{

//---------------------------------------------------------------------------//
// Particle accessor
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

//---------------------------------------------------------------------------//
// ParticleView accessor
template<class FieldTag,
         class ... FieldTags,
         class ... IndexTypes,
         int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION
typename ParticleView<
    VectorLength,FieldTags...>::soa_type::template member_value_type<
    FieldTagIndexer<FieldTag,FieldTags...>::index>
get( const ParticleView<VectorLength,FieldTags...>& particle,
     FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<FieldTagIndexer<FieldTag,FieldTags...>::index>(
        particle.soa(), particle.vectorIndex(), indices...);
}

template<class FieldTag,
         class ... FieldTags,
         class ... IndexTypes,
         int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION
typename ParticleView<
    VectorLength,FieldTags...>::soa_type::template member_reference_type<
    FieldTagIndexer<FieldTag,FieldTags...>::index>
get( ParticleView<VectorLength,FieldTags...>& particle,
     FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<FieldTagIndexer<FieldTag,FieldTags...>::index>(
        particle.soa(), particle.vectorIndex(), indices...);
}

//---------------------------------------------------------------------------//
// Get a view of a particle member as a vector. (Works for both Particle and
// ParticleView)
template<class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    LinearAlgebra::is_vector<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        &(Cabana::get(particle,tag,0)), 1 );
}

template<class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    LinearAlgebra::is_vector<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( const ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        const_cast<typename FieldTag::value_type*>(
            &(Cabana::get(particle,tag,0))), 1 );
}

// Get a view of a particle member as a matrix. (Works for both Particle and
// ParticleView)
template<class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    LinearAlgebra::is_matrix<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        &(Cabana::get(particle,tag,0,0)), 1 );
}

template<class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    LinearAlgebra::is_matrix<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( const ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        const_cast<typename FieldTag::value_type*>(
            &(Cabana::get(particle,tag,0,0))), 1 );
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

    using particle_view_type =
        ParticleView<aosoa_type::vector_length,FieldTags...>;

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

    // Redistribute particles to new owning grids. Return true if the
    // particles were actually redistributed.
    bool redistribute( const bool force_redistribute = false )
    {
        return ParticleCommunication::redistribute(
            *(_mesh->localGrid()),
            _mesh->minimumHaloWidth(),
            this->slice(Field::LogicalPosition()),
            _aosoa,
            force_redistribute );
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

} // end namespace Picasso

#endif // end PICASSO_PARTICLELIST_HPP
