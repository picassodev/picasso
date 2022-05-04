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

#ifndef PICASSO_PARTICLELIST_HPP
#define PICASSO_PARTICLELIST_HPP

#include <Picasso_AdaptiveMesh.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <memory>
#include <string>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Particle Traits
//---------------------------------------------------------------------------//
template <class... FieldTags>
struct ParticleTraits
{
    using member_types = Cabana::MemberTypes<typename FieldTags::data_type...>;
};

//---------------------------------------------------------------------------//
// Particle copy. Wraps a tuple copy of a particle.
//---------------------------------------------------------------------------//
template <class... FieldTags>
struct Particle
{
    using traits = ParticleTraits<FieldTags...>;
    using tuple_type = Cabana::Tuple<typename traits::member_types>;

    static constexpr int vector_length = 1;

    // Default constructor.
    Particle() = default;

    // Tuple wrapper constructor.
    KOKKOS_FORCEINLINE_FUNCTION
    Particle( const tuple_type& tuple )
        : _tuple( tuple )
    {
    }

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
template <int VectorLength, class... FieldTags>
struct ParticleView
{
    using traits = ParticleTraits<FieldTags...>;
    using soa_type = Cabana::SoA<typename traits::member_types, VectorLength>;

    static constexpr int vector_length = VectorLength;

    // Default constructor.
    ParticleView() = default;

    // Tuple wrapper constructor.
    KOKKOS_FORCEINLINE_FUNCTION
    ParticleView( soa_type& soa, const int vector_index )
        : _soa( soa )
        , _vector_index( vector_index )
    {
    }

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
// Particle accessor
template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Particle<FieldTags...>::tuple_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const Particle<FieldTags...>& particle, FieldTag, IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.tuple(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Particle<FieldTags...>::tuple_type::template member_reference_type<
        TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( Particle<FieldTags...>& particle, FieldTag, IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.tuple(), indices... );
}

//---------------------------------------------------------------------------//
// ParticleView accessor
template <class FieldTag, class... FieldTags, class... IndexTypes,
          int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename ParticleView<VectorLength, FieldTags...>::soa_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const ParticleView<VectorLength, FieldTags...>& particle, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.soa(), particle.vectorIndex(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes,
          int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename ParticleView<VectorLength, FieldTags...>::soa_type::
        template member_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( ParticleView<VectorLength, FieldTags...>& particle, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.soa(), particle.vectorIndex(), indices... );
}

//---------------------------------------------------------------------------//
// Get a view of a particle member as a vector. (Works for both Particle
// and ParticleView)
template <class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    LinearAlgebra::is_vector<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        &( get( particle, tag, 0 ) ), ParticleType::vector_length );
}

template <class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    LinearAlgebra::is_vector<typename FieldTag::linear_algebra_type>::value,
    const typename FieldTag::linear_algebra_type>::type
get( const ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        const_cast<typename FieldTag::value_type*>(
            &( get( particle, tag, 0 ) ) ),
        ParticleType::vector_length );
}

//---------------------------------------------------------------------------//
// Get a view of a particle member as a matrix. (Works for both Particle
// and ParticleView)
template <class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    LinearAlgebra::is_matrix<typename FieldTag::linear_algebra_type>::value,
    typename FieldTag::linear_algebra_type>::type
get( ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        &( get( particle, tag, 0, 0 ) ),
        ParticleType::vector_length * FieldTag::dim1,
        ParticleType::vector_length );
}

template <class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    LinearAlgebra::is_matrix<typename FieldTag::linear_algebra_type>::value,
    const typename FieldTag::linear_algebra_type>::type
get( const ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        const_cast<typename FieldTag::value_type*>(
            &( get( particle, tag, 0, 0 ) ) ),
        ParticleType::vector_length * FieldTag::dim1,
        ParticleType::vector_length );
}

//---------------------------------------------------------------------------//
// Particle List
//---------------------------------------------------------------------------//
template <class Mesh, class... FieldTags>
class ParticleList
{
  public:
    using mesh_type = Mesh;

    using memory_space = typename Mesh::memory_space;

    using traits = ParticleTraits<FieldTags...>;

    using aosoa_type =
        Cabana::AoSoA<typename traits::member_types, memory_space>;

    using tuple_type = typename aosoa_type::tuple_type;

    template <std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    using particle_type = Particle<FieldTags...>;

    using particle_view_type =
        ParticleView<aosoa_type::vector_length, FieldTags...>;

    // Default constructor.
    ParticleList( const std::string& label, const std::shared_ptr<Mesh>& mesh )
        : _aosoa( label )
        , _mesh( mesh )
        , _label( label )
    {
    }

    // Get the number of particles in the list.
    std::size_t size() const { return _aosoa.size(); }

    // Get the AoSoA
    aosoa_type& aosoa() { return _aosoa; }
    const aosoa_type& aosoa() const { return _aosoa; }

    // Get the mesh.
    const Mesh& mesh() { return *_mesh; }

    // Get the label
    const std::string& label() { return _label; }
    const std::string& label() const { return _label; }

    // Get a slice of a given field.
    template <class FieldTag>
    slice_type<TypeIndexer<FieldTag, FieldTags...>::index>
    slice( FieldTag ) const
    {
        return Cabana::slice<TypeIndexer<FieldTag, FieldTags...>::index>(
            _aosoa, FieldTag::label() );
    }

    // Redistribute particles to new owning grids. Return true if the
    // particles were actually redistributed.
    bool redistribute( const bool force_redistribute = false )
    {
        return Cajita::particleGridMigrate(
            *( _mesh->localGrid() ),
            this->slice( Field::LogicalPosition<mesh_type::num_space_dim>() ),
            _aosoa, _mesh->minimumHaloWidth(), force_redistribute );
    }

  private:
    aosoa_type _aosoa;
    std::shared_ptr<Mesh> _mesh;
    std::string _label;
};

//---------------------------------------------------------------------------//
// Creation function.
template <class Mesh, class... FieldTags>
std::shared_ptr<ParticleList<Mesh, FieldTags...>>
createParticleList( const std::string& label, const std::shared_ptr<Mesh>& mesh,
                    ParticleTraits<FieldTags...> )
{
    return std::make_shared<ParticleList<Mesh, FieldTags...>>( label, mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_PARTICLELIST_HPP
