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
// Particle accessor.
//---------------------------------------------------------------------------//
// Particle accessor
template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Cabana::Particle<FieldTags...>::tuple_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const Cabana::Particle<FieldTags...>& particle, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.tuple(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Cabana::Particle<FieldTags...>::tuple_type::
        template member_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( Cabana::Particle<FieldTags...>& particle, FieldTag, IndexTypes... indices )
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
    typename Cabana::ParticleView<VectorLength, FieldTags...>::soa_type::
        template member_const_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( const Cabana::ParticleView<VectorLength, FieldTags...>& particle, FieldTag,
     IndexTypes... indices )
{
    return Cabana::get<TypeIndexer<FieldTag, FieldTags...>::index>(
        particle.soa(), particle.vectorIndex(), indices... );
}

template <class FieldTag, class... FieldTags, class... IndexTypes,
          int VectorLength>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    sizeof...( IndexTypes ) == FieldTag::rank,
    typename Cabana::ParticleView<VectorLength, FieldTags...>::soa_type::
        template member_reference_type<
            TypeIndexer<FieldTag, FieldTags...>::index>>::type
get( Cabana::ParticleView<VectorLength, FieldTags...>& particle, FieldTag,
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
        &( Cabana::get( particle, tag, 0 ) ), ParticleType::vector_length );
}

template <class ParticleType, class FieldTag>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    LinearAlgebra::is_vector<typename FieldTag::linear_algebra_type>::value,
    const typename FieldTag::linear_algebra_type>::type
get( const ParticleType& particle, FieldTag tag )
{
    return typename FieldTag::linear_algebra_type(
        const_cast<typename FieldTag::value_type*>(
            &( Cabana::get( particle, tag, 0 ) ) ),
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
        &( Cabana::get( particle, tag, 0, 0 ) ),
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
            &( Cabana::get( particle, tag, 0, 0 ) ) ),
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

    using traits = Cabana::ParticleTraits<FieldTags...>;

    using aosoa_type =
        Cabana::AoSoA<typename traits::member_types, memory_space>;

    using tuple_type = typename aosoa_type::tuple_type;

    template <std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    using particle_type = Cabana::Particle<FieldTags...>;

    using particle_view_type =
        Cabana::ParticleView<aosoa_type::vector_length, FieldTags...>;

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
                    Cabana::ParticleTraits<FieldTags...> )
{
    return std::make_shared<ParticleList<Mesh, FieldTags...>>( label, mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_PARTICLELIST_HPP
