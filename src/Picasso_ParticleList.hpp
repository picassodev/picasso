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
#include <Cabana_Grid.hpp>

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

} // end namespace Picasso

#endif // end PICASSO_PARTICLELIST_HPP
