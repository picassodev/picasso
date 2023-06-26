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

#ifndef PICASSO_PARTICLELEVELSET_HPP
#define PICASSO_PARTICLELEVELSET_HPP

#include <Picasso_FieldManager.hpp>
#include <Picasso_LevelSet.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <ArborX.hpp>

#include <cfloat>

//---------------------------------------------------------------------------//
// ArborX Data
//---------------------------------------------------------------------------//
// Search primitives. We build the tree from particles of the color we want
// the level set for.
namespace Picasso
{
template <class CoordinateSlice, class ViewType>
struct ParticleLevelSetPrimitiveData
{
    using memory_space = typename CoordinateSlice::memory_space;
    using size_type = typename CoordinateSlice::size_type;
    CoordinateSlice x;
    ViewType c;
    int num_color;
};

// Predicate storage - store the the 3d index of the mesh entity we are going to
// search the tree with along with its coordinates.
template <class IndexType>
struct ParticleLevelSetPredicateStorage
{
    using size_type = IndexType;
    size_type i;
    size_type j;
    size_type k;
    float x;
    float y;
    float z;
};

// Search predicates. We search the tree with the mesh entities on which we
// want to build the level set.
template <class LocalMesh, class EntityType>
struct ParticleLevelSetPredicateData
{
    using memory_space = typename LocalMesh::memory_space;
    using size_type = int;
    using entity_type = EntityType;
    LocalMesh local_mesh;
    size_type size;
    size_type i_size;
    size_type ij_size;

    template <class LocalGrid>
    ParticleLevelSetPredicateData( const LocalMesh& lm,
                                   const LocalGrid& local_grid )
        : local_mesh( lm )
    {
        auto ghost_entities = local_grid.indexSpace(
            Cajita::Ghost(), entity_type(), Cajita::Local() );
        size = ghost_entities.size();
        i_size = ghost_entities.extent( Dim::I );
        ij_size = i_size * ghost_entities.extent( Dim::J );
    }

    KOKKOS_FUNCTION void
    convertIndexTo3d( size_type i,
                      ParticleLevelSetPredicateStorage<size_type>& ijk ) const
    {
        ijk.k = i / ij_size;
        size_type k_size = ijk.k * ij_size;
        ijk.j = ( i - k_size ) / i_size;
        ijk.i = i - ijk.j * i_size - k_size;
    }
};

//---------------------------------------------------------------------------//
// Query callback. When the query occurs we store the distance to the closest
// point. We don't use the tree graph so we don't insert anything into the
// graph.
template <class CoordinateSlice, class DistanceEstimateView>
struct ParticleLevelSetCallback
{
    ParticleLevelSetPrimitiveData<
        CoordinateSlice,
        Kokkos::View<int*, typename CoordinateSlice::memory_space>>
        primitive_data;
    DistanceEstimateView distance_estimate;
    float radius;

    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        // Get the actual index of the particle.
        auto p = primitive_data.c( primitive_index );

        // Get the predicate storage.
        auto storage = getData( predicate );

        // Compute the distance from the grid entity to the particle sphere.
        float dx = static_cast<float>( primitive_data.x( p, 0 ) ) - storage.x;
        float dy = static_cast<float>( primitive_data.x( p, 1 ) ) - storage.y;
        float dz = static_cast<float>( primitive_data.x( p, 2 ) ) - storage.z;
        distance_estimate( storage.i, storage.j, storage.k, 0 ) =
            sqrt( dx * dx + dy * dy + dz * dz ) - radius;
    }
};

} // end namespace Picasso

//---------------------------------------------------------------------------//
// ArborX traits.
namespace ArborX
{

// Create the primitives we build the tree from. These are the particle
// coordinates of the color we build the level set for.
template <class CoordinateSlice, class ViewType>
struct AccessTraits<
    Picasso::ParticleLevelSetPrimitiveData<CoordinateSlice, ViewType>,
    PrimitivesTag>
{
    using primitive_data =
        Picasso::ParticleLevelSetPrimitiveData<CoordinateSlice, ViewType>;
    using memory_space = typename primitive_data::memory_space;
    using size_type = typename primitive_data::size_type;
    static size_type size( const primitive_data& data )
    {
        return data.num_color;
    }
    static KOKKOS_FUNCTION Point get( const primitive_data& data, size_type i )
    {
        // Get the actual index of the particle.
        auto p = data.c( i );

        // Return a point made from the particles.
        return { static_cast<float>( data.x( p, 0 ) ),
                 static_cast<float>( data.x( p, 1 ) ),
                 static_cast<float>( data.x( p, 2 ) ) };
    }
};

// Create the predicates we search the tree with. These are the mesh entities
// on which we build the level set.
template <class LocalMesh, class EntityType>
struct AccessTraits<
    Picasso::ParticleLevelSetPredicateData<LocalMesh, EntityType>,
    PredicatesTag>
{
    using predicate_data =
        Picasso::ParticleLevelSetPredicateData<LocalMesh, EntityType>;
    using entity_type = typename predicate_data::entity_type;
    using memory_space = typename predicate_data::memory_space;
    using size_type = typename predicate_data::size_type;
    static size_type size( const predicate_data& data ) { return data.size; }
    static KOKKOS_FUNCTION auto get( const predicate_data& data, size_type i )
    {
        // Get the entity index.
        Picasso::ParticleLevelSetPredicateStorage<size_type> storage;
        data.convertIndexTo3d( i, storage );
        int index[3] = { storage.i, storage.j, storage.k };

        // Get the coordinates of the entity.
        double x[3];
        data.local_mesh.coordinates( entity_type(), index, x );
        storage.x = x[0];
        storage.y = x[1];
        storage.z = x[2];

        // Find the nearest particle to the entity. Attach the entity index to
        // use in the callback.
        return attach( nearest( Point{ storage.x, storage.y, storage.z }, 1 ),
                       storage );
    }
};

} // end namespace ArborX

//---------------------------------------------------------------------------//
namespace Picasso
{
//---------------------------------------------------------------------------//
// Particle level set. Composes a signed distance function for particles of a
// given color.
template <class MeshType, class SignedDistanceLocation>
class ParticleLevelSet
{
  public:
    using mesh_type = MeshType;
    using memory_space = typename mesh_type::memory_space;
    using location_type = SignedDistanceLocation;
    using entity_type = typename location_type::entity_type;
    using halo_type = Cajita::Halo<memory_space>;
    using level_set = LevelSet<MeshType, SignedDistanceLocation>;

    /*!
      \brief Construct the level set with particles of a given color. If the
      input color is negative then all particles will be used in the level set
      regardless of color.
      \param inputs Level set settings.
      \param mesh The mesh over which to build the signed distnace function.
      \param color The particle color over which to build the level set. Use
      -1 if the level set is to be built over all particles.
    */
    ParticleLevelSet( const nlohmann::json inputs,
                      const std::shared_ptr<MeshType>& mesh, const int color )
        : _color( color )
        , _ls( createLevelSet<SignedDistanceLocation>( inputs, mesh ) )
    {
        // Extract parameters.
        const auto& params = inputs["particle_level_set"];

        // Particles have an analytic spherical level set. Get the radius as a
        // fraction of the cell size.
        _dx = mesh->localGrid()->globalGrid().globalMesh().cellSize( 0 );
        _radius = 0.5;
        if ( params.count( "particle_radius" ) )
            _radius = params["particle_radius"];
        _radius *= _dx;
    }

    Kokkos::View<const int*, memory_space> colorIndex() const
    {
        return _color_indices;
    }

    int colorCount() const { return _color_count; }

    /*!
      \brief Update the set of particle indices for the color we are building
      the set for. This operation is needed any time the particle population
      is updated (e.g. after a redistribution) and therefore is separated from
      the distance estimate because it may only be called periodically.
      \param exec_space The execution space to use for parallel kernels.
      \param c_p A view or slice containing the particle colors. The number
      and order of particles with respect to these colors must remain
      consistent in between calls to this function (e.g. in subsequent calls
      to estimateSignedDistance()).
    */
    template <class ExecutionSpace, class ParticleColors>
    void updateParticleColors( const ExecutionSpace& exec_space,
                               const ParticleColors& c_p )
    {
        Kokkos::Profiling::pushRegion(
            "Picasso::ParticleLevelSet::updateParticleColors" );

        // Initialize color indices.
        _color_indices = Kokkos::View<int*, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "color_indices" ),
            c_p.size() );

        // If we dont have any particles on this rank we have nothing to do.
        if ( 0 == c_p.size() )
        {
            _color_count = 0;
            return;
        }

        // If the color is not negative then count how many particles have the
        // color we want and compute the indices.
        else if ( _color >= 0 )
        {
            Kokkos::View<int, memory_space> color_count( "color_count" );
            int color = _color;
            auto color_ind = _color_indices;
            Kokkos::parallel_for(
                "Picasso::ParticleLevelSet::CountColor",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                     c_p.size() ),
                KOKKOS_LAMBDA( const int p ) {
                    if ( color == c_p( p ) )
                    {
                        color_ind(
                            Kokkos::atomic_fetch_add( &color_count(), 1 ) ) = p;
                    }
                } );
            Kokkos::deep_copy( _color_count, color_count );
            _color_indices = color_ind;
        }

        // Otherwise a negative color means all particles are included in the
        // level set so we can more efficiently generate the array.
        else
        {
            int color_count = c_p.size();
            auto color_ind = _color_indices;
            Kokkos::parallel_for(
                "Picasso::ParticleLevelSet::FillColorIndices",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                     c_p.size() ),
                KOKKOS_LAMBDA( const int p ) { color_ind( p ) = p; } );

            _color_count = color_count;
            _color_indices = color_ind;
        }

        Kokkos::Profiling::popRegion();
    }

    /*!
      \brief Compute the signed distance function estimate from the current
      particle locations. All the positive values will be correct but the
      negative values will only have the correct sign until redistanced.
      \param exec_space The execution space to use for parallel kernels.
      \param x_p A view or slice of particle positions. If the grid is
      adaptive these positions must be in the logical frame. The number and
      order of particles with respect to these positions must be consistent
      with the colors provided to the last call to updateParticleColors().
    */
    template <class ExecutionSpace, class ParticlePositions>
    void estimateSignedDistance( const ExecutionSpace& exec_space,
                                 const ParticlePositions& x_p )
    {
        Kokkos::Profiling::pushRegion(
            "Picasso::ParticleLevelSet::estimateSignedDistance" );

        // Distance estimate.
        auto distance_estimate = _ls->getDistanceEstimate();

        // View of the distance estimate.
        auto estimate_view = distance_estimate->view();

        // Local mesh.
        auto local_grid = distance_estimate->layout()->localGrid();
        auto local_mesh = Cajita::createLocalMesh<memory_space>( *local_grid );

        // If we have no particles of the given color on this rank then we are
        // in a region of positive distance. Estimate the signed distance
        // everywhere to be the minimum halo width as this is the minimum
        // distance to a particle surface we could have resolved that wouldn't
        // show up in the min-reduce operation.
        if ( 0 == _color_count )
        {
            double min_dist = _dx * local_grid->haloCellWidth();
            Cajita::ArrayOp::assign( *distance_estimate, min_dist,
                                     Cajita::Ghost() );
        }

        // Otherwise we have particles so build a tree from the particles of
        // the given color and estimate the distance.
        else
        {
            // Build the tree
            ParticleLevelSetPrimitiveData<ParticlePositions,
                                          Kokkos::View<int*, memory_space>>
                primitive_data;
            primitive_data.x = x_p;
            primitive_data.c = _color_indices;
            primitive_data.num_color = _color_count;
            ArborX::BVH<memory_space> bvh( exec_space, primitive_data );

            // Make the search predicates.
            ParticleLevelSetPredicateData<decltype( local_mesh ), entity_type>
                predicate_data( local_mesh, *local_grid );

            // Make the distance callback.
            ParticleLevelSetCallback<ParticlePositions,
                                     decltype( estimate_view )>
                distance_callback;
            distance_callback.primitive_data = primitive_data;
            distance_callback.distance_estimate = estimate_view;
            distance_callback.radius = static_cast<float>( _radius );

            // Query the particle tree with the mesh entities to find the
            // closest particle and compute the initial signed distance
            // estimate. Dummy arguments are needed even though we don't care
            // about the output.
            bvh.query( exec_space, predicate_data, distance_callback );
        }

        // Do a reduction to get the minimum distance within the minimum halo
        // width.
        _ls->getHalo()->scatter( exec_space, Cajita::ScatterReduce::Min(),
                                 *distance_estimate );

        Kokkos::Profiling::popRegion();
    }

    // Get the particle radius.
    double particleRadius() const { return _radius; }

    // Get the level set.
    std::shared_ptr<level_set> levelSet() const { return _ls; }

  private:
    int _color;
    double _radius;
    double _dx;
    Kokkos::View<int*, memory_space> _color_indices;
    int _color_count;
    std::shared_ptr<level_set> _ls;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a particle level set over particles of the given color. A
  color of -1 will create the level set over all particles.
  \param inputs Level set settings.
  \param mesh The mesh over which to build the signed distance function.
  \param color The particle color over which to build the level set. Use -1 if
  the level set is to be built over all particles.
*/
template <class SignedDistanceLocation, class MeshType>
std::shared_ptr<ParticleLevelSet<MeshType, SignedDistanceLocation>>
createParticleLevelSet( const nlohmann::json inputs,
                        const std::shared_ptr<MeshType>& mesh, const int color )
{
    return std::make_shared<ParticleLevelSet<MeshType, SignedDistanceLocation>>(
        inputs, mesh, color );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_PARTICLELEVELSET_HPP
