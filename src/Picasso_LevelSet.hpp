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

#ifndef PICASSO_LEVELSET_HPP
#define PICASSO_LEVELSET_HPP

#include <Picasso_FieldManager.hpp>
#include <Picasso_LevelSetRedistance.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <cfloat>

//---------------------------------------------------------------------------//
namespace Picasso
{
//---------------------------------------------------------------------------//
// Level set. Composes a signed distance function.
template <class MeshType, class SignedDistanceLocation>
class LevelSet
{
  public:
    using mesh_type = MeshType;
    using memory_space = typename mesh_type::memory_space;
    using location_type = SignedDistanceLocation;
    using entity_type = typename location_type::entity_type;
    using array_type = Cajita::Array<double, entity_type,
                                     Cajita::UniformMesh<double>, memory_space>;
    using halo_type = Cajita::Halo<memory_space>;

    /*!
      \brief Construct the level set over the given mesh.
      \param inputs Level set settings.
      \param mesh The mesh over which to build the signed distance function.
      \param The particle color over which to build the level set. Use -1 if
      the level set is to be built over all particles.
    */
    LevelSet( const nlohmann::json inputs,
              const std::shared_ptr<MeshType>& mesh )
        : _mesh( mesh )
    {
        // Create array data.
        _distance_estimate =
            createArray( *_mesh, location_type(), Field::DistanceEstimate() );
        _signed_distance =
            createArray( *_mesh, location_type(), Field::SignedDistance() );
        _halo = Cajita::createHalo( Cajita::NodeHaloPattern<3>(), -1,
                                    *( _signed_distance ) );

        // Cell size.
        _dx = _mesh->localGrid()->globalGrid().globalMesh().cellSize( 0 );

        // Extract parameters.
        const auto& params = inputs["level_set"];

        // Get the Hopf-Lax redistancing parameters.
        if ( params.count( "redistance_secant_tol" ) )
            _redistance_secant_tol = params["redistance_secant_tol"];
        if ( params.count( "redistance_max_secant_iter" ) )
            _redistance_max_secant_iter = params["redistance_max_secant_iter"];
        if ( params.count( "redistance_num_random_guess" ) )
            _redistance_num_random_guess =
                params["redistance_num_random_guess"];
        if ( params.count( "redistance_projection_tol" ) )
            _redistance_projection_tol = params["redistance_projection_tol"];
        if ( params.count( "redistance_max_projection_iter" ) )
            _redistance_max_projection_iter =
                params["redistance_max_projection_iter"];
    }

    /*!
      \brief Redistance the signed distance function. The distance estimate on
      the fine grid local entities must have been computed before this
      function is called.
      \param exec_space The execution space to use for parallel kernels.
    */
    template <class ExecutionSpace>
    void redistance( const ExecutionSpace& exec_space )
    {
        Kokkos::Profiling::pushRegion( "Picasso::LevelSet::redistance" );

        // Local mesh.
        auto local_mesh =
            Cajita::createLocalMesh<memory_space>( *( _mesh->localGrid() ) );

        // Views.
        auto estimate_view = _distance_estimate->view();
        auto distance_view = _signed_distance->view();

        // Local-to-global indexer.
        auto l2g = Cajita::IndexConversion::createL2G( *( _mesh->localGrid() ),
                                                       entity_type() );

        // Gather to get updated ghost values.
        _halo->gather( exec_space, *_distance_estimate );

        // Redistance on the coarse grid.
        auto own_entities = _mesh->localGrid()->indexSpace(
            Cajita::Own(), entity_type(), Cajita::Local() );

        double secant_tol = _redistance_secant_tol;
        int max_secant_iter = _redistance_max_secant_iter;
        int num_random_guess = _redistance_num_random_guess;
        double projection_tol = _redistance_projection_tol;
        int max_projection_iter = _redistance_max_projection_iter;

        Kokkos::parallel_for(
            "Picasso::LevelSet::RedistanceCoarse",
            Cajita::createExecutionPolicy( own_entities, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get the global id of the entity.
                int gi, gj, gk;
                l2g( i, j, k, gi, gj, gk );

                // Only redistance on even-numbered entities. This effectively
                // generates a coarse grid where 2x2x2 blocks of cells on the
                // fine grid are combined.
                if ( !( gi % 2 ) && !( gj % 2 ) && !( gk % 2 ) )
                {
                    int entity_index[3] = { i, j, k };
                    distance_view( i, j, k, 0 ) =
                        LevelSetRedistance::redistanceEntity(
                            entity_type(), estimate_view, local_mesh,
                            entity_index, secant_tol, max_secant_iter,
                            num_random_guess, projection_tol,
                            max_projection_iter );
                }
            } );

        // Gather the coarse grid estimate to get updated ghost values.
        _halo->gather( exec_space, *_signed_distance );

        // Interpolate from coarse grid to fine grid.
        Kokkos::parallel_for(
            "Picasso::LevelSet::RedistanceInterpolate",
            Cajita::createExecutionPolicy( own_entities, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Get the global id of the entity.
                int gi, gj, gk;
                l2g( i, j, k, gi, gj, gk );

                // Interpolate even-numbered entities to the other entities.
                // All odd case
                if ( ( gi % 2 ) && ( gj % 2 ) && ( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.125 * ( distance_view( i - 1, j - 1, k - 1, 0 ) +
                                  distance_view( i - 1, j - 1, k + 1, 0 ) +
                                  distance_view( i - 1, j + 1, k - 1, 0 ) +
                                  distance_view( i - 1, j + 1, k + 1, 0 ) +
                                  distance_view( i + 1, j - 1, k - 1, 0 ) +
                                  distance_view( i + 1, j - 1, k + 1, 0 ) +
                                  distance_view( i + 1, j + 1, k - 1, 0 ) +
                                  distance_view( i + 1, j + 1, k + 1, 0 ) );
                }

                // Even i case
                else if ( !( gi % 2 ) && ( gj % 2 ) && ( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.25 * ( distance_view( i, j - 1, k - 1, 0 ) +
                                 distance_view( i, j - 1, k + 1, 0 ) +
                                 distance_view( i, j + 1, k - 1, 0 ) +
                                 distance_view( i, j + 1, k + 1, 0 ) );
                }

                // Even j case
                else if ( ( gi % 2 ) && !( gj % 2 ) && ( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.25 * ( distance_view( i - 1, j, k - 1, 0 ) +
                                 distance_view( i - 1, j, k + 1, 0 ) +
                                 distance_view( i + 1, j, k - 1, 0 ) +
                                 distance_view( i + 1, j, k + 1, 0 ) );
                }

                // Even k case.
                else if ( ( gi % 2 ) && ( gj % 2 ) && !( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.25 * ( distance_view( i - 1, j - 1, k, 0 ) +
                                 distance_view( i - 1, j + 1, k, 0 ) +
                                 distance_view( i + 1, j - 1, k, 0 ) +
                                 distance_view( i + 1, j + 1, k, 0 ) );
                }

                // Even ij
                else if ( !( gi % 2 ) && !( gj % 2 ) && ( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.5 * ( distance_view( i, j, k - 1, 0 ) +
                                distance_view( i, j, k + 1, 0 ) );
                }

                // Even ik
                else if ( !( gi % 2 ) && ( gj % 2 ) && !( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.5 * ( distance_view( i, j - 1, k, 0 ) +
                                distance_view( i, j + 1, k, 0 ) );
                }

                // Even jk
                else if ( ( gi % 2 ) && !( gj % 2 ) && !( gk % 2 ) )
                {
                    estimate_view( i, j, k, 0 ) =
                        0.5 * ( distance_view( i - 1, j, k, 0 ) +
                                distance_view( i + 1, j, k, 0 ) );
                }

                // Even-numbered entities don't change.
                else
                {
                    estimate_view( i, j, k, 0 ) = distance_view( i, j, k, 0 );
                }
            } );

        // Gather to get updated ghost values.
        _halo->gather( exec_space, *_distance_estimate );

        // Narrow-band redistance on the fine grid. Only redistance within the
        // threshold tolerance. We can't resolve the level set any further
        // than the width of the halo.
        auto threshold = _dx * _mesh->localGrid()->haloCellWidth();
        Kokkos::parallel_for(
            "Picasso::LevelSet::RedistanceFine",
            Cajita::createExecutionPolicy( own_entities, exec_space ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                // Only redistance on the fine grid if the estimate is less
                // than the threshold distance.
                if ( Kokkos::fabs( estimate_view( i, j, k, 0 ) ) < threshold )
                {
                    int entity_index[3] = { i, j, k };
                    distance_view( i, j, k, 0 ) =
                        LevelSetRedistance::redistanceEntity(
                            entity_type(), estimate_view, local_mesh,
                            entity_index, secant_tol, max_secant_iter,
                            num_random_guess, projection_tol,
                            max_projection_iter );
                }

                // Otherwise just assign the distance to be our estimate.
                else
                {
                    distance_view( i, j, k, 0 ) = estimate_view( i, j, k, 0 );
                }
            } );

        // Gather again to get narrow-band updated ghost values. We could
        // defer this gather to when the user needs it but most things that we
        // will do with this level set function will required the gathered
        // values.
        _halo->gather( exec_space, *_signed_distance );

        Kokkos::Profiling::popRegion();
    }

    // Get the signed distance estimate.
    std::shared_ptr<array_type> getDistanceEstimate() const
    {
        return _distance_estimate;
    }

    // Get the redistanced signed distance function.
    std::shared_ptr<array_type> getSignedDistance() const
    {
        return _signed_distance;
    }

    // Get the halo for the signed distance arrays.
    std::shared_ptr<halo_type> getHalo() const { return _halo; }

  private:
    std::shared_ptr<MeshType> _mesh;
    std::shared_ptr<array_type> _distance_estimate;
    std::shared_ptr<array_type> _signed_distance;
    std::shared_ptr<halo_type> _halo;
    double _dx;
    double _redistance_secant_tol = 0.25;
    int _redistance_max_secant_iter = 10;
    int _redistance_num_random_guess = 5;
    double _redistance_projection_tol = 1.0e-4;
    int _redistance_max_projection_iter = 200;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a level set over particles of the given color.
  \param inputs Level set settings.
  \param mesh The mesh over which to build the signed distance function.
*/
template <class SignedDistanceLocation, class MeshType>
std::shared_ptr<LevelSet<MeshType, SignedDistanceLocation>>
createLevelSet( const nlohmann::json inputs,
                const std::shared_ptr<MeshType>& mesh )
{
    return std::make_shared<LevelSet<MeshType, SignedDistanceLocation>>( inputs,
                                                                         mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_LEVELSET_HPP
