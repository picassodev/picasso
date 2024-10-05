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

#ifndef PICASSO_GRIDUPDATE_HPP
#define PICASSO_GRIDUPDATE_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_FieldTypes.hpp>

#include <Cabana_Grid.hpp>

#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Compute nodal velocity from mass-weighted momentum.
//---------------------------------------------------------------------------//
template <class MassType = Field::Mass, class VelocityType = Field::Velocity,
          class OldVelType = Field::OldU>
struct ComputeGridVelocity
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType&, const GatherDependencies&,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                const int i, const int j, const int k ) const
    {
        // Get the local dependencies.
        auto m_i = local_deps.get( Picasso::FieldLocation::Node(), MassType{} );
        auto u_i =
            local_deps.get( Picasso::FieldLocation::Node(), VelocityType{} );
        auto old_u_i =
            local_deps.get( Picasso::FieldLocation::Node(), OldVelType{} );

        // Compute velocity.
        for ( int d = 0; d < 3; ++d )
        {
            old_u_i( i, j, k, d ) = ( m_i( i, j, k ) > 0.0 )
                                        ? old_u_i( i, j, k, d ) / m_i( i, j, k )
                                        : 0.0;
            u_i( i, j, k, d ) = old_u_i( i, j, k, d );
        }
    }
};
} // namespace Picasso

#endif // end PICASSO_GRIDUPDATE_HPP
