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

#ifndef PICASSO_TYPES_HPP
#define PICASSO_TYPES_HPP

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Logical dimension index.
using Dim = Cajita::Dim;

//---------------------------------------------------------------------------//
// Spatial dimension selector.
template <std::size_t N>
struct SpaceDim : public std::integral_constant<std::size_t, N>
{
};

using ALL_INDEX_t = decltype( Kokkos::ALL );

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // PICASSO_TYPES_HPP
