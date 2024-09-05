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

#ifndef PICASSO_INPUTPARSER_HPP
#define PICASSO_INPUTPARSER_HPP

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

#include <Kokkos_Core.hpp>

namespace Picasso
{

template <typename T, std::size_t N>
Kokkos::Array<T, N> copy( const std::array<T, N> in )
{
    Kokkos::Array<T, N> out;
    for ( std::size_t d = 0; d < in.size(); ++d )
        out[d] = in[d];
    return out;
}

inline nlohmann::json parse( const std::string& filename )
{
    std::ifstream stream( filename );
    return nlohmann::json::parse( stream );
}

} // end namespace Picasso

#endif // end PICASSO_INPUTPARSER_HPP
