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

#include <Picasso_InputParser.hpp>

#include <fstream>

namespace Picasso
{

//---------------------------------------------------------------------------//
//! Parse JSON file.
nlohmann::json parse( const std::string& filename )
{
    std::ifstream stream( filename );
    return nlohmann::json::parse( stream );
}

} // end namespace Picasso
