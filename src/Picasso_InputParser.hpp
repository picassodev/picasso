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

#include <string>

namespace Picasso
{

nlohmann::json parse( const std::string& filename );

} // end namespace Picasso

#endif // end PICASSO_INPUTPARSER_HPP
