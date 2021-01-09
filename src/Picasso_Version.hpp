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

#ifndef PICASSO_VERSION_HPP
#define PICASSO_VERSION_HPP

#include <Picasso_config.hpp>

#include <string>

namespace Picasso
{

std::string version();

std::string git_commit_hash();

} // end namespace Picasso

#endif // end PICASSO_VERSION_HPP
