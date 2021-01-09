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

#include <Picasso_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( picasso_version, version_test )
{
    auto const version_id = Picasso::version();
    std::cout << "Picasso version " << version_id << std::endl;

    auto const commit_hash = Picasso::git_commit_hash();
    std::cout << "Picasso commit hash " << commit_hash << std::endl;
}

} // end namespace Test
