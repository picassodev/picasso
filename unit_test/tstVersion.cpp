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
