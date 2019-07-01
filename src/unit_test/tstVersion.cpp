#include <Harlow_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( harlow_version, version_test )
{
    auto const version_id = Harlow::version();
    std::cout << "Harlow version " << version_id << std::endl;

    auto const commit_hash = Harlow::git_commit_hash();
    std::cout << "Harlow commit hash " << commit_hash << std::endl;
}

} // end namespace Test
