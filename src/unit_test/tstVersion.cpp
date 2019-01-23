#include <Harlow_Version.hpp>

#include <iostream>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test {

class harlow_version : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

TEST_F( harlow_version, version_test )
{
    auto const version_id = Harlow::version();
    EXPECT_TRUE( !version_id.empty() );
    std::cout << "Harlow version " << version_id << std::endl;

    auto const commit_hash = Harlow::git_commit_hash();
    EXPECT_TRUE( !commit_hash.empty() );
    std::cout << "Harlow commit hash " << commit_hash << std::endl;
}

} // end namespace Test
