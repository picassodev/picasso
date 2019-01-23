#ifndef HARLOW_TEST_PTHREAD_CATEGORY_HPP
#define HARLOW_TEST_PTHREAD_CATEGORY_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_Threads.hpp>

#include <gtest/gtest.h>

namespace Test {

class pthread : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY pthread
#define TEST_EXECSPACE Kokkos::Threads
#define TEST_MEMSPACE Kokkos::HostSpace

#endif // end HARLOW_TEST_PTHREAD_CATEGORY_HPP
