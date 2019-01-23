#ifndef HARLOW_TEST_OPENMP_CATEGORY_HPP
#define HARLOW_TEST_OPENMP_CATEGORY_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_OpenMP.hpp>

#include <gtest/gtest.h>

namespace Test {

class openmp : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY openmp
#define TEST_EXECSPACE Kokkos::OpenMP
#define TEST_MEMSPACE Kokkos::HostSpace

#endif // end HARLOW_TEST_OPENMP_CATEGORY_HPP
