#ifndef HARLOW_TEST_CUDA_CATEGORY_HPP
#define HARLOW_TEST_CUDA_CATEGORY_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_Cuda.hpp>

#include <gtest/gtest.h>

namespace Test {

class cuda : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY cuda
#define TEST_EXECSPACE Kokkos::Cuda
#define TEST_MEMSPACE Kokkos::CudaSpace

#endif // end HARLOW_TEST_CUDA_CATEGORY_HPP
