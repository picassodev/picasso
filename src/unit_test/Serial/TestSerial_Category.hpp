#ifndef HARLOW_TEST_SERIAL_CATEGORY_HPP
#define HARLOW_TEST_SERIAL_CATEGORY_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_Serial.hpp>

#include <gtest/gtest.h>

namespace Test {

class serial : public ::testing::Test {
protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};

} // namespace Test

#define TEST_CATEGORY serial
#define TEST_EXECSPACE Kokkos::Serial
#define TEST_MEMSPACE Kokkos::HostSpace

#endif // end HARLOW_TEST_SERIAL_CATEGORY_HPP
