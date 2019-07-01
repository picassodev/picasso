# project settings
cmake_minimum_required(VERSION 3.12)

project(Harlow LANGUAGES CXX VERSION 0.1.0)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED)
find_package(Cajita REQUIRED)
find_package(Silo REQUIRED)

# enable testing
option(Harlow_ENABLE_TESTING "Build tests" OFF)
if(Harlow_ENABLE_TESTING)
  enable_testing()
endif()

# add source
add_subdirectory(src)
