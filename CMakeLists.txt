# project settings
cmake_minimum_required(VERSION 3.12)

project(Picasso LANGUAGES CXX VERSION 0.1.0)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(Kokkos 3 REQUIRED)
find_package(KokkosKernels 3 REQUIRED)
find_package(Cabana REQUIRED COMPONENTS Cabana::Cajita Cabana::cabanacore)
find_package(Silo REQUIRED)
find_package(ArborX REQUIRED)
find_package(Boost 1.66.0 REQUIRED)

# enable testing
option(Picasso_ENABLE_TESTING "Build tests" OFF)
if(Picasso_ENABLE_TESTING)
  enable_testing()
endif()

# library
add_subdirectory(src)

# tests
add_subdirectory(unit_test)

# examples
add_subdirectory(examples)
