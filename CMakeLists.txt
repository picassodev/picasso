# project settings
cmake_minimum_required(VERSION 3.12)

project(Harlow LANGUAGES CXX VERSION 0.1.0)

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
find_package(Cabana REQUIRED)
find_package(Cajita REQUIRED)
find_package(Silo REQUIRED)
find_package(nlohmann_json 3 REQUIRED)

# enable testing
option(Harlow_ENABLE_TESTING "Build tests" OFF)
if(Harlow_ENABLE_TESTING)
  enable_testing()
endif()

# add source
add_subdirectory(src)
