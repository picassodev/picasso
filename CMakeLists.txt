# project settings
cmake_minimum_required(VERSION 3.12)

# FIXME: C language only for Cabana HDF5
project(Picasso LANGUAGES CXX C VERSION 0.1.0)

include(GNUInstallDirs)

# find dependencies (MPI and Kokkos found through Cabana)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(CLANG_FORMAT 10)
find_package(Cabana REQUIRED 0.5 COMPONENTS Cabana::Cajita Cabana::cabanacore)
find_package(ArborX REQUIRED)
find_package(Boost 1.66.0 REQUIRED)

if(Kokkos_VERSION_MAJOR LESS 4 AND Kokkos_VERSION_MINOR LESS 7)
  message( FATAL_ERROR "Picasso requires Kokkos >= 3.7" )
endif()

option( Picasso_REQUIRE_SILO "Require Picasso to build with Cabana Silo support" OFF )
if( Picasso_REQUIRE_SILO AND NOT Cabana_ENABLE_SILO )
    message( FATAL_ERROR "Cabana was not compiled with SILO" )
endif()

# find dependencies (from Cabana CMakeLists.txt)
macro(Picasso_add_dependency)
  cmake_parse_arguments(PICASSO_DEPENDENCY "" "PACKAGE" "" ${ARGN})
  find_package( ${PICASSO_DEPENDENCY_PACKAGE} QUIET )
  string(TOUPPER "${PICASSO_DEPENDENCY_PACKAGE}" PICASSO_DEPENDENCY_OPTION )
  option(
    Picasso_REQUIRE_${PICASSO_DEPENDENCY_OPTION}
    "Require Picasso to build with ${PICASSO_DEPENDENCY_PACKAGE} support" ${PICASSO_DEPENDENCY_PACKAGE}_FOUND)
  if(Picasso_REQUIRE_${PICASSO_DEPENDENCY_OPTION})
    find_package( ${PICASSO_DEPENDENCY_PACKAGE} REQUIRED )
  endif()
  set(Picasso_ENABLE_${PICASSO_DEPENDENCY_OPTION} ${${PICASSO_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# Print the revision number to stdout
find_package(Git)
if(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  execute_process(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   Picasso_GIT_COMMIT_HASH
    )
else()
  set(Picasso_GIT_COMMIT_HASH "Not a git repository")
endif()
message(STATUS "Picasso Revision = '${Picasso_GIT_COMMIT_HASH}'")

# library
add_subdirectory(src)

# enable testing
option(Picasso_ENABLE_TESTING "Build tests" OFF)
if(Picasso_ENABLE_TESTING)
  find_package(GTest 1.10 REQUIRED)
  # Workaround for FindGTest module in CMake older than 3.20
  if(TARGET GTest::gtest)
    set(gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(gtest_target GTest::GTest)
  else()
    message(FATAL_ERROR "bug in GTest find module workaround")
  endif()
  enable_testing()
  add_subdirectory(unit_test)
endif()

# examples
add_subdirectory(examples)

# Package Configuration
write_basic_package_version_file("PicassoConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PicassoConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/PicassoConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/PicassoConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/PicassoConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Picasso)

# Clang Format
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.[c,h]pp unit_test/*.[c,h]pp)
  add_custom_target(picasso-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
