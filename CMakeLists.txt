##---------------------------------------------------------------------------##
## Harlow/CMakeLists.txt
##---------------------------------------------------------------------------##
cmake_minimum_required(VERSION 3.9)

##---------------------------------------------------------------------------##
# Define your project name and set up major project options
##---------------------------------------------------------------------------##
project(Harlow LANGUAGES CXX VERSION 0.1.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

find_package(MPI REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(KOKKOS REQUIRED)

option(Harlow_ENABLE_Serial "Build Harlow with Serial support" ON)

option(Harlow_ENABLE_Pthread "Build Harlow with Pthread support" OFF)
if( Harlow_ENABLE_Pthread )
  find_package(Threads)
endif()

option(Harlow_ENABLE_OpenMP "Build Harlow with OpenMP support" OFF)
if( Harlow_ENABLE_OpenMP )
  find_package(OpenMP)
endif()

option(Harlow_ENABLE_Cuda "Build Harlow with Cuda support" OFF)
if( Harlow_ENABLE_Cuda )
  enable_language( CUDA )
endif()

set(HARLOW_SUPPORTED_DEVICES Serial Pthread OpenMP Cuda)
set(ENABLE_DEVICES)
foreach(_device ${HARLOW_SUPPORTED_DEVICES})
  if(Harlow_ENABLE_${_device})
    if(NOT KOKKOS_GMAKE_DEVICES MATCHES ".*${_device}.*")
      message(FATAL_ERROR "Kokkos built with devices '${KOKKOS_GMAKE_DEVICES}', but Harlow with ${_device} requested.")
    endif()
    list(APPEND ENABLE_DEVICES "${_device}")
  endif()
endforeach()
if(NOT ENABLE_DEVICES)
  message(FATAL_ERROR "NO Harlow backend enabled")
endif()
message(STATUS "Enable Devices: ${ENABLE_DEVICES}")

option(Harlow_ENABLE_TESTING "Build tests" OFF)
if(Harlow_ENABLE_TESTING)
  enable_testing()
endif()

find_package(Doxygen)
if(Doxygen_FOUND)
  doxygen_add_docs(doxygen core)
endif()

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Harlow_BUILD_MARCH "" CACHE STRING "Arch to use with -march= (if empty CMake will try to use 'native') in release build and only release build")

  # Try -march first. On platforms that don't support it, GCC will issue
  # a hard error, so we'll know not to use it.
  if(Harlow_BUILD_MARCH)
    set(INTERNAL_Harlow_BUILD_MARCH ${Harlow_BUILD_MARCH})
  else()
    set(INTERNAL_Harlow_BUILD_MARCH "native")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${INTERNAL_Harlow_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_Harlow_BUILD_MARCH}")
  elseif(Harlow_BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_Harlow_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_Harlow_BUILD_MARCH)
  endif()
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##

FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   Harlow_GIT_COMMIT_HASH
        )
ELSE()
    SET(Harlow_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Harlow Revision = '${Harlow_GIT_COMMIT_HASH}'")

add_subdirectory(src)
