# setup gtest
if(NOT Harlow_ENABLE_TESTING)
  return()
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_PROJECT_BINARY_DIR}/googletest-src/googletest/include)

set(gtest_args --gtest_color=yes)

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
add_executable(Version_test tstVersion.cpp unit_test_main.cpp)
target_link_libraries(Version_test harlow gtest)
add_test(
  NAME Version_test
  COMMAND Version_test --gtest_color=yes)

add_executable(InputParser_test tstInputParser.cpp unit_test_main.cpp)
target_link_libraries(InputParser_test harlow gtest)
add_test(
  NAME InputParser_test
  COMMAND InputParser_test --gtest_color=yes)
add_custom_command(
  TARGET InputParser_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/input_parser_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/input_parser_test.json)


##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
set(HARLOW_SUPPORTED_DEVICES SERIAL OPENMP CUDA)

macro(Harlow_add_tests)
  cmake_parse_arguments(HARLOW_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(HARLOW_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _np 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_np})
      list(APPEND HARLOW_UNIT_TEST_MPIEXEC_NUMPROCS ${_np})
    endif()
  endforeach()
  set(HARLOW_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND HARLOW_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  set(HARLOW_UNIT_TEST_MAIN unit_test_main.cpp)
  foreach(_device ${HARLOW_SUPPORTED_DEVICES})
    if(Kokkos_ENABLE_${_device})
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
      file(MAKE_DIRECTORY ${_dir})
      foreach(_test ${HARLOW_UNIT_TEST_NAMES})
        set(_file ${_dir}/tst${_test}_${_device}.cpp)
        file(WRITE ${_file} "#include <Test${_device}_Category.hpp>\n")
        file(APPEND ${_file} "#include <tst${_test}.hpp>\n")
        set(_target ${_test}_test_${_device})
        add_executable(${_target} ${_file} ${HARLOW_UNIT_TEST_MAIN})
        target_include_directories(${_target} PUBLIC ${_dir})
        target_link_libraries(${_target} harlow gtest)
        if(HARLOW_UNIT_TEST_MPI)
          foreach(_np ${HARLOW_UNIT_TEST_MPIEXEC_NUMPROCS})
            add_test(NAME ${_target}_${_np} COMMAND
              ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args})
          endforeach()
        else()
          if(_device STREQUAL OpenMP)
            foreach(_thread ${HARLOW_UNIT_TEST_NUMTHREADS})
              add_test(NAME ${_target}_${_thread} COMMAND
                ${_target} ${gtest_args} --kokkos-threads=${_thread})
            endforeach()
          else()
            add_test(NAME ${_target} COMMAND ${_target} ${gtest_args})
          endif()
        endif()
      endforeach()
    endif()
  endforeach()
endmacro()

Harlow_add_tests(NAMES DenseLinearAlgebra)
Harlow_add_tests(MPI NAMES ParticleInit ParticleCommunication SiloParticleWriter)
