# setup gtest
if(NOT Harlow_ENABLE_TESTING)
  return()
endif()

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/tpls)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(harlow_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
set_target_properties(harlow_gtest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test Version GridBlock Splines)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test harlow harlow_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test --gtest_color=yes)
endforeach()

##--------------------------------------------------------------------------##
## Kokkos-only tests with a single execution space.
##--------------------------------------------------------------------------##
foreach(_device ${HARLOW_SUPPORTED_DEVICES})
  if(Harlow_ENABLE_${_device})
    foreach(_test GridField GridExecPolicy ParticleFieldOps ParticleInterpolation)
      add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp unit_test_main.cpp)
      target_link_libraries(${_test}_test_${_device} harlow harlow_gtest)
      if(_device STREQUAL Pthread OR _device STREQUAL OpenMP)
        foreach(_thread 1 2)
          add_test(NAME ${_test}_test_${_device}_${_thread} COMMAND
            ${_test}_test_${_device} --gtest_color=yes --kokkos-threads=${_thread})
        endforeach()
      else()
        add_test(NAME ${_test}_test_${_device} COMMAND ${_test}_test_${_device} --gtest_color=yes)
      endif()
    endforeach()
  endif()
endforeach()

##--------------------------------------------------------------------------##
## MPI-only tests.
##--------------------------------------------------------------------------##
if( ${MPIEXEC_MAX_NUMPROCS} GREATER 4 )
  set(MPI_TEST_NUM_PROC 4)
else()
  set(MPI_TEST_NUM_PROC ${MPIEXEC_MAX_NUMPROCS})
endif()

foreach(_test GlobalGrid)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test harlow harlow_gtest)
  foreach(_np 1 ${MPI_TEST_NUM_PROC})
    add_test(NAME ${_test}_test_${_np} COMMAND
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${_test}_test --gtest_color=yes)
  endforeach()
endforeach()

##--------------------------------------------------------------------------##
## MPI+Kokkos tests.
##--------------------------------------------------------------------------##
foreach(_device ${HARLOW_SUPPORTED_DEVICES})
  if(Harlow_ENABLE_${_device})
    foreach(_test GridCommunicationFace GridCommunicationNode GridFieldVectorOps ParticleCommunication BovGridFieldWriter SiloParticleFieldWriter)
      add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp unit_test_main.cpp)
      target_link_libraries(${_test}_test_${_device} harlow harlow_gtest)
      foreach(_np 1 ${MPI_TEST_NUM_PROC})
        add_test(NAME ${_test}_test_${_device}_${_np} COMMAND
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${_test}_test_${_device} --gtest_color=yes)
      endforeach()
    endforeach()
  endif()
endforeach()
