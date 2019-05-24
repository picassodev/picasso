# setup gtest
if(NOT Harlow_ENABLE_TESTING)
  return()
endif()

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/tpls)

include_directories(${GTEST_SOURCE_DIR})
add_library(cajita_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
set_target_properties(cajita_gtest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test CajitaVersion GridBlock)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test cajita cajita_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test --gtest_color=yes)
endforeach()

##--------------------------------------------------------------------------##
## Kokkos-only tests with a single execution space.
##--------------------------------------------------------------------------##
foreach(_device ${HARLOW_SUPPORTED_DEVICES})
  if(Harlow_ENABLE_${_device})
    foreach(_test GridField GridExecPolicy)
      add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp unit_test_main.cpp)
      target_link_libraries(${_test}_test_${_device} cajita cajita_gtest)
      if(_device STREQUAL OpenMP)
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
  target_link_libraries(${_test}_test cajita cajita_gtest)
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
    foreach(_test GridCommunicationFace GridCommunicationNode GridFieldVectorOps BovGridFieldWriter)
      add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp unit_test_main.cpp)
      target_link_libraries(${_test}_test_${_device} cajita cajita_gtest)
      foreach(_np 1 ${MPI_TEST_NUM_PROC})
        add_test(NAME ${_test}_test_${_device}_${_np} COMMAND
          ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${_test}_test_${_device} --gtest_color=yes)
      endforeach()
    endforeach()
  endif()
endforeach()
