##--------------------------------------------------------------------------##
## Input files.
##--------------------------------------------------------------------------##
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/input_parser_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/input_parser_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/input_parser_test.xml
  ${CMAKE_CURRENT_BINARY_DIR}/input_parser_test.xml
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/stl_reader_test.stl
  ${CMAKE_CURRENT_BINARY_DIR}/stl_reader_test.stl
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/facet_geometry_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/facet_geometry_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/uniform_mesh_test_1.json
  ${CMAKE_CURRENT_BINARY_DIR}/uniform_mesh_test_1.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/uniform_mesh_test_2.json
  ${CMAKE_CURRENT_BINARY_DIR}/uniform_mesh_test_2.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/adaptive_mesh_test_1.json
  ${CMAKE_CURRENT_BINARY_DIR}/adaptive_mesh_test_1.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/adaptive_mesh_test_2.json
  ${CMAKE_CURRENT_BINARY_DIR}/adaptive_mesh_test_2.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/particle_init_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/particle_init_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/particle_interpolation_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/particle_interpolation_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/polypic_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/polypic_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/field_manager_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/field_manager_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/facet_init_example.json
  ${CMAKE_CURRENT_BINARY_DIR}/facet_init_example.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/level_set_redistance_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/level_set_redistance_test.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/particle_level_set_zalesaks_disk.json
  ${CMAKE_CURRENT_BINARY_DIR}/particle_level_set_zalesaks_disk.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/zalesaks_disk.stl
  ${CMAKE_CURRENT_BINARY_DIR}/zalesaks_disk.stl
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/particle_level_set_zalesaks_sphere.json
  ${CMAKE_CURRENT_BINARY_DIR}/particle_level_set_zalesaks_sphere.json
  COPYONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/zalesaks_sphere.stl
  ${CMAKE_CURRENT_BINARY_DIR}/zalesaks_sphere.stl
  COPYONLY)

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
add_executable(Version_test tstVersion.cpp unit_test_main.cpp)
target_link_libraries(Version_test picasso ${gtest_target})
add_test(NAME Version_test COMMAND Version_test ${gtest_args})

##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
set(PICASSO_SUPPORTED_DEVICES SERIAL OPENMP CUDA HIP SYCL)

macro(Picasso_add_tests)
  cmake_parse_arguments(PICASSO_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(PICASSO_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _np 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_np})
      list(APPEND PICASSO_UNIT_TEST_MPIEXEC_NUMPROCS ${_np})
    endif()
  endforeach()
  set(PICASSO_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND PICASSO_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  set(PICASSO_UNIT_TEST_MAIN unit_test_main.cpp)
  foreach(_device ${PICASSO_SUPPORTED_DEVICES})
    if(Kokkos_ENABLE_${_device})
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
      file(MAKE_DIRECTORY ${_dir})
      foreach(_test ${PICASSO_UNIT_TEST_NAMES})
        # FIXME: tests failing only for HIP (and not in downstream tests).
        if(${_device} STREQUAL HIP)
          if (${_test} STREQUAL PolyPIC OR ${_test} STREQUAL APIC)
            continue()
          endif()
        endif()

        set(_file ${_dir}/tst${_test}_${_device}.cpp)
        file(WRITE ${_file} "#include <Test${_device}_Category.hpp>\n")
        file(APPEND ${_file} "#include <tst${_test}.hpp>\n")
        set(_target ${_test}_test_${_device})
        add_executable(${_target} ${_file} ${PICASSO_UNIT_TEST_MAIN})
        target_include_directories(${_target} PUBLIC ${_dir} ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${_target} picasso ${gtest_target})
        if(PICASSO_UNIT_TEST_MPI)
          foreach(_np ${PICASSO_UNIT_TEST_MPIEXEC_NUMPROCS})
            add_test(NAME ${_target}_${_np} COMMAND
              ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_PREFLAGS} ${gtest_args})
          endforeach()
        else()
          if(_device STREQUAL OpenMP)
            foreach(_thread ${PICASSO_UNIT_TEST_NUMTHREADS})
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

Picasso_add_tests(NAMES
  BatchedLinearAlgebra
  PolyPIC
  APIC
  )

Picasso_add_tests(MPI NAMES
  FacetGeometry
  ParticleList
  UniformCartesianMeshMapping
  BilinearMeshMapping
  UniformMesh
  AdaptiveMesh
  FieldManager
  GridOperator3d
  GridOperator2d
  ParticleInterpolation
  LevelSetRedistance
  ParticleLevelSet
  MarchingCubes
  )
