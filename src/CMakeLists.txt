configure_file(Harlow_config.hpp.cmakein Harlow_config.hpp)

set(CORE_HEADERS
  core/Harlow_AdaptiveMesh.hpp
  core/Harlow_APIC.hpp
  core/Harlow_DenseLinearAlgebra.hpp
  core/Harlow_FacetGeometry.hpp
  core/Harlow_FieldManager.hpp
  core/Harlow_FieldTypes.hpp
  core/Harlow_InputParser.hpp
  core/Harlow_LevelSetRedistance.hpp
  core/Harlow_ParticleCommunication.hpp
  core/Harlow_ParticleInit.hpp
  core/Harlow_ParticleLevelSet.hpp
  core/Harlow_ParticleList.hpp
  core/Harlow_PolyPIC.hpp
  core/Harlow_SiloParticleWriter.hpp
  core/Harlow_Types.hpp
  core/Harlow_UniformMesh.hpp
  core/Harlow_Version.hpp
  )

set(CORE_SOURCES
  core/Harlow_InputParser.cpp
  core/Harlow_UniformMesh.cpp
  core/Harlow_Version.cpp
  )

set(LPBF_HEADERS
  lpbf/Harlow_LPBF_AuxiliaryFieldTypes.hpp
  lpbf/Harlow_LPBF_LaserSource.hpp
  lpbf/Harlow_LPBF_ProblemManager.hpp
  lpbf/Harlow_LPBF_Solver.hpp
  lpbf/Harlow_LPBF_TimeIntegrator.hpp
  )

add_library(harlow ${CORE_SOURCES})

target_link_libraries(harlow
  Kokkos::kokkos
  MPI::MPI_CXX
  Cabana::cabanacore
  Cajita::Cajita
  Silo::silo
  ArborX::ArborX
  Boost::boost )

target_include_directories(harlow
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/lpbf
  ${CMAKE_CURRENT_BINARY_DIR}
  )

add_executable( HarlowLPBF lpbf/harlow_lpbf.cpp )
target_link_libraries( HarlowLPBF PRIVATE harlow )
target_include_directories( HarlowLPBF
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/lpbf
  ${CMAKE_CURRENT_BINARY_DIR} )
