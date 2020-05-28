# core library
configure_file(Picasso_config.hpp.cmakein Picasso_config.hpp)

set(CORE_HEADERS
  core/Picasso_AdaptiveMesh.hpp
  core/Picasso_APIC.hpp
  core/Picasso_DenseLinearAlgebra.hpp
  core/Picasso_FacetGeometry.hpp
  core/Picasso_FieldManager.hpp
  core/Picasso_FieldTypes.hpp
  core/Picasso_InputParser.hpp
  core/Picasso_LevelSetRedistance.hpp
  core/Picasso_ParticleCommunication.hpp
  core/Picasso_ParticleInit.hpp
  core/Picasso_ParticleLevelSet.hpp
  core/Picasso_ParticleList.hpp
  core/Picasso_PolyPIC.hpp
  core/Picasso_SiloParticleWriter.hpp
  core/Picasso_Types.hpp
  core/Picasso_UniformMesh.hpp
  core/Picasso_Version.hpp
  )

set(CORE_SOURCES
  core/Picasso_InputParser.cpp
  core/Picasso_UniformMesh.cpp
  core/Picasso_Version.cpp
  )

add_library(picasso ${CORE_SOURCES})

target_link_libraries(picasso
  Kokkos::kokkos
  MPI::MPI_CXX
  Cabana::cabanacore
  Cajita::Cajita
  Silo::silo
  ArborX::ArborX
  Boost::boost )

target_include_directories(picasso
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_BINARY_DIR}
  )

# lpbf solver
set(LPBF_HEADERS
  lpbf/Picasso_LPBF_AuxiliaryFieldTypes.hpp
  lpbf/Picasso_LPBF_LaserSource.hpp
  lpbf/Picasso_LPBF_ProblemManager.hpp
  lpbf/Picasso_LPBF_Solver.hpp
  lpbf/Picasso_LPBF_TimeIntegrator.hpp
  )

add_executable( PicassoLPBF lpbf/picasso_lpbf.cpp )
target_link_libraries( PicassoLPBF PRIVATE picasso )
target_include_directories( PicassoLPBF
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lpbf
  ${CMAKE_CURRENT_BINARY_DIR} )

# flip solver
set(FLIP_HEADERS
  flip/Picasso_FLIP_AuxiliaryFieldTypes.hpp
  flip/Picasso_FLIP_EquationOfState.hpp
  flip/Picasso_FLIP_ProblemManager.hpp
  flip/Picasso_FLIP_Solver.hpp
  flip/Picasso_FLIP_TimeIntegrator.hpp
  )

add_executable( PicassoFLIP flip/picasso_flip.cpp )
target_link_libraries( PicassoFLIP PRIVATE picasso )
target_include_directories( PicassoFLIP
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/flip
  ${CMAKE_CURRENT_BINARY_DIR} )
