# core library
configure_file(Picasso_config.hpp.cmakein Picasso_config.hpp)

set(HEADERS
  Picasso.hpp
  Picasso_AdaptiveMesh.hpp
  Picasso_APIC.hpp
  Picasso_BatchedLinearAlgebra.hpp
  Picasso_BilinearMeshMapping.hpp
  Picasso_CurvilinearMesh.hpp
  Picasso_FacetGeometry.hpp
  Picasso_FieldManager.hpp
  Picasso_FieldTypes.hpp
  Picasso_GridOperator.hpp
  Picasso_InputParser.hpp
  Picasso_LevelSet.hpp
  Picasso_LevelSetRedistance.hpp
  Picasso_MarchingCubes.hpp
  Picasso_ParticleInit.hpp
  Picasso_ParticleInterpolation.hpp
  Picasso_ParticleLevelSet.hpp
  Picasso_ParticleList.hpp
  Picasso_PolyPIC.hpp
  Picasso_Types.hpp
  Picasso_UniformMesh.hpp
  Picasso_UniformCartesianMeshMapping.hpp
  Picasso_Version.hpp
  )

set(SOURCES
  Picasso_InputParser.cpp
  Picasso_UniformMesh.cpp
  Picasso_Version.cpp
  )

add_library(picasso ${SOURCES})

target_link_libraries(picasso
  Cabana::cabanacore
  Cabana::Cajita
  ArborX::ArborX
  nlohmann_json::nlohmann_json
  )

target_include_directories(picasso
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS picasso
  EXPORT Picasso_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT Picasso_Targets
  FILE Picasso_Targets.cmake
  NAMESPACE Picasso::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Picasso)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Picasso_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)

add_library(Picasso::picasso ALIAS picasso)
