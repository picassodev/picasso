# core library
configure_file(Picasso_config.hpp.cmakein Picasso_config.hpp)

set(HEADERS
  Picasso_AdaptiveMesh.hpp
  Picasso_APIC.hpp
  Picasso_BatchedLinearAlgebra.hpp
  Picasso_FacetGeometry.hpp
  Picasso_FieldManager.hpp
  Picasso_FieldTypes.hpp
  Picasso_InputParser.hpp
  Picasso_LevelSetRedistance.hpp
  Picasso_ParticleCommunication.hpp
  Picasso_ParticleInit.hpp
  Picasso_ParticleInterpolation.hpp
  Picasso_ParticleLevelSet.hpp
  Picasso_ParticleList.hpp
  Picasso_PolyPIC.hpp
  Picasso_Types.hpp
  Picasso_UniformMesh.hpp
  Picasso_Version.hpp
  )

if(Picasso_ENABLE_SILO)
  list(APPEND HEADERS Picasso_SiloParticleWriter.hpp)
endif()

set(SOURCES
  Picasso_InputParser.cpp
  Picasso_UniformMesh.cpp
  Picasso_Version.cpp
  )

add_library(picasso ${SOURCES})

target_link_libraries(picasso
  Kokkos::kokkos
  Kokkos::kokkoskernels
  MPI::MPI_CXX
  Cabana::cabanacore
  Cabana::Cajita
  ArborX::ArborX
  Boost::boost )

if(Picasso_ENABLE_SILO)
  target_link_libraries(picasso Silo::silo)
endif()

target_include_directories(picasso
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )
