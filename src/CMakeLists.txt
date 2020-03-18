configure_file(Harlow_config.hpp.cmakein Harlow_config.hpp)

set(HEADERS
  Harlow_AdaptiveMesh.hpp
  Harlow_DenseLinearAlgebra.hpp
  Harlow_FacetGeometry.hpp
  Harlow_InputParser.hpp
  Harlow_ParticleCommunication.hpp
  Harlow_ParticleInit.hpp
  Harlow_ParticleList.hpp
  Harlow_SiloParticleWriter.hpp
  Harlow_Types.hpp
  Harlow_UniformMesh.hpp
  Harlow_Version.hpp
  )

set(SOURCES
  Harlow_InputParser.cpp
  Harlow_UniformMesh.cpp
  Harlow_Version.cpp
  )

add_library(harlow ${SOURCES})

target_link_libraries(harlow
  Kokkos::kokkos
  MPI::MPI_CXX
  Cabana::cabanacore
  Cajita::Cajita
  Silo::silo
  Boost::boost )

target_include_directories(harlow
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

add_subdirectory(unit_test)

add_subdirectory(examples)
