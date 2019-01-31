configure_file(Harlow_config.hpp.cmakein Harlow_config.hpp)

#-----------------------------------------------------------------------------

set(HEADERS
  Harlow_GridBlock.hpp
  Harlow_GridCommunication.hpp
  Harlow_GridExecPolicy.hpp
  Harlow_GridField.hpp
  Harlow_GridFieldVectorOps.hpp
  Harlow_MpiTraits.hpp
  Harlow_ParticleInterpolation.hpp
  Harlow_Splines.hpp
  Harlow_Types.hpp
  Harlow_Version.hpp
  )

set(SOURCES
  Harlow_GlobalGrid.cpp
  Harlow_GridBlock.cpp
  Harlow_Version.cpp
  )

#-----------------------------------------------------------------------------

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Harlow_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#-----------------------------------------------------------------------------

add_library(harlow ${SOURCES})
target_include_directories(harlow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(harlow Kokkos::kokkos)
target_link_libraries(harlow MPI::MPI_CXX)
install(TARGETS harlow LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

#-----------------------------------------------------------------------------

add_subdirectory(unit_test)