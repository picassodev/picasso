configure_file(Cajita_config.hpp.cmakein Cajita_config.hpp)

#-----------------------------------------------------------------------------

set(HEADERS
  Cajita_BovFieldWriter.hpp
  Cajita_GlobalGrid.hpp
  Cajita_GridBlock.hpp
  Cajita_GridCommunication.hpp
  Cajita_GridExecPolicy.hpp
  Cajita_Field.hpp
  Cajita_FieldVectorOps.hpp
  Cajita_MpiTraits.hpp
  Cajita_Types.hpp
  Cajita_Version.hpp
  )

set(SOURCES
  Cajita_GlobalGrid.cpp
  Cajita_GridBlock.cpp
  Cajita_Version.cpp
  )

#-----------------------------------------------------------------------------

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cajita_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#-----------------------------------------------------------------------------

add_library(cajita ${SOURCES})
target_include_directories(cajita PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cajita Kokkos::kokkos)
target_link_libraries(cajita MPI::MPI_CXX)
install(TARGETS cajita LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

#-----------------------------------------------------------------------------

add_subdirectory(unit_test)
