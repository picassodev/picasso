add_subdirectory(sources)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/particle_init_test.json
  ${CMAKE_CURRENT_BINARY_DIR}/particle_init_test.json
  COPYONLY)

add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE picasso )
target_include_directories( DamBreak PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS DamBreak DESTINATION ${CMAKE_INSTALL_BINDIR})
