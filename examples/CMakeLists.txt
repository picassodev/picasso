configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/dam_break.json
  ${CMAKE_CURRENT_BINARY_DIR}/dam_break.json
  COPYONLY)

add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE Picasso::picasso )
target_include_directories( DamBreak PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS DamBreak DESTINATION ${CMAKE_INSTALL_BINDIR})
