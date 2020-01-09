include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( MpmIncompressible mpm_incompressible.cpp )
target_link_libraries( MpmIncompressible PRIVATE harlow)
target_include_directories( MpmIncompressible PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
