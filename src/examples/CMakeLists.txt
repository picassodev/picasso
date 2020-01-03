include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( NohProblem noh_problem.cpp )
target_link_libraries( NohProblem PRIVATE harlow)
target_include_directories( NohProblem PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
