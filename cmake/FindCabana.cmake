find_package(PkgConfig QUIET)
pkg_check_modules(PC_CABANA cabana QUIET)

find_path(CABANA_INCLUDE_DIR Cabana_Core.hpp HINTS ${PC_CABANA_INCLUDE_DIRS})
find_library(CABANA_LIBRARY NAMES cabanacore HINTS ${PC_CABANA_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CABANA DEFAULT_MSG CABANA_LIBRARY)

mark_as_advanced(CABANA_INCLUDE_DIR CABANA_LIBRARY)

if(CABANA_INCLUDE_DIR AND CABANA_LIBRARY)
  add_library(Cabana::core UNKNOWN IMPORTED)
  set_target_properties(Cabana::core PROPERTIES
    IMPORTED_LOCATION ${CABANA_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CABANA_INCLUDE_DIR})
  if(CABANA_LINK_FLAGS)
    set_property(TARGET Cabana::core APPEND_STRING PROPERTY INTERFACE_LINK_LIBRARIES " ${CABANA_LINK_FLAGS}")
  endif()
endif()
