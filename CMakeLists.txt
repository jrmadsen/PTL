################################################################################
#   cmake settings
#
cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)


################################################################################
#   version
#
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")
set(PTL_VERSION "${VERSION_STRING}" CACHE STRING "Version of PTL project" FORCE)
message(STATUS "PTL version ${PTL_VERSION}")


################################################################################
#   project
#
project(PTL LANGUAGES C CXX VERSION ${PTL_VERSION})


################################################################################
#
#
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(PTL_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}}
        CACHE STRING "PTL ${_TYPE} version" FORCE)
    mark_as_advanced(PTL_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(PTL_VERSION)

################################################################################

include(MacroUtilities)
include(Options)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

################################################################################
#   PTL source
################################################################################

add_subdirectory(source)

################################################################################
#   Installation and info
################################################################################

include(ConfigurePackage)

if(PTL_BUILD_EXAMPLES)
    set(PTL_DIR ${PROJECT_BINARY_DIR})
    add_subdirectory(examples)
endif()

print_features()
