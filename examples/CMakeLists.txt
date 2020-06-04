# $Id: CMakeLists.txt 100707 2016-10-31 20:21:07Z ihrivnac $

#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(PTL_Examples LANGUAGES C CXX VERSION ${PTL_VERSION})

# add the common "library"
add_subdirectory(common)
list(APPEND EXTERNAL_LIBRARIES common)

add_subdirectory(minimal)
add_subdirectory(basic)
add_subdirectory(extended)
add_subdirectory(gpu)
