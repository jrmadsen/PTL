# $Id: CMakeLists.txt 100707 2016-10-31 20:21:07Z ihrivnac $

#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
project(PTL_Examples LANGUAGES C CXX VERSION ${PTL_VERSION})

include_directories(${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/common)
file(GLOB common_headers ${CMAKE_CURRENT_LIST_DIR}/common/*.hh)

add_subdirectory(basic)
add_subdirectory(gpu)
