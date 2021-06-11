#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(PTL_Minimal_Examples LANGUAGES C CXX)


#----------------------------------------------------------------------------
# Find PTL project
#
find_package(PTL REQUIRED)

#----------------------------------------------------------------------------
# standard example
#
add_executable(ptl-minimal minimal.cc)
target_link_libraries(ptl-minimal PRIVATE PTL::ptl)
