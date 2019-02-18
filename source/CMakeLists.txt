
cmake_policy(SET CMP0053 NEW)
include(CheckCXXCompilerFlag)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${TiMemory_INCLUDE_DIRS})

if(PTL_USE_CLANG_TIDY AND CLANG_TIDY)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};checks=-*,readability-*")
endif()

################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################

if(PTL_USE_TBB)
    add_definitions(-DPTL_USE_TBB)
endif(PTL_USE_TBB)

if(PTL_USE_TIMEMORY)
    add_definitions(-DPTL_USE_TIMEMORY)
endif(PTL_USE_TIMEMORY)

# libraries to install
set(INSTALL_LIBRARIES )

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB ptl_headers ${CMAKE_CURRENT_LIST_DIR}/PTL/*.hh
                      ${CMAKE_CURRENT_LIST_DIR}/PTL/*.icc)
file(GLOB ptl_sources ${CMAKE_CURRENT_LIST_DIR}/*.cc)
# set language
set_source_files_properties(${ptl_sources} PROPERTIES LANGUAGE CXX)


################################################################################
#                                                                              #
#                              PTL Library                                     #
#                                                                              #
################################################################################
if(BUILD_SHARED_LIBS)
    build_library(VERSION
        TYPE SHARED
        TARGET_NAME ptl-shared
        OUTPUT_NAME ptl
        SOURCES ${ptl_headers} ${ptl_sources}
        LINK_LIBRARIES ${EXTERNAL_LIBRARIES}
        EXTRA_ARGS POSITION_INDEPENDENT_CODE ON)
    list(APPEND INSTALL_LIBRARIES ptl-shared)
endif()

if(BUILD_STATIC_LIBS)
    build_library(
        TYPE STATIC
        TARGET_NAME ptl-static
        OUTPUT_NAME ptl
        SOURCES ${ptl_headers} ${ptl_sources}
        LINK_LIBRARIES ${EXTERNAL_LIBRARIES})
    if(PTL_DEVELOPER_INSTALL)
        list(APPEND INSTALL_LIBRARIES ptl-static)
    endif()
endif()


################################################################################
#                                                                              #
#                              Installation                                    #
#                                                                              #
################################################################################
#------------------------------------------------------------------------------#
# Install the targets and export
#
if(PTL_DEVELOPER_INSTALL OR BUILD_SHARED_LIBS)
    # libraries
    install(TARGETS ${INSTALL_LIBRARIES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}LibraryDepends
        COMPONENT development)

    # install export
    install(EXPORT ${PROJECT_NAME}LibraryDepends
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
        COMPONENT development)
endif()

if(PTL_DEVELOPER_INSTALL)
    # headers
    install(FILES ${ptl_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/PTL
        COMPONENT development)
endif()

# generate export for build tree
export(TARGETS ${INSTALL_LIBRARIES}
    FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Build.cmake)
