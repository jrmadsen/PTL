
cmake_policy(SET CMP0053 NEW)
include(CheckCXXCompilerFlag)

include_directories(${CMAKE_CURRENT_LIST_DIR})

execute_process(COMMAND
    ${CMAKE_COMMAND} -E remove -f
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake)


################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################

# libraries to install
set(INSTALL_LIBRARIES )

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB_RECURSE ptl_headers ${CMAKE_CURRENT_LIST_DIR}/PTL/*.hh
                              ${CMAKE_CURRENT_LIST_DIR}/PTL/*.icc)
file(GLOB_RECURSE ptl_sources ${CMAKE_CURRENT_LIST_DIR}/*.cc)
# set language
set_source_files_properties(${ptl_sources} PROPERTIES LANGUAGE CXX)

get_property(PTL_INTERFACE_LIBRARIES GLOBAL PROPERTY ${PROJECT_NAME}_INTERFACE_LIBRARIES)

################################################################################
#                                                                              #
#                              PTL Library                                     #
#                                                                              #
################################################################################

if(BUILD_SHARED_LIBS)

    ptl_build_library(VERSION
        TYPE SHARED
        TARGET_NAME ptl-shared
        OUTPUT_NAME ptl
        SOURCES ${ptl_headers} ${ptl_sources}
        EXTRA_ARGS POSITION_INDEPENDENT_CODE ON
                   WINDOWS_EXPORT_ALL_SYMBOLS ON)

    target_link_libraries(ptl-shared PUBLIC ptl-external-packages)
    target_link_libraries(ptl-shared PRIVATE ptl-external-libraries ptl-compile-options)

    target_include_directories(ptl-shared INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

    list(APPEND INSTALL_LIBRARIES ptl-shared)
    export(TARGETS ptl-shared ${PTL_INTERFACE_LIBRARIES}
        FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-shared.cmake)
endif()

if(BUILD_STATIC_LIBS)

    ptl_build_library(
        TYPE STATIC
        TARGET_NAME ptl-static
        OUTPUT_NAME ptl
        SOURCES ${ptl_headers} ${ptl_sources}
        EXTRA_ARGS POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(ptl-static PUBLIC ptl-external-packages)
    target_link_libraries(ptl-static PRIVATE ptl-external-libraries ptl-compile-options)

    target_include_directories(ptl-static INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

    list(APPEND INSTALL_LIBRARIES ptl-static)
    export(TARGETS ptl-static ${PTL_INTERFACE_LIBRARIES}
        FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-static.cmake)
endif()


################################################################################
#                                                                              #
#                              Installation                                    #
#                                                                              #
################################################################################

# install export
install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE PTL::
    DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
    COMPONENT development)

# headers
install(FILES ${ptl_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/PTL
    COMPONENT development)

# generate export for build tree
