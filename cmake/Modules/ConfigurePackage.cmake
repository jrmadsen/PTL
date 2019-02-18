
################################################################################
#
#        PTL Package installation
#
################################################################################

if(NOT PTL_DEVELOPER_INSTALL)
    return()
endif()

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/MacroUtilities.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/Packages.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/FindGPerfTools.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/FindTBB.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/Findittnotify.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/CudaConfig.cmake
    DESTINATION ${CMAKE_INSTALL_CONFIGDIR})

if(PTL_BUILD_DOCS)
    # documentation
    set_property(GLOBAL PROPERTY PTL_DOCUMENTATION_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/PTL)

    set(EXCLUDE_LIST )

    include(Documentation)

    SET(CMAKE_INSTALL_MESSAGE NEVER)
    Generate_Documentation(Doxyfile.${PROJECT_NAME})
    SET(CMAKE_INSTALL_MESSAGE LAZY)

endif()
