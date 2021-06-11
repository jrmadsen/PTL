# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

set(GPerfTools_CHECK_COMPONENTS TRUE)
if(CMAKE_VERSION VERSION_GREATER 2.8.7)
  set(GPerfTools_CHECK_COMPONENTS FALSE)
endif()

#------------------------------------------------------------------------------#

# Component options
set(_GPerfTools_COMPONENT_OPTIONS
    profiler
    tcmalloc
    tcmalloc_and_profiler
    tcmalloc_debug
    tcmalloc_minimal
    tcmalloc_minimal_debug
)

#------------------------------------------------------------------------------#

if("${GPerfTools_FIND_COMPONENTS}" STREQUAL "")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        list(APPEND GPerfTools_FIND_COMPONENTS profiler tcmalloc_debug)
    else()
        list(APPEND GPerfTools_FIND_COMPONENTS tcmalloc_and_profiler)
    endif()
endif()

#------------------------------------------------------------------------------#

set(_GPerfTools_POSSIBLE_LIB_SUFFIXES lib lib64 lib32)

#------------------------------------------------------------------------------#

find_path(GPerfTools_ROOT_DIR
    NAMES
        include/gperftools/tcmalloc.h
        include/google/tcmalloc.h
        include/gperftools/profiler.h
        include/google/profiler.h
    DOC "Google perftools root directory")

#------------------------------------------------------------------------------#

find_path(GPerfTools_INCLUDE_DIR
    NAMES
        gperftools/tcmalloc.h
        google/tcmalloc.h
        gperftools/profiler.h
        google/profiler.h
    HINTS
        ${GPerfTools_ROOT_DIR}
    PATH_SUFFIXES
        include
    DOC "Google perftools profiler include directory")

#------------------------------------------------------------------------------#

set(GPerfTools_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIR})

#------------------------------------------------------------------------------#
# Find components

foreach(_GPerfTools_COMPONENT ${GPerfTools_FIND_COMPONENTS})
    if(NOT "${_GPerfTools_COMPONENT_OPTIONS}" MATCHES "${_GPerfTools_COMPONENT}")
        message(WARNING "${_GPerfTools_COMPONENT} is not listed as a real component")
    endif()

    string(TOUPPER ${_GPerfTools_COMPONENT} _GPerfTools_COMPONENT_UPPER)
    set(_GPerfTools_LIBRARY_BASE GPerfTools_${_GPerfTools_COMPONENT_UPPER}_LIBRARY)

    set(_GPerfTools_LIBRARY_NAME ${_GPerfTools_COMPONENT})

    find_library(${_GPerfTools_LIBRARY_BASE}
        NAMES ${_GPerfTools_LIBRARY_NAME}
        HINTS ${GPerfTools_ROOT_DIR}
        PATH_SUFFIXES ${_GPerfTools_POSSIBLE_LIB_SUFFIXES}
        DOC "MKL ${_GPerfTools_COMPONENT} library")

    mark_as_advanced(${_GPerfTools_LIBRARY_BASE})

    set(GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND TRUE)

    if(NOT ${_GPerfTools_LIBRARY_BASE})
        # Component missing: record it for a later report
        list(APPEND _GPerfTools_MISSING_COMPONENTS ${_GPerfTools_COMPONENT})
        set(GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND FALSE)
    endif()

    set(GPerfTools_${_GPerfTools_COMPONENT}_FOUND
        ${GPerfTools_${_GPerfTools_COMPONENT_UPPER}_FOUND})

    if(${_GPerfTools_LIBRARY_BASE})
        # setup the GPerfTools_<COMPONENT>_LIBRARIES variable
        set(GPerfTools_${_GPerfTools_COMPONENT_UPPER}_LIBRARIES
            ${${_GPerfTools_LIBRARY_BASE}})
        list(APPEND GPerfTools_LIBRARIES ${${_GPerfTools_LIBRARY_BASE}})
    else()
        list(APPEND _GPerfTools_MISSING_LIBRARIES ${_GPerfTools_LIBRARY_BASE})
    endif()
endforeach()


#----- Missing components
if(DEFINED _GPerfTools_MISSING_COMPONENTS AND _GPerfTools_CHECK_COMPONENTS)
    if(NOT GPerfTools_FIND_QUIETLY)
        message(STATUS "One or more MKL components were not found:")
        # Display missing components indented, each on a separate line
        foreach(_GPerfTools_MISSING_COMPONENT ${_GPerfTools_MISSING_COMPONENTS})
            message(STATUS "  " ${_GPerfTools_MISSING_COMPONENT})
        endforeach()
    endif()
endif()
#------------------------------------------------------------------------------#

mark_as_advanced(GPerfTools_INCLUDE_DIR)
find_package_handle_standard_args(GPerfTools DEFAULT_MSG
    GPerfTools_ROOT_DIR
    GPerfTools_INCLUDE_DIRS ${_GPerfTools_MISSING_LIBRARIES})
