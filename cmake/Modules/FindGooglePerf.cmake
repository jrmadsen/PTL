# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

IF (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (GooglePerf_CHECK_COMPONENTS FALSE)
ELSE (CMAKE_VERSION VERSION_GREATER 2.8.7)
  SET (GooglePerf_CHECK_COMPONENTS TRUE)
ENDIF (CMAKE_VERSION VERSION_GREATER 2.8.7)

#------------------------------------------------------------------------------#

IF("${GooglePerf_FIND_COMPONENTS}" STREQUAL "")
    IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        LIST(APPEND GooglePerf_FIND_COMPONENTS profiler tcmalloc_debug)
    ELSE()
        LIST(APPEND GooglePerf_FIND_COMPONENTS profiler tcmalloc)
    ENDIF()
ENDIF("${GooglePerf_FIND_COMPONENTS}" STREQUAL "")

#------------------------------------------------------------------------------#

# Component options
set(_GooglePerf_COMPONENT_OPTIONS
    profiler
    tcmalloc
    tcmalloc_and_profiler
    tcmalloc_debug
    tcmalloc_minimal
    tcmalloc_minimal_debug
)

#------------------------------------------------------------------------------#

set(_GooglePerf_POSSIBLE_LIB_SUFFIXES lib lib64 lib32)

#------------------------------------------------------------------------------#

find_path(GooglePerf_ROOT_DIR
    NAMES
        include/gperftools/profiler.h
        include/google/profiler.h
    DOC "Google perftools root directory")

#------------------------------------------------------------------------------#

find_path(GooglePerf_INCLUDE_DIR
    NAMES
        gperftools/profiler.h
        google/profiler.h
    HINTS
        ${GooglePerf_ROOT_DIR}
    PATH_SUFFIXES
        include
    DOC "Google perftools profiler include directory")

#------------------------------------------------------------------------------#
# Find components

FOREACH (_GooglePerf_COMPONENT ${GooglePerf_FIND_COMPONENTS})
    IF(NOT "${_GooglePerf_COMPONENT_OPTIONS}" MATCHES "${_GooglePerf_COMPONENT}")
        MESSAGE(WARNING "${_GooglePerf_COMPONENT} is not listed as a real component")
    ENDIF()

    STRING (TOUPPER ${_GooglePerf_COMPONENT} _GooglePerf_COMPONENT_UPPER)
    SET (_GooglePerf_LIBRARY_BASE GooglePerf_${_GooglePerf_COMPONENT_UPPER}_LIBRARY)

    SET (_GooglePerf_LIBRARY_NAME ${_GooglePerf_COMPONENT})

    FIND_LIBRARY (${_GooglePerf_LIBRARY_BASE}
        NAMES ${_GooglePerf_LIBRARY_NAME}
        HINTS ${GooglePerf_ROOT_DIR}
        PATH_SUFFIXES ${_GooglePerf_POSSIBLE_LIB_SUFFIXES}
        DOC "MKL ${_GooglePerf_COMPONENT} library")

    MARK_AS_ADVANCED (${_GooglePerf_LIBRARY_BASE})

    SET (GooglePerf_${_GooglePerf_COMPONENT_UPPER}_FOUND TRUE)

    IF (NOT ${_GooglePerf_LIBRARY_BASE})
        # Component missing: record it for a later report
        LIST (APPEND _GooglePerf_MISSING_COMPONENTS ${_GooglePerf_COMPONENT})
        SET (GooglePerf_${_GooglePerf_COMPONENT_UPPER}_FOUND FALSE)
    ENDIF (NOT ${_GooglePerf_LIBRARY_BASE})

    SET (GooglePerf_${_GooglePerf_COMPONENT}_FOUND
        ${GooglePerf_${_GooglePerf_COMPONENT_UPPER}_FOUND})

    IF (${_GooglePerf_LIBRARY_BASE})
        # setup the GooglePerf_<COMPONENT>_LIBRARIES variable
        SET (GooglePerf_${_GooglePerf_COMPONENT_UPPER}_LIBRARIES
            ${${_GooglePerf_LIBRARY_BASE}})
        LIST (APPEND GooglePerf_LIBRARIES ${${_GooglePerf_LIBRARY_BASE}})
    ELSE (${_GooglePerf_LIBRARY_BASE})
        LIST (APPEND _GooglePerf_MISSING_LIBRARIES ${_GooglePerf_LIBRARY_BASE})
    ENDIF (${_GooglePerf_LIBRARY_BASE})

ENDFOREACH (_GooglePerf_COMPONENT ${GooglePerf_FIND_COMPONENTS})


#----- Missing components
IF (DEFINED _GooglePerf_MISSING_COMPONENTS AND _GooglePerf_CHECK_COMPONENTS)
    IF (NOT GooglePerf_FIND_QUIETLY)
        MESSAGE (STATUS "One or more MKL components were not found:")
        # Display missing components indented, each on a separate line
        FOREACH (_GooglePerf_MISSING_COMPONENT ${_GooglePerf_MISSING_COMPONENTS})
            MESSAGE (STATUS "  " ${_GooglePerf_MISSING_COMPONENT})
        ENDFOREACH (_GooglePerf_MISSING_COMPONENT ${_GooglePerf_MISSING_COMPONENTS})
    ENDIF (NOT GooglePerf_FIND_QUIETLY)
ENDIF (DEFINED _GooglePerf_MISSING_COMPONENTS AND _GooglePerf_CHECK_COMPONENTS)

#------------------------------------------------------------------------------#

mark_as_advanced(GooglePerf_INCLUDE_DIR)
find_package_handle_standard_args(GooglePerf DEFAULT_MSG
    GooglePerf_ROOT_DIR
    GooglePerf_INCLUDE_DIR ${_GooglePerf_MISSING_LIBRARIES})
