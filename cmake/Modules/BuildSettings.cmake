################################################################################
#
#        Handles the build settings
#
################################################################################

include(GNUInstallDirs)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(IS_SUBPROJECT OFF)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(IS_SUBPROJECT ON)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

# ---------------------------------------------------------------------------- #
# check C flag
macro(add_c_if_avail VAR FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        string(REGEX REPLACE "^-" "c_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
        check_c_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            add(PROJECT_C_FLAGS "${FLAG}")
        endif()
    endif()
endmacro()


# ---------------------------------------------------------------------------- #
# check CXX flag
macro(add_cxx_if_avail VAR FLAG)
    if(NOT "${FLAG}" STREQUAL "")
        string(REGEX REPLACE "^-" "cxx_" FLAG_NAME "${FLAG}")
        string(REPLACE "-" "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE " " "_" FLAG_NAME "${FLAG_NAME}")
        string(REPLACE "=" "_" FLAG_NAME "${FLAG_NAME}")
        check_cxx_compiler_flag("${FLAG}" ${FLAG_NAME})
        if(${FLAG_NAME})
            add(PROJECT_CXX_FLAGS "${FLAG}")
        endif()
    endif()
endmacro()


# ---------------------------------------------------------------------------- #
if(NOT IS_SUBPROJECT)

    # Special Intel compiler flags for NERSC Cori
    foreach(_LANG C CXX)
        if(CMAKE_${_LANG}_COMPILER_IS_INTEL)
            if(INTEL_${_LANG}_AVX512)
                set(INTEL_${_LANG}_COMPILER_FLAGS "-xHOST -axMIC-AVX512")
            else(INTEL_${_LANG}_AVX512)
                set(INTEL_${_LANG}_COMPILER_FLAGS "-xHOST")
            endif(INTEL_${_LANG}_AVX512)
            add_feature(INTEL_${_LANG}_COMPILER_FLAGS "Intel ${_LANG} compiler flags")
        endif(CMAKE_${_LANG}_COMPILER_IS_INTEL)
    endforeach(_LANG C CXX)

    set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    if(WIN32)
        set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
    else(WIN32)
        set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
    endif(WIN32)

    if(GOOD_CMAKE)
        set(CMAKE_INSTALL_MESSAGE LAZY)
    endif(GOOD_CMAKE)

    # ensure only C++11, C++14, or C++17
    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "11" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "14" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "17" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1y" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1z")

        if(WIN32)
            set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard" FORCE)
        else(WIN32)
            set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard" FORCE)
        endif(WIN32)

    endif(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "11" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "14" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "17" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1y" AND
        NOT "${CMAKE_CXX_STANDARD}" STREQUAL "1z")

    if(CMAKE_CXX_COMPILER_IS_GNU)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
    elseif(CMAKE_CXX_COMPILER_IS_CLANG)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD} -stdlib=libc++")
    elseif(CMAKE_CXX_COMPILER_IS_INTEL)
        add(CMAKE_CXX_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
    elseif(CMAKE_CXX_COMPILER_IS_PGI)
        add(CMAKE_CXX_FLAGS "--c++${CMAKE_CXX_STANDARD} -A")
    elseif(CMAKE_CXX_COMPILER_IS_XLC)
        if(CMAKE_CXX_STANDARD GREATER 11)
            add(CMAKE_CXX_FLAGS "-std=c++1y")
        else(CMAKE_CXX_STANDARD GREATER 11)
            add(CMAKE_CXX_FLAGS "-std=c++11")
        endif(CMAKE_CXX_STANDARD GREATER 11)
    elseif(CMAKE_CXX_COMPILER_IS_MSVC)
        add(CMAKE_CXX_FLAGS "-std:c++${CMAKE_CXX_STANDARD}")
    endif(CMAKE_CXX_COMPILER_IS_GNU)

endif(NOT IS_SUBPROJECT)


# ---------------------------------------------------------------------------- #
# set the output directory (critical on Windows

foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if ${PROJECT_NAME}_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
        set(${PROJECT_NAME}_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)


# ---------------------------------------------------------------------------- #
# used by configure_package_*
#
set(LIBNAME ptl)

# ---------------------------------------------------------------------------- #
#  debug macro
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DNDEBUG)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    include(Coverage)
    add_c_if_avail(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_COVERAGE}")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_COVERAGE}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

# ---------------------------------------------------------------------------- #
# set the compiler flags
#
add_c_if_avail(CMAKE_C_FLAGS "-W")
add_c_if_avail(CMAKE_C_FLAGS "-Wall")
add_c_if_avail(CMAKE_C_FLAGS "-Wextra")
add_c_if_avail(CMAKE_C_FLAGS "-std=c11")
if(NOT c_std_c11)
    add_c_if_avail(CMAKE_C_FLAGS "-std=c99")
endif()

# pthreads
add_c_if_avail(CMAKE_C_FLAGS "-pthread")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-pthread")

# SIMD OpenMP
add_c_if_avail(CMAKE_C_FLAGS "-fopenmp-simd")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-fopenmp-simd")

# general warnings
add_cxx_if_avail(CMAKE_CXX_FLAGS "-W")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-Wall")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-Wextra")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-Wshadow")
add_cxx_if_avail(CMAKE_CXX_FLAGS "-faligned-new")

if(USE_ARCH)
    if(CMAKE_C_COMPILER_IS_INTEL)
        add_c_if_avail(CMAKE_C_FLAGS "-xHOST")
    else()
        add_c_if_avail(CMAKE_C_FLAGS "-march")
        add_c_if_avail(CMAKE_C_FLAGS "-msse2")
        add_c_if_avail(CMAKE_C_FLAGS "-msse3")
        add_c_if_avail(CMAKE_C_FLAGS "-msse4")
        add_c_if_avail(CMAKE_C_FLAGS "-mavx")
        add_c_if_avail(CMAKE_C_FLAGS "-mavx2")
    endif()

    if(CMAKE_CXX_COMPILER_IS_INTEL)
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-xHOST")
    else()
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-march")
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-msse2")
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-msse3")
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-msse4")
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx")
        add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx2")
    endif()

    if(USE_AVX512)
        if(CMAKE_C_COMPILER_IS_INTEL)
            add_c_if_avail(CMAKE_C_FLAGS "-axMIC-AVX512")
        else()
            add_c_if_avail(CMAKE_C_FLAGS "-mavx512f")
            add_c_if_avail(CMAKE_C_FLAGS "-mavx512pf")
            add_c_if_avail(CMAKE_C_FLAGS "-mavx512er")
            add_c_if_avail(CMAKE_C_FLAGS "-mavx512cd")
        endif()

        if(CMAKE_CXX_COMPILER_IS_INTEL)
            add_cxx_if_avail(CMAKE_CXX_FLAGS "-axMIC-AVX512")
        else()
            add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx512f")
            add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx512pf")
            add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx512er")
            add_cxx_if_avail(CMAKE_CXX_FLAGS "-mavx512cd")
        endif()
    endif()
endif()

if(PTL_USE_SANITIZER)
    add_c_if_avail(CMAKE_C_FLAGS "-fsanitize=leak")
    add_c_if_avail(CMAKE_C_FLAGS "-fsanitize=address")
    add_c_if_avail(CMAKE_C_FLAGS "-fsanitize=all")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-fsanitize=leak")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-fsanitize=address")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-fsanitize=all")
endif()

if(PTL_USE_PROFILE)
    add_c_if_avail(CMAKE_C_FLAGS "-p")
    add_c_if_avail(CMAKE_C_FLAGS "-pg")
    add_c_if_avail(CMAKE_C_FLAGS "-fbranch-probabilities")
    if(c_fbranch_probabilities)
        add(PROJECT_C_FLAGS "-fprofile-arcs")
        add(PROJECT_C_FLAGS "-fprofile-dir=${CMAKE_BINARY_DIR}")
        add(PROJECT_C_FLAGS "-fprofile-generate=${CMAKE_BINARY_DIR}/profile")
    endif()
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-p")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-pg")
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-fbranch-probabilities")
    if(cxx_fbranch_probabilities)
        add(PROJECT_CXX_FLAGS "-fprofile-arcs")
        add(PROJECT_CXX_FLAGS "-fprofile-dir=${CMAKE_BINARY_DIR}")
        add(PROJECT_CXX_FLAGS "-fprofile-generate=${CMAKE_BINARY_DIR}/profile")
        add(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs")
        add_feature(CMAKE_EXE_LINKER_FLAGS "Linker flags")
    endif()
endif()

if(PTL_USE_COVERAGE)
    add_c_if_avail(CMAKE_C_FLAGS "-ftest-coverage")
    if(c_ftest_coverage)
        add(PROJECT_C_FLAGS "-fprofile-arcs")
        add(PROJECT_C_FLAGS "-fprofile-dir=${CMAKE_BINARY_DIR}")
    endif()
    add_cxx_if_avail(CMAKE_CXX_FLAGS "-ftest-coverage")
    if(cxx_ftest_coverage)
        add(PROJECT_CXX_FLAGS "-fprofile-arcs")
        add(PROJECT_CXX_FLAGS "-fprofile-dir=${CMAKE_BINARY_DIR}")

        #add(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs")
        add_feature(CMAKE_EXE_LINKER_FLAGS "Linker flags")
    endif()
endif()

# user customization
add_c_if_avail(CMAKE_C_FLAGS "${CFLAGS}")
add_c_if_avail(CMAKE_C_FLAGS "$ENV{CFLAGS}")
add_cxx_if_avail(CMAKE_CXX_FLAGS "${CXXFLAGS}")
add_cxx_if_avail(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS}")

# remove duplicates
add_c_flags(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
add_cxx_flags(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# message
add(CMAKE_C_FLAGS "${PROJECT_C_FLAGS}")
add(CMAKE_CXX_FLAGS "${PROJECT_CXX_FLAGS}")
