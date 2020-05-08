################################################################################
#
#        Handles the build settings
#
################################################################################
#
set(LIBNAME ptl)

include(GNUInstallDirs)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(Compilers)
include(MacroUtilities)

ptl_add_interface_library(ptl-compile-options)
ptl_add_interface_library(ptl-external-libraries)

# ---------------------------------------------------------------------------- #
#
set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
set(CMAKE_INSTALL_MESSAGE LAZY)
if(WIN32)
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
else(WIN32)
    set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
endif(WIN32)

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
#  debug macro
#
string(TOUPPER "${CMAKE_BUILD_TYPE}" UC_BUILD_TYPE)
if("${UC_BUILD_TYPE}" STREQUAL "DEBUG")
    target_compile_definitions(ptl-compile-options INTERFACE DEBUG)
else()
    target_compile_definitions(ptl-compile-options INTERFACE NDEBUG)
endif()

# ---------------------------------------------------------------------------- #
# set the compiler flags
#
add_c_flag_if_avail("-W")
if(NOT WIN32)
    add_c_flag_if_avail("-Wall")
endif()
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-std=c11")
if(NOT c_std_c11)
    add_c_flag_if_avail("-std=c99")
endif()

# general warnings
add_cxx_flag_if_avail("-W")
if(NOT WIN32)
    add_cxx_flag_if_avail("-Wall")
endif()
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wshadow")
add_cxx_flag_if_avail("-faligned-new")

if(PTL_USE_SANITIZER)
    add_target_flag_if_avail(ptl-external-libraries "-fsanitize=${PTL_SANITIZER_TYPE}")
    if(cxx_fsanitize_${PTL_SANITIZER_TYPE})
        if("${PTL_SANITIZER_TYPE}" STREQUAL "leak")
            target_link_libraries(ptl-external-libraries INTERFACE lsan)
        elseif("${PTL_SANITIZER_TYPE}" STREQUAL "address")
            target_link_libraries(ptl-external-libraries INTERFACE asan)
        elseif("${PTL_SANITIZER_TYPE}" STREQUAL "memory")
            target_link_libraries(ptl-external-libraries INTERFACE msan)
        elseif("${PTL_SANITIZER_TYPE}" STREQUAL "thread")
            target_link_libraries(ptl-external-libraries INTERFACE tsan)
        endif()
    endif()
endif()

if(PTL_USE_PROFILE)
    add_c_flag_if_avail("-p")
    add_c_flag_if_avail("-pg")
    add_c_flag_if_avail("-fbranch-probabilities")
    if(c_fbranch_probabilities)
        add_target_c_flag(ptl-compile-options "-fprofile-arcs")
    endif()
    add_cxx_flag_if_avail("-p")
    add_cxx_flag_if_avail("-pg")
    add_cxx_flag_if_avail("-fbranch-probabilities")
    if(cxx_fbranch_probabilities)
        add_target_cxx_flag(ptl-compile-options "-fprofile-arcs")
        target_link_options(ptl-compile-options INTERFACE "-fprofile-arcs")
    endif()
endif()

if(PTL_USE_COVERAGE)
    add_c_flag_if_avail("-ftest-coverage")
    if(c_ftest_coverage)
        list(APPEND ${PROJECT_NAME}_C_FLAGS "-fprofile-arcs")
    endif()
    add_cxx_flag_if_avail("-ftest-coverage")
    if(cxx_ftest_coverage)
        add_target_cxx_flag(ptl-compile-options "-fprofile-arcs")
        target_link_options(ptl-compile-options INTERFACE "-fprofile-arcs")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# user customization
to_list(_CFLAGS "${CFLAGS};$ENV{CFLAGS}")
foreach(_FLAG ${_CFLAGS})
    add_c_flag_if_avail("${_FLAG}")
endforeach()

to_list(_CXXFLAGS "${CXXFLAGS};$ENV{CXXFLAGS}")
foreach(_FLAG ${_CXXFLAGS})
    add_cxx_flag_if_avail("${_FLAG}")
endforeach()
