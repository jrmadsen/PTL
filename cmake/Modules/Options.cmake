
################################################################################
#
#        TiMemory Options
#
################################################################################

include(MacroUtilities)

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(CMAKE_C_FLAGS "C compiler flags")
add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")

# options (always available)
add_option(BUILD_STATIC_LIBS "Build static library" ON)
add_option(BUILD_SHARED_LIBS "Build shared library" ON)
add_option(PTL_BUILD_TESTING "Enable testing" OFF)
add_option(PTL_BUILD_EXAMPLES "Build examples" OFF)
add_option(PTL_BUILD_DOCS "Build documentation with Doxygen" OFF)

add_option(PTL_USE_TBB "Enable TBB" ON)
add_option(PTL_USE_GPU "Enable GPU preprocessor" OFF)
add_option(PTL_USE_GPERF "Enable gperftools" OFF)
add_option(PTL_USE_ITTNOTIFY "Enable ittnotify library for VTune" OFF)
add_option(PTL_USE_TIMEMORY "Enable TiMemory for timing+memory analysis" OFF)
add_option(PTL_USE_ARCH "Enable architecture specific flags" OFF)
add_option(PTL_USE_AVX512 "Enable AVX-512 flags (if available)" OFF)
add_option(PTL_USE_SANITIZER "Enable -fsanitize=<type>" OFF)
add_option(PTL_USE_CLANG_TIDY "Enable running clang-tidy on" OFF)
add_option(PTL_USE_COVERAGE "Enable code coverage" OFF)
add_option(PTL_USE_PROFILE "Enable profiling" OFF)

if(PTL_USE_SANITIZER)
    add_feature(PTL_SANITITZER_TYPE "Sanitizer type (-fsanitize=<type>)")
    set(PTL_SANITITZER_TYPE leak CACHE STRING "Sanitizer type (-fsanitize=<type>)")
endif()

if(PTL_USE_CLANG_TIDY)
    find_program(CLANG_TIDY
        NAMES clang-tidy)
    add_feature(CLANG_TIDY "Path to clang-tidy")
endif()

if(PTL_USE_GPU)
    add_definitions(-DPTL_USE_GPU)

    # default settings
    set(CUDA_FOUND OFF)

    # possible options (sometimes available)
    find_package(CUDA QUIET)

    if(CUDA_FOUND)
        set(_USE_CUDA ON)
    else()
        set(_USE_CUDA OFF)
    endif()

    add_option(PTL_USE_CUDA "Enable CUDA option for GPU execution" ${_USE_CUDA})
    add_option(PTL_USE_NVTX "Enable NVTX for Nsight" ${_USE_CUDA})
    add_option(CUDA_SEPARABLE_COMPILATION "Enable separable compilation" OFF)

    set(CUDA_ARCH "sm_35" CACHE STRING "CUDA architecture flag")
    add_feature(CUDA_ARCH "CUDA architecture (e.g. sm_35)")
    add_feature(CUDA_GENERATED_OUTPUT_DIR "CUDA output directory for generated files")

    if(PTL_USE_CUDA)
        # find the cuda compiler
        find_program(CMAKE_CUDA_COMPILER nvcc
            PATHS /usr/local/cuda
            HINTS /usr/local/cuda
            PATH_SUFFIXES bin)
        if(CMAKE_CUDA_COMPILER)
            include(CudaConfig)
        endif(CMAKE_CUDA_COMPILER)
    endif(PTL_USE_CUDA)

endif(PTL_USE_GPU)

if(APPLE)
    add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Hardcode installation rpath based on link path" ON)
endif()
