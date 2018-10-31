#
#   Find packages
#

include(FindPackageHandleStandardArgs)
include("${CMAKE_CURRENT_LIST_DIR}/MacroUtilities.cmake")

################################################################################
#
#                               Threads
#
################################################################################

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

find_package(Threads)
if(Threads_FOUND)
    list(APPEND EXTERNAL_LIBRARIES Threads::Threads)
endif()


################################################################################
#
#                               TiMemory
#
################################################################################

if(PTL_USE_TIMEMORY)
    find_package(TiMemory)

    if(TiMemory_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TiMemory_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES
            ${TiMemory_LIBRARIES})
        add_definitions(-DPTL_USE_TIMEMORY)
    endif(TiMemory_FOUND)

endif(PTL_USE_TIMEMORY)


################################################################################
#
#        Google PerfTools
#
################################################################################

if(PTL_USE_GPERF)
    find_package(GPerfTools)

    if(GPerfTools_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GPerfTools_LIBRARIES})
        add_definitions(-DPTL_USE_GPERF)
    endif(GPerfTools_FOUND)

endif(PTL_USE_GPERF)


################################################################################
#
#        TBB
#
################################################################################

if(PTL_USE_TBB)
    find_package(TBB)

    if(TBB_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${TBB_LIBRARIES})
        add_definitions(-DPTL_USE_TBB)
    endif(TBB_FOUND)

endif(PTL_USE_TBB)


################################################################################
#
#        CUDA
#
################################################################################

if(PTL_USE_CUDA AND PTL_USE_GPU)
    find_package(CUDA)

    if(CUDA_FOUND)
        foreach(_DIR ${CUDA_INCLUDE_DIRS})
            include_directories(SYSTEM ${_DIR})
        endforeach(_DIR ${CUDA_INCLUDE_DIRS})

        find_library(CUDA_LIBRARY
            NAMES cuda
            PATHS /usr/local/cuda
            HINTS /usr/local/cuda
            PATH_SUFFIXES lib lib64)

        find_library(CUDART_LIBRARY
            NAMES cudart
            PATHS /usr/local/cuda
            HINTS /usr/local/cuda
            PATH_SUFFIXES lib lib64)

        find_library(CUDART_STATIC_LIBRARY
            NAMES cudart_static
            PATHS /usr/local/cuda
            HINTS /usr/local/cuda
            PATH_SUFFIXES lib lib64)

        find_library(RT_LIBRARY
            NAMES rt
            PATHS /usr /usr/local /opt/local
            HINTS /usr /usr/local /opt/local
            PATH_SUFFIXES lib lib64)

        find_library(DL_LIBRARY
            NAMES dl
            PATHS /usr /usr/local /opt/local
            HINTS /usr /usr/local /opt/local
            PATH_SUFFIXES lib lib64)

        foreach(NAME CUDA CUDART CUDART_STATIC RT DL)
            if(${NAME}_LIBRARY)
                list(APPEND EXTERNAL_CUDA_LIBRARIES ${${NAME}_LIBRARY})
            endif()
        endforeach()

        add(CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH}")
        add_definitions(-DPTL_USE_CUDA)

        if(PTL_USE_NVTX)
            find_library(NVTX_LIBRARY
                NAMES nvToolsExt
                PATHS /usr/local/cuda
                HINTS /usr/local/cuda
                PATH_SUFFIXES lib lib64)
            if(NVTX_LIBRARY)
                add_definitions(-DPTL_USE_NVTX)
                list(APPEND EXTERNAL_CUDA_LIBRARIES ${NVTX_LIBRARY})
            endif()
        endif()

    endif()

endif(PTL_USE_CUDA AND PTL_USE_GPU)


################################################################################
#
#        ITTNOTIFY (for VTune)
#
################################################################################
if(PTL_USE_ITTNOTIFY)

    find_package(ittnotify)

    if(ittnotify_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${ITTNOTIFY_LIBRARIES})
        add_definitions(-DPTL_USE_ITTNOTIFY)
    else()
        message(WARNING "ittnotify not found. Set \"VTUNE_AMPLIFIER_DIR\" in environment")
    endif()

endif()


################################################################################
#
#        External variables
#
################################################################################

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach(_DIR ${EXTERNAL_INCLUDE_DIRS})


