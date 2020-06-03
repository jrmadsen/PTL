#
#   Find packages
#

include(FindPackageHandleStandardArgs)
include(MacroUtilities)

ptl_add_interface_library(ptl-external-packages)
ptl_add_interface_library(ptl-threads)
ptl_add_interface_library(ptl-tbb)
ptl_add_interface_library(ptl-cuda)

################################################################################
#
#                               Threads
#
################################################################################

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
endif()

find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(ptl-threads INTERFACE Threads::Threads)
    target_link_libraries(ptl-external-packages INTERFACE ptl-threads)
endif()


################################################################################
#
#        TBB
#
################################################################################

if(PTL_USE_TBB)
    find_package(TBB)

    if(TBB_FOUND)
        target_compile_definitions(ptl-tbb INTERFACE PTL_USE_TBB)
        target_include_directories(ptl-tbb SYSTEM INTERFACE ${TBB_INCLUDE_DIRS})
        target_link_libraries(ptl-tbb INTERFACE ${TBB_LIBRARIES})
        target_link_libraries(ptl-external-packages INTERFACE ptl-tbb)
    endif()

endif()


################################################################################
#
#        CUDA
#
################################################################################

if(PTL_USE_CUDA AND PTL_USE_GPU)
    find_package(CUDA)

    if(CUDA_FOUND)
        foreach(_DIR ${CUDA_INCLUDE_DIRS})
            target_include_directories(ptl-cuda SYSTEM INTERFACE ${_DIR})
        endforeach()

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
                target_link_libraries(ptl-cuda INTERFACE ${${NAME}_LIBRARY})
            endif()
        endforeach()

        add(CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH}")
        target_compile_definitions(ptl-cuda INTERFACE PTL_USE_CUDA)

    endif()

endif()
