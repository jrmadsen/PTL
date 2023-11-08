# SPDX-FileCopyrightText: 2003-2018 University of Illinois at Urbana-Champaign.
#
# SPDX-License-Identifier: BSD-3-Clause

#[=======================================================================[.rst:
CheckAtomic
-----------

Check if the compiler supports std:atomic out of the box or if libatomic is
needed for atomic support. If it is needed libatomicis added to
``CMAKE_REQUIRED_LIBRARIES``. So after running CheckAtomic you can use
std:atomic.

Since 5.75.0.
#]=======================================================================]

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)

cmake_push_check_state()

# Sometimes linking against libatomic is required for atomic ops, if the platform doesn't
# support lock-free atomics.

# 64-bit target test code
if(CMAKE_SIZE_OF_VOID_P EQUAL 8)
    set(ATOMIC_CODE_64
        "
        std::atomic<uint64_t> x(0);
        uint64_t i = x.load(std::memory_order_relaxed);
        ")
endif()

# Test code
string(
    CONFIGURE
        [[
        #include <atomic>
        #include <cstdint>
        int main() {
            std::atomic<int> a;
            std::atomic<short> b;
            std::atomic<char> c;
            ++c;
            ++b;
            ++a;
            @ATOMIC_CODE_64@
            return a;
        }
        ]]
        ATOMIC_CODE
    @ONLY)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++11")
check_cxx_source_compiles("${ATOMIC_CODE}" CXX_ATOMIC_NO_LINK)

set(ATOMIC_FOUND ${CXX_ATOMIC_NO_LINK})

if(NOT CXX_ATOMIC_NO_LINK)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} atomic)
    check_cxx_source_compiles("${ATOMIC_CODE}" CXX_ATOMIC_WITH_LINK)
    set(ATOMIC_FOUND ${CXX_ATOMIC_WITH_LINK})
endif()

cmake_pop_check_state()

if(ATOMIC_FOUND)
    if(CXX_ATOMIC_NO_LINK)
        message(VERBOSE "Found std::atomic with no linking")
    elseif(CXX_ATOMIC_WITH_LINK)
        message(VERBOSE "Found std::atomic with linking")
    endif()
else()
    message(WARNING "std::atomic not found with compilation checks")
endif()
