//
// MIT License
// Copyright (c) 2020 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Tasking native types
//

#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// Disable warning C4786 on WIN32 architectures:
// identifier was truncated to '255' characters
// in the debug information
//
#    pragma warning(disable : 4786)
//
// Define DLL export macro for WIN32 systems for
// importing/exporting external symbols to DLLs
//
#    if defined LIB_BUILD_DLL
#        define DLLEXPORT __declspec(dllexport)
#        define DLLIMPORT __declspec(dllimport)
#    else
#        define DLLEXPORT
#        define DLLIMPORT
#    endif
//
// Unique identifier for global module
//
#    if defined PTL_ALLOC_EXPORT
#        define PTL_DLL DLLEXPORT
#    else
#        define PTL_DLL DLLIMPORT
#    endif
#else
#    define DLLEXPORT
#    define DLLIMPORT
#    define PTL_DLL
#endif

#if !defined(PTL_DEFAULT_OBJECT)
#    define PTL_DEFAULT_OBJECT(NAME)                                                     \
        NAME()            = default;                                                     \
        ~NAME()           = default;                                                     \
        NAME(const NAME&) = default;                                                     \
        NAME(NAME&&)      = default;                                                     \
        NAME& operator=(const NAME&) = default;                                          \
        NAME& operator=(NAME&&) = default;
#endif

#include <atomic>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

namespace PTL
{
//--------------------------------------------------------------------------------------//
//
namespace api
{
struct native
{};
struct tbb
{};
}  // namespace api

//======================================================================================//

struct ScopeDestructor
{
    template <typename FuncT>
    ScopeDestructor(FuncT&& _func)
    : m_functor(std::forward<FuncT>(_func))
    {}

    // delete copy operations
    ScopeDestructor(const ScopeDestructor&) = delete;
    ScopeDestructor& operator=(const ScopeDestructor&) = delete;

    // allow move operations
    ScopeDestructor(ScopeDestructor&& rhs) noexcept
    : m_functor(std::move(rhs.m_functor))
    {
        rhs.m_functor = []() {};
    }

    ScopeDestructor& operator=(ScopeDestructor&& rhs) noexcept
    {
        if(this != &rhs)
        {
            m_functor     = std::move(rhs.m_functor);
            rhs.m_functor = []() {};
        }
        return *this;
    }

    ~ScopeDestructor() { m_functor(); }

private:
    std::function<void()> m_functor = []() {};
};

//======================================================================================//

}  // namespace PTL

// Forward declation of void type argument for usage in direct object
// persistency to define fake default constructors
//
class __void__;
