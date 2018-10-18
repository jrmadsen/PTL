//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
// Thread Local Storage typedefs

#ifndef ThreadLocalStatic_hh_
#define ThreadLocalStatic_hh_

#if ( defined(__MACH__) && defined(__clang__) && defined(__x86_64__) ) || \
    ( defined(__linux__) && defined(__clang__) )
#   define ThreadLocalStatic static thread_local
#   define ThreadLocal thread_local

#elif ( (defined(__linux__) || defined(__MACH__)) && \
    !defined(__INTEL_COMPILER) && defined(__GNUC__) && \
    (__GNUC__>=4 && __GNUC_MINOR__<9))
#   define ThreadLocalStatic static __thread
#   define ThreadLocal thread_local

#elif ( (defined(__linux__) || defined(__MACH__)) && \
    !defined(__INTEL_COMPILER) && defined(__GNUC__) && \
    (__GNUC__>=4 && __GNUC_MINOR__>=9) || __GNUC__>=5 )
#   define ThreadLocalStatic static thread_local
#   define ThreadLocal thread_local

#elif ( (defined(__linux__) || defined(__MACH__)) && \
    defined(__INTEL_COMPILER) )
#   if __INTEL_COMPILER>=1500
#       define ThreadLocalStatic static thread_local
#       define ThreadLocal thread_local
#   else
#       define ThreadLocalStatic static __thread
#       define ThreadLocal __thread
#   endif

#elif defined(_AIX)
#   define ThreadLocalStatic static thread_local
#   define ThreadLocal thread_local

#elif defined(WIN32)
#   define ThreadLocalStatic static thread_local
#   define ThreadLocal thread_local

#else
// just assume at this point
#   define ThreadLocalStatic static __thread
#   define ThreadLocal __thread

#endif

#endif
