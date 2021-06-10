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
//
// ---------------------------------------------------------------
// Tasking class header file
//
/// Class Description:
///
/// These are just aliases to STL mechanisms
/// (Jonathan R. Madsen - June 7th 2021)
///

#pragma once

#include "PTL/Threading.hh"

#include <mutex>

namespace PTL
{
// -------------------------------------------------------------------------- //
//
//      Use the non-template types below:
//          - AutoLock with Mutex
//          - RecursiveAutoLock with RecursiveMutex
//
// -------------------------------------------------------------------------- //

using AutoLock          = std::unique_lock<Mutex>;
using RecursiveAutoLock = std::unique_lock<RecursiveMutex>;

// provide abbriviated type if another mutex type is desired to be used
// aside from above
template <typename Tp>
using TAutoLock = std::unique_lock<Tp>;

}  // namespace PTL
