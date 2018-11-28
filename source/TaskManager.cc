//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
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
// ---------------------------------------------------------------
//  Tasking class implementation
//
// Class Description:
//
// This file creates a class for handling the wrapping of functions
// into task objects and submitting to thread pool
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"

//============================================================================//

TaskManager*&
TaskManager::fgInstance()
{
    ThreadLocalStatic TaskManager* _instance = nullptr;
    return _instance;
}

//============================================================================//

TaskManager*
TaskManager::GetInstance()
{
    if(!fgInstance())
    {
        auto nthreads = std::thread::hardware_concurrency();
        std::cout << "Allocating mad::TaskManager with " << nthreads
                  << " thread(s)..." << std::endl;
        new TaskManager(TaskRunManager::GetMasterRunManager()->GetThreadPool());
    }
    return fgInstance();
}

//============================================================================//

TaskManager*
TaskManager::GetInstanceIfExists()
{
    return fgInstance();
}

//============================================================================//

TaskManager::TaskManager(ThreadPool* _pool)
: m_pool(_pool)
{
    if(!fgInstance())
        fgInstance() = this;
}

//============================================================================//

TaskManager::~TaskManager()
{
    finalize();
    if(fgInstance() == this)
        fgInstance() = nullptr;
}

//============================================================================//

TaskManager::TaskManager(const TaskManager& rhs)
: m_pool(rhs.m_pool)
{
}

//============================================================================//

TaskManager&
TaskManager::operator=(const TaskManager& rhs)
{
    if(this == &rhs)
        return *this;

    m_pool = rhs.m_pool;

    return *this;
}

//============================================================================//
