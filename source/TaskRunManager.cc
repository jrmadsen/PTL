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

#include "PTL/TaskRunManager.hh"
#include "PTL/AutoLock.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/TiMemory.hh"
#include "PTL/Utility.hh"

#include <cstdlib>
#include <cstring>
#include <iterator>

//============================================================================//

TaskRunManager*&
TaskRunManager::GetPrivateMasterRunManager(bool init, bool useTBB)
{
    static TaskRunManager* _instance =
        (init) ? new TaskRunManager(useTBB) : nullptr;
    return _instance;
}

//============================================================================//

TaskRunManager*&
TaskRunManager::GetMasterRunManager(bool useTBB)
{
    static TaskRunManager* _instance = GetPrivateMasterRunManager(true, useTBB);
    return _instance;
}

//============================================================================//

TaskRunManager*
TaskRunManager::GetInstance(bool useTBB)
{
    return GetMasterRunManager(useTBB);
}

//============================================================================//

TaskRunManager::TaskRunManager(bool useTBB)
: isInitialized(false)
, verbose(0)
, nworkers(std::thread::hardware_concurrency())
, taskQueue(nullptr)
, threadPool(nullptr)
, taskManager(nullptr)
, workTaskGroup(nullptr)
, workTaskGroupTBB(nullptr)
{
    if(!GetPrivateMasterRunManager(false))
        GetPrivateMasterRunManager(false) = this;

#ifdef PTL_USE_TBB
    int _useTBB = GetEnv<int>("FORCE_TBB", (int) useTBB);
    if(_useTBB > 0)
        useTBB = true;
#endif

    // handle TBB
    ThreadPool::set_use_tbb(useTBB);

    nworkers = GetEnv<uint64_t>("PTL_NUM_THREADS", nworkers);

    /*
#if defined(PTL_USE_TIMEMORY)
    tim::manager::instance()->set_get_num_threads_func([=] () { return nworkers;
}); #endif
    */
}

//============================================================================//

TaskRunManager::~TaskRunManager() { Terminate(); }

//============================================================================//

void
TaskRunManager::Initialize(uint64_t n)
{
    nworkers = n;

    // create threadpool if needed + task manager
    if(!threadPool)
    {
        if(verbose > 0)
            std::cout << "TaskRunManager :: Creating thread pool..."
                      << std::endl;
        threadPool = new ThreadPool(nworkers, taskQueue);
        if(verbose > 0)
            std::cout << "TaskRunManager :: Creating task manager..."
                      << std::endl;
        taskManager = new TaskManager(threadPool);
    }
    // or resize
    else if(nworkers != threadPool->size())
    {
        if(verbose > 0)
            std::cout << "TaskRunManager :: Resizing thread pool from "
                      << threadPool->size() << " to " << nworkers
                      << " threads ..." << std::endl;
        threadPool->resize(nworkers);
    }

    // create the joiners
    if(ThreadPool::using_tbb())
    {
        if(verbose > 0)
            std::cout << "TaskRunManager :: Using TBB..." << std::endl;
        if(!workTaskGroupTBB)
        {
            workTaskGroupTBB = new RunTaskGroupTBB();
        }
    }
    else
    {
        if(verbose > 0)
            std::cout << "TaskRunManager :: Using ThreadPool..." << std::endl;
        if(!workTaskGroup)
        {
            workTaskGroup = new RunTaskGroup();
        }
    }

    isInitialized = true;
    if(verbose > 0)
        std::cout << "TaskRunManager :: initialized..." << std::endl;

    /*
#if defined(PTL_USE_TIMEMORY)
    tim::manager::instance()->set_get_num_threads_func([=] () { return nworkers;
}); #endif
    */
}

//============================================================================//

void
TaskRunManager::Terminate()
{
    isInitialized = false;

    if(workTaskGroupTBB)
        workTaskGroupTBB->join();
    if(workTaskGroup)
        workTaskGroup->join();
    delete workTaskGroupTBB;
    delete workTaskGroup;
    delete taskManager;
    delete threadPool;
    workTaskGroupTBB = nullptr;
    workTaskGroup    = nullptr;
    taskManager      = nullptr;
    threadPool       = nullptr;
}

//============================================================================//

void
TaskRunManager::Wait()
{
    // Now join threads.
    if(workTaskGroupTBB)
        workTaskGroupTBB->join();

    if(workTaskGroup)
        workTaskGroup->join();
}

//============================================================================//

void
TaskRunManager::TiMemoryReport(std::string fname, bool echo_stdout) const
{
#ifdef PTL_USE_TIMEMORY
    if(fname.length() > 0 || echo_stdout)
    {
        std::cout << "\nOutputting TiMemory results...\n" << std::endl;
        tim::manager* timemory_manager = tim::manager::instance();

        if(echo_stdout)
            timemory_manager->write_report(std::cout, true);

        if(fname.length() > 0)
        {
            fname += "_x" + std::to_string(threadPool->size());
            timemory_manager->write_report(fname + ".txt");
            timemory_manager->write_serialization(fname + ".json");
        }
    }
#else
    ConsumeParameters(fname, echo_stdout);
#endif
}

//============================================================================//
