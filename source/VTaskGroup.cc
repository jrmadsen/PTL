//
// MIT License
// Copyright (c) 2019 Jonathan R. Madsen
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
// This file creates an abstract base class for the grouping the thread-pool
// tasking system into independently joinable units
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#include "PTL/VTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/VTask.hh"

//======================================================================================//

std::atomic_uintmax_t&
vtask_group_counter()
{
    static std::atomic_uintmax_t _instance(0);
    return _instance;
}

//======================================================================================//

VTaskGroup::VTaskGroup(ThreadPool* tp)
: m_tot_task_count(0)
, m_id(vtask_group_counter()++)
, m_pool(tp)
, m_task_lock()
, m_main_tid(std::this_thread::get_id())
{
    if(!m_pool && TaskRunManager::GetMasterRunManager())
        m_pool = TaskRunManager::GetMasterRunManager()->GetThreadPool();

#ifdef DEBUG
    if(!m_pool && GetEnv<int>("PTL_VERBOSE", 0) > 0)
    {
        std::cerr << __FUNCTION__ << "@" << __LINE__ << " :: Warning! "
                  << "nullptr to thread pool!" << std::endl;
    }
#endif
}

//======================================================================================//

VTaskGroup::~VTaskGroup() {}

//======================================================================================//

void
VTaskGroup::execute_this_threads_tasks()
{
    // for internal threads
    ThreadData* data = ThreadData::GetInstance();

    ThreadPool* _tpool = (m_pool) ? m_pool : ((data) ? data->thread_pool : nullptr);

    VUserTaskQueue* _taskq =
        (m_pool) ? m_pool->get_queue() : ((data) ? data->current_queue : nullptr);

    // for external threads
    bool ext_is_master   = (data) ? data->is_master : false;
    bool ext_within_task = (data) ? data->within_task : true;

    // for external threads
    if(!data)
    {
        _tpool = TaskRunManager::GetMasterRunManager()->GetThreadPool();
        _taskq = _tpool->get_queue();
    }

    // something is wrong, didn't create thread-pool?
    if(!_tpool || !_taskq)
    {
#ifdef DEBUG
        if(GetEnv<int>("PTL_VERBOSE", 0) > 0)
            std::cerr << __FUNCTION__ << "@" << __LINE__ << " :: Warning! "
                      << "nullptr to thread data!" << std::endl;
#endif
        return;
    }

    // only want to process if within a task
    if((!ext_is_master || _tpool->size() < 2) && ext_within_task)
    {
        if(!_taskq)
            return;
        int        bin  = static_cast<int>(_taskq->GetThreadBin());
        const auto nitr = (_tpool) ? _tpool->size() : Thread::hardware_concurrency();
        while(this->pending() > 0)
        {
            _taskq->GetTask(bin, static_cast<int>(nitr));
        }
    }
}

//======================================================================================//

void
VTaskGroup::wait()
{
    // if no pool was initially present at creation
    if(!m_pool)
    {
        // check for master MT run-manager
        if(TaskRunManager::GetMasterRunManager())
            m_pool = TaskRunManager::GetMasterRunManager()->GetThreadPool();

        // if MTRunManager does not exist or no thread pool created
        if(!m_pool)
        {
#ifdef DEBUG
            if(GetEnv<int>("PTL_VERBOSE", 0) > 0)
            {
                std::cerr << __FUNCTION__ << "@" << __LINE__ << " :: Warning! "
                          << "nullptr to thread pool!" << std::endl;
            }
#endif
            return;
        }
    }

    // return if thread pool isn't built
    if(!m_pool->is_alive() || !is_native_task_group())
        return;

    // execute_this_threads_tasks();
    // return;

    auto is_active_state = [&]() {
        return static_cast<int>(m_pool->state()) != static_cast<int>(state::STOPPED);
    };

    intmax_t _pool_size = m_pool->size();
    AutoLock _lock(m_task_lock, std::defer_lock);

    while(is_active_state())
    {
        execute_this_threads_tasks();

        intmax_t _pending = 0;
        // while loop protects against spurious wake-ups
        while((_pending = pending()) > 0 && is_active_state())
        {
            // lock before sleeping on condition
            if(!_lock.owns_lock())
                _lock.lock();
            // Wait until signaled that a task has been competed
            // Unlock mutex while wait, then lock it back when signaled
            if((_pending = pending()) > _pool_size)  // for safety
                m_task_cond.wait(_lock);
            else
                m_task_cond.wait_for(_lock, std::chrono::milliseconds(10));
            if(_lock.owns_lock())
                _lock.unlock();
        }

        // if pending is not greater than zero, we are joined
        if((_pending = pending()) <= 0)
            break;
    }

    if(_lock.owns_lock())
        _lock.unlock();

    intmax_t ntask = this->task_count().load();
    if(ntask > 0)
    {
        std::stringstream ss;
        ss << "\nWarning! Join operation issue! " << ntask << " tasks still "
           << "are running!" << std::endl;
        std::cout << ss.str();
        this->wait();
        // throw std::runtime_error(ss.str().c_str());
    }
}

//======================================================================================//
