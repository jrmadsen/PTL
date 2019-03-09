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
// Tasking class header file
//
// Class Description:
//
// This file creates an abstract base class for the grouping the thread-pool
// tasking system into independently joinable units
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#pragma once

#include "PTL/AutoLock.hh"
#include "PTL/Threading.hh"
#include "PTL/VTask.hh"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

class ThreadPool;

#define _MOVE_MEMBER(_member) _member = std::move(rhs._member)

class VTaskGroup
{
public:
    template <typename _Tp>
    using container_type = std::list<_Tp>;
    template <typename _Tp>
    using list_type = std::list<_Tp>;
    template <typename _Key, typename _Mapped>
    using map_type = std::unordered_map<_Key, _Mapped>;

    typedef VTaskGroup                     this_type;
    typedef std::thread::id                tid_type;
    typedef VTask                          task_type;
    typedef uintmax_t                      size_type;
    typedef Mutex                          lock_t;
    typedef std::atomic_intmax_t           atomic_int;
    typedef std::atomic_uintmax_t          atomic_uint;
    typedef map_type<tid_type, atomic_int> task_count_map;
    typedef Condition                      condition_t;
    typedef std::shared_ptr<task_type>     task_pointer;
    typedef container_type<task_pointer>   vtask_list_type;

public:
    // Constructor and Destructors
    explicit VTaskGroup(ThreadPool* tp = nullptr);
    // Virtual destructors are required by abstract classes
    // so add it by default, just in case
    virtual ~VTaskGroup();

    VTaskGroup(const this_type&) = delete;
    VTaskGroup(this_type&& rhs)
    {
        *this = std::move(rhs);
        /*
        m_clear_count.store(m_clear_count.load());
        m_clear_freq.store(m_clear_freq.load());
        m_tot_task_count.store(m_tot_task_count.load());
        _MOVE_MEMBER(m_tid_task_count);
        _MOVE_MEMBER(m_id);
        _MOVE_MEMBER(m_pool);
        _MOVE_MEMBER(m_main_tid);
        _MOVE_MEMBER(vtask_list);*/
    }

    this_type& operator=(const this_type&) = delete;
    this_type& operator                    =(this_type&& rhs)
    {
        if(this == &rhs)
            return *this;

        m_clear_count.store(m_clear_count.load());
        m_clear_freq.store(m_clear_freq.load());
        m_tot_task_count.store(m_tot_task_count.load());
        _MOVE_MEMBER(m_tid_task_count);
        _MOVE_MEMBER(m_id);
        _MOVE_MEMBER(m_pool);
        _MOVE_MEMBER(m_main_tid);
        _MOVE_MEMBER(vtask_list);
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    // wait to finish
    virtual void wait();

    //------------------------------------------------------------------------//
    // increment (prefix)
    intmax_t increase(tid_type thread_id)
    {
        intmax_t _totc = ++m_tot_task_count;
        ConsumeParameters(thread_id);
        return _totc;
    }
    //------------------------------------------------------------------------//
    // increment (prefix)
    intmax_t reduce(tid_type thread_id)
    {
        intmax_t _totc = --m_tot_task_count;
        ConsumeParameters(thread_id);
        return _totc;
    }
    //------------------------------------------------------------------------//
    // increment (prefix)
    intmax_t count(tid_type thread_id)
    {
        intmax_t _totc = m_tot_task_count.load();
        ConsumeParameters(thread_id);
        return _totc;
    }

    // get the locks/conditions
    lock_t&            task_lock() { return m_task_lock; }
    condition_t&       task_cond() { return m_task_cond; }
    const lock_t&      task_lock() const { return m_task_lock; }
    const condition_t& task_cond() const { return m_task_cond; }

    // identifier
    const uintmax_t& id() const { return m_id; }

    // thread pool
    void         set_pool(ThreadPool* tp) { m_pool = tp; }
    ThreadPool*& pool() { return m_pool; }
    ThreadPool*  pool() const { return m_pool; }

    virtual task_pointer store(task_pointer ptr);
    virtual void         clear() { vtask_list.clear(); }
    virtual bool         is_native_task_group() const { return true; }
    virtual bool         is_master() const { return this_tid() == m_main_tid; }

    //------------------------------------------------------------------------//
    // set the number of join calls before clear (zero == never)
    virtual void set_clear_frequency(uint32_t val) { m_clear_freq.store(val); }

    //------------------------------------------------------------------------//
    // get the number of join calls before clear
    virtual uintmax_t get_clear_frequency() const { return m_clear_freq.load(); }

    //------------------------------------------------------------------------//
    // check if any tasks are still pending
    virtual intmax_t pending() { return m_tot_task_count.load(); }

protected:
    //------------------------------------------------------------------------//
    // get the thread id
    static tid_type this_tid() { return std::this_thread::get_id(); }

    //------------------------------------------------------------------------//
    // get the task count
    atomic_int&       task_count() { return m_tot_task_count; }
    const atomic_int& task_count() const { return m_tot_task_count; }

    //------------------------------------------------------------------------//
    // process tasks in personal bin
    void execute_this_threads_tasks();

protected:
    // Private variables
    atomic_uint            m_clear_count;
    atomic_uint            m_clear_freq;
    atomic_int             m_tot_task_count;
    mutable task_count_map m_tid_task_count;
    uintmax_t              m_id;
    ThreadPool*            m_pool;
    condition_t            m_task_cond;
    lock_t                 m_task_lock;
    tid_type               m_main_tid;
    vtask_list_type        vtask_list;

protected:
    enum class state : int
    {
        STARTED = 0,
        STOPPED = 1,
        NONINIT = 2
    };
};

//--------------------------------------------------------------------------------------//

inline VTaskGroup::task_pointer
VTaskGroup::store(task_pointer ptr)
{
    // store in list
    vtask_list.push_back(ptr);
    // thread-safe increment of tasks in task group that are to run in pool
    ptr->operator++();
    // return reference
    return vtask_list.back();
}

//--------------------------------------------------------------------------------------//

// don't pollute
#undef _MOVE_MEMBER
