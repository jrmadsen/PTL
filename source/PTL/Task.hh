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
//
// ---------------------------------------------------------------
// Tasking class header file
//
// Class Description:
//
// This file defines the task types for TaskManager and ThreadPool
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#pragma once

#include "TaskAllocator.hh"
#include "VTask.hh"

#include <cstdint>
#include <functional>
#include <stdexcept>

#define _forward_args_t(_Args, _args) std::forward<_Args>(_args)...

class VTaskGroup;
class ThreadPool;

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename _Ret, typename... _Args>
class PackagedTask : public VTask
//, public TaskAllocator<PackagedTask<_Ret, _Args...>>
{
public:
    typedef PackagedTask<_Ret, _Args...>  this_type;
    typedef std::function<_Ret(_Args...)> function_type;
    typedef std::promise<_Ret>            promise_type;
    typedef std::future<_Ret>             future_type;
    typedef std::packaged_task<_Ret()>    packaged_task_type;
    typedef _Ret                          result_type;

public:
    // pass a free function pointer
    PackagedTask(function_type&& func, _Args&&... args)
    : VTask()
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    PackagedTask(VTaskGroup* group, function_type&& func, _Args&&... args)
    : VTask(group)
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    PackagedTask(ThreadPool* pool, function_type&& func, _Args&&... args)
    : VTask(pool)
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    virtual ~PackagedTask() {}

public:
    // execution operator
    virtual void operator()() override { m_ptask(); }
    future_type  get_future() { return m_ptask.get_future(); }
    virtual bool is_native_task() const override { return true; }

private:
    packaged_task_type m_ptask;
};

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename _Ret, typename... _Args>
class Task : public VTask
//, public TaskAllocator<Task<_Ret, _Args...>>
{
public:
    typedef Task<_Ret, _Args...>          this_type;
    typedef std::function<_Ret(_Args...)> function_type;
    typedef std::promise<_Ret>            promise_type;
    typedef std::future<_Ret>             future_type;
    typedef std::packaged_task<_Ret()>    packaged_task_type;
    typedef _Ret                          result_type;

public:
    Task(function_type&& func, _Args&&... args)
    : VTask()
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    Task(VTaskGroup* group, function_type&& func, _Args&&... args)
    : VTask(group)
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    Task(ThreadPool* pool, function_type&& func, _Args&&... args)
    : VTask(pool)
    , m_ptask([&]() { return func(std::forward<_Args>(args)...); })
    {
    }

    virtual ~Task() {}

public:
    // execution operator
    virtual void operator()() override
    {
        m_ptask();
        // decrements the task-group counter on active tasks
        // when the counter is < 2, if the thread owning the task group is
        // sleeping at the TaskGroup::wait(), it signals the thread to wake
        // up and check if all tasks are finished, proceeding if this
        // check returns as true
        this_type::operator--();
    }

    virtual bool is_native_task() const override { return true; }
    future_type  get_future() { return m_ptask.get_future(); }

private:
    packaged_task_type m_ptask;
};

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <>
class Task<void, void> : public VTask
//, public TaskAllocator<Task<void, void>>
{
public:
    typedef void                       _Ret;
    typedef Task<void, void>           this_type;
    typedef std::function<_Ret()>      function_type;
    typedef std::promise<_Ret>         promise_type;
    typedef std::future<_Ret>          future_type;
    typedef std::packaged_task<_Ret()> packaged_task_type;
    typedef _Ret                       result_type;

public:
    explicit Task(function_type&& func)
    : VTask()
    , m_ptask(func)
    {
    }

    Task(VTaskGroup* group, function_type&& func)
    : VTask(group)
    , m_ptask(func)
    {
    }

    Task(ThreadPool* pool, function_type&& func)
    : VTask(pool)
    , m_ptask(func)
    {
    }

    virtual ~Task() {}

public:
    // execution operator
    virtual void operator()() override
    {
        m_ptask();
        // decrements the task-group counter on active tasks
        // when the counter is < 2, if the thread owning the task group is
        // sleeping at the TaskGroup::wait(), it signals the thread to wake
        // up and check if all tasks are finished, proceeding if this
        // check returns as true
        this_type::operator--();
    }

    virtual bool is_native_task() const override { return true; }
    future_type  get_future() { return m_ptask.get_future(); }

private:
    packaged_task_type m_ptask;
};

//======================================================================================//

// don't pollute
#undef _forward_args_t
