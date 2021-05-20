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
// Class Description:
//
// This file defines the task types for TaskManager and ThreadPool
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#pragma once

#include "Globals.hh"
#include "VTask.hh"

#include <cstdint>
#include <functional>
#include <stdexcept>

namespace PTL
{
class ThreadPool;

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename RetT, typename... Args>
class PackagedTask : public VTask
{
public:
    typedef PackagedTask<RetT, Args...>       this_type;
    typedef std::promise<RetT>                promise_type;
    typedef std::future<RetT>                 future_type;
    typedef std::packaged_task<RetT(Args...)> packaged_task_type;
    typedef RetT                              result_type;
    typedef std::tuple<Args...>               tuple_type;

public:
    // pass a free function pointer
    template <typename FuncT>
    PackagedTask(FuncT func, Args... args)
    : VTask{ true, 0 }
    , m_ptask{ std::move(func) }
    , m_args{ args... }
    {}

    template <typename FuncT>
    PackagedTask(bool _is_native, intmax_t _depth, FuncT func, Args... args)
    : VTask{ _is_native, _depth }
    , m_ptask{ std::move(func) }
    , m_args{ args... }
    {}

    virtual ~PackagedTask() {}

public:
    // execution operator
    virtual void operator()() final { mpl::apply(std::move(m_ptask), std::move(m_args)); }
    inline future_type get_future() { return m_ptask.get_future(); }
    inline RetT        get() { return get_future().get(); }

private:
    packaged_task_type m_ptask;
    tuple_type         m_args;
};

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename RetT, typename... Args>
class Task : public VTask
{
public:
    typedef Task<RetT, Args...>               this_type;
    typedef std::promise<RetT>                promise_type;
    typedef std::future<RetT>                 future_type;
    typedef std::packaged_task<RetT(Args...)> packaged_task_type;
    typedef RetT                              result_type;
    typedef std::tuple<Args...>               tuple_type;

public:
    template <typename FuncT>
    Task(FuncT func, Args... args)
    : VTask{}
    , m_ptask{ std::move(func) }
    , m_args{ args... }
    {}

    template <typename FuncT>
    Task(bool _is_native, intmax_t _depth, FuncT func, Args... args)
    : VTask{ _is_native, _depth }
    , m_ptask{ std::move(func) }
    , m_args{ args... }
    {}

    virtual ~Task() {}

public:
    // execution operator
    virtual void operator()() final { mpl::apply(std::move(m_ptask), std::move(m_args)); }
    inline future_type get_future() { return m_ptask.get_future(); }
    inline RetT        get() { return get_future().get(); }

private:
    packaged_task_type m_ptask{};
    tuple_type         m_args{};
};

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename RetT>
class Task<RetT, void> : public VTask
{
public:
    typedef Task<RetT>                 this_type;
    typedef std::promise<RetT>         promise_type;
    typedef std::future<RetT>          future_type;
    typedef std::packaged_task<RetT()> packaged_task_type;
    typedef RetT                       result_type;

public:
    template <typename FuncT>
    Task(FuncT func)
    : VTask()
    , m_ptask{ std::move(func) }
    {}

    template <typename FuncT>
    Task(bool _is_native, intmax_t _depth, FuncT func)
    : VTask{ _is_native, _depth }
    , m_ptask{ std::move(func) }
    {}

    virtual ~Task() {}

public:
    // execution operator
    virtual void       operator()() final { m_ptask(); }
    inline future_type get_future() { return m_ptask.get_future(); }
    inline RetT        get() { return get_future().get(); }

private:
    packaged_task_type m_ptask{};
};

//======================================================================================//

/// \brief The task class is supplied to thread_pool.
template <>
class Task<void, void> : public VTask
{
public:
    typedef void                       RetT;
    typedef Task<void, void>           this_type;
    typedef std::promise<RetT>         promise_type;
    typedef std::future<RetT>          future_type;
    typedef std::packaged_task<RetT()> packaged_task_type;
    typedef RetT                       result_type;

public:
    template <typename FuncT>
    explicit Task(FuncT func)
    : VTask()
    , m_ptask{ std::move(func) }
    {}

    template <typename FuncT>
    Task(bool _is_native, intmax_t _depth, FuncT func)
    : VTask{ _is_native, _depth }
    , m_ptask{ std::move(func) }
    {}

    virtual ~Task() {}

public:
    // execution operator
    virtual void       operator()() final { m_ptask(); }
    inline future_type get_future() { return m_ptask.get_future(); }
    inline RetT        get() { return get_future().get(); }

private:
    packaged_task_type m_ptask{};
};

//======================================================================================//

}  // namespace PTL
