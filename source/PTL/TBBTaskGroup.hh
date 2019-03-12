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
// This file wraps a TBB task_group into a TaskGroup
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Jun 21st 2018)
// ---------------------------------------------------------------

#pragma once

#include "PTL/TaskGroup.hh"

class ThreadPool;

#if defined(PTL_USE_TBB)

#    include <functional>
#    include <memory>
#    include <tbb/tbb.h>

class ThreadPool;
namespace
{
typedef tbb::task_group tbb_task_group_t;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Arg = _Tp>
class TBBTaskGroup : public TaskGroup<_Tp, _Arg>
{
public:
    typedef typename std::remove_const<typename std::remove_reference<_Arg>::type>::type
        ArgTp;

    template <typename... _Args>
    using task_type = Task<ArgTp, _Args...>;

    template <typename... _Args>
    using task_pointer = std::shared_ptr<task_type<_Args...>>;

    using func_task_type    = Task<ArgTp>;
    using func_task_pointer = std::shared_ptr<func_task_type>;

    typedef TBBTaskGroup<_Tp, _Arg>                this_type;
    typedef TaskGroup<_Tp, _Arg>                   base_type;
    typedef typename base_type::result_type        result_type;
    typedef typename base_type::data_type          data_type;
    typedef typename base_type::packaged_task_type packaged_task_type;
    typedef typename base_type::future_type        future_type;
    typedef typename base_type::promise_type       promise_type;
    typedef tbb::task_group                        tbb_task_group_t;

public:
    // Constructor
    template <typename _Func>
    TBBTaskGroup(const _Func& _join, ThreadPool* tp = nullptr)
    : base_type(_join, tp)
    , m_tbb_task_group(new tbb_task_group_t())
    {
    }

    template <typename _Func>
    TBBTaskGroup(int _freq, const _Func& _join, ThreadPool* tp = nullptr)
    : base_type(_freq, _join, tp)
    , m_tbb_task_group(new tbb_task_group_t())
    {
    }

    // Destructor
    virtual ~TBBTaskGroup() { delete m_tbb_task_group; }

    // delete copy-construct
    TBBTaskGroup(const this_type&) = delete;
    // define move-construct
    TBBTaskGroup(this_type&& rhs) = default;

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs) = default;

public:
    //------------------------------------------------------------------------//
    template <typename... _Args>
    task_pointer<_Args...>& operator+=(task_pointer<_Args...>& _task)
    {
        // store in list
        vtask_list.push_back(_task);
        // thread-safe increment of tasks in task group
        operator++();
        // return
        return _task;
    }
    //------------------------------------------------------------------------//
    func_task_pointer& operator+=(func_task_pointer& _task)
    {
        // store in list
        vtask_list.push_back(_task);
        // thread-safe increment of tasks in task group
        operator++();
        // return
        return _task;
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    task_pointer<_Args...> wrap(const _Func& func, _Args... args)
    {
        return task_pointer<_Args...>(
            new task_type<_Args...>(this, func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    func_task_pointer wrap(const _Func& func)
    {
        return func_task_pointer(new func_task_type(this, func));
        ;
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(const _Func& func, _Args... args)
    {
        auto _task = wrap(func, std::forward<_Args>(args)...);
        auto _fut  = _task->get_future();
        auto _func = [=]() {
            (*_task)();
            intmax_t _count = operator--();
            if(_count < 1)
            {
                AutoLock l(this->task_lock());
                CONDITIONBROADCAST(&(this->task_cond()));
            }
        };
        m_task_set.push_back(data_type(false, std::move(_fut), ArgTp()));
        m_tbb_task_group->run(_func);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void run(const _Func& func)
    {
        auto _func = [=]() {
            func();
            intmax_t _count = operator--();
            if(_count < 1)
            {
                AutoLock l(this->task_lock());
                CONDITIONBROADCAST(&(this->task_cond()));
            }
        };
        m_tbb_task_group->run(func);
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(const _Func& func, _Args... args)
    {
        run(func, std::forward<_Args>(args)...);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void exec(const _Func& func)
    {
        run(func);
    }

public:
    //------------------------------------------------------------------------//
    // this is not a native Tasking task group
    virtual bool is_native_task_group() const override { return false; }

    //------------------------------------------------------------------------//
    // wait on tbb::task_group, not internal thread-pool
    virtual void wait() override
    {
        VTaskGroup::wait();
        m_tbb_task_group->wait();
    }

protected:
    // Protected variables
    tbb_task_group_t* m_tbb_task_group;
    using base_type::m_join_function;
    using base_type::m_promise;
    using base_type::m_task_set;
    using base_type::vtask_list;
    using base_type::operator++;
    using base_type::operator--;
};

//--------------------------------------------------------------------------------------//
// specialization for void type
template <>
class TBBTaskGroup<void, void> : public TaskGroup<void, void>
{
public:
    typedef TBBTaskGroup<void, void>               this_type;
    typedef TaskGroup<void, void>                  base_type;
    typedef typename base_type::result_type        result_type;
    typedef typename base_type::ArgTp              ArgTp;
    typedef typename base_type::data_type          data_type;
    typedef typename base_type::packaged_task_type packaged_task_type;
    typedef typename base_type::future_type        future_type;
    typedef typename base_type::promise_type       promise_type;
    typedef tbb::task_group                        tbb_task_group_t;

public:
    // Constructor
    explicit TBBTaskGroup(ThreadPool* _tp = nullptr)
    : base_type(_tp)
    , m_tbb_task_group(new tbb_task_group_t())
    {
    }
    template <typename _Func>
    TBBTaskGroup(const _Func& _join, ThreadPool* tp = nullptr)
    : base_type(_join, tp)
    , m_tbb_task_group(new tbb_task_group_t())
    {
    }

    // Destructor
    virtual ~TBBTaskGroup() { delete m_tbb_task_group; }

    // delete copy-construct
    TBBTaskGroup(const this_type&) = delete;
    // define move-construct
    TBBTaskGroup(this_type&&) = default;

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs) = default;

public:
    //------------------------------------------------------------------------//
    template <typename... _Args>
    task_pointer<_Args...>& operator+=(task_pointer<_Args...>& _task)
    {
        // store in list
        vtask_list.push_back(_task);
        // thread-safe increment of tasks in task group
        operator++();
        // return
        return _task;
    }
    //------------------------------------------------------------------------//
    func_task_pointer& operator+=(func_task_pointer& _task)
    {
        // store in list
        vtask_list.push_back(_task);
        // thread-safe increment of tasks in task group
        operator++();
        // return
        return _task;
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    task_pointer<_Args...> wrap(const _Func& func, _Args... args)
    {
        return task_pointer<_Args...>(
            new task_type<_Args...>(this, func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    func_task_pointer wrap(const _Func& func)
    {
        return func_task_pointer(new func_task_type(this, func));
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(const _Func& func, _Args... args)
    {
        auto _task = wrap(func, std::forward<_Args>(args)...);
        auto _func = [=]() {
            (*_task)();
            intmax_t _count = --m_tot_task_count;
            if(_count < 1)
            {
                AutoLock l(this->task_lock());
                CONDITIONBROADCAST(&(this->task_cond()));
            }
        };
        m_tbb_task_group->run(_func);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void run(const _Func& func)
    {
        auto _func = [=]() {
            func();
            intmax_t _count = --m_tot_task_count;
            if(_count < 1)
            {
                AutoLock l(this->task_lock());
                CONDITIONBROADCAST(&(this->task_cond()));
            }
        };
        m_tbb_task_group->run(_func);
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(const _Func& func, _Args... args)
    {
        run(func, std::forward<_Args>(args)...);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void exec(const _Func& func)
    {
        run(func);
    }

public:
    //------------------------------------------------------------------------//
    // this is not a native Tasking task group
    virtual bool is_native_task_group() const override { return false; }

    //------------------------------------------------------------------------//
    // wait on tbb::task_group, not internal thread-pool
    virtual void wait() override
    {
        VTaskGroup::wait();
        m_tbb_task_group->wait();
    }

protected:
    // Private variables
    using base_type::m_tot_task_count;
    tbb_task_group_t* m_tbb_task_group;
};

//--------------------------------------------------------------------------------------//
#else

template <typename _Tp, typename _Arg = _Tp>
using TBBTaskGroup = TaskGroup<_Tp, _Arg>;

#endif

//--------------------------------------------------------------------------------------//

#include "PTL/TBBTaskGroup.icc"

//--------------------------------------------------------------------------------------//
