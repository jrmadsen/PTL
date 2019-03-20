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
#include <functional>
#include <memory>

class ThreadPool;

//--------------------------------------------------------------------------------------//
#if defined(PTL_USE_TBB)
//--------------------------------------------------------------------------------------//

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
    //------------------------------------------------------------------------//
    typedef typename std::remove_const<typename std::remove_reference<_Arg>::type>::type
        ArgTp;
    //------------------------------------------------------------------------//
    template <typename... _Args>
    using task_type = Task<ArgTp, _Args...>;
    //------------------------------------------------------------------------//
    template <typename... _Args>
    using task_pointer = std::shared_ptr<task_type<_Args...>>;
    //------------------------------------------------------------------------//
    template <bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;
    //------------------------------------------------------------------------//

    typedef TBBTaskGroup<_Tp, _Arg>                                    this_type;
    typedef TaskGroup<_Tp, _Arg>                                       base_type;
    typedef typename base_type::result_type                            result_type;
    typedef typename base_type::packaged_task_type                     packaged_task_type;
    typedef typename base_type::future_type                            future_type;
    typedef typename base_type::promise_type                           promise_type;
    typedef typename base_type::template JoinFunction<_Tp, _Arg>::Type join_type;
    typedef tbb::task_group                                            tbb_task_group_t;

public:
    // Constructor
    template <typename _Func>
    TBBTaskGroup(_Func&& _join, ThreadPool* _tp = nullptr)
    : base_type(std::forward<_Func>(_join), _tp)
    , m_tbb_task_group(new tbb_task_group_t)
    {
    }
    template <typename _Up = _Tp, enable_if_t<std::is_same<_Up, void>::value, int> = 0>
    TBBTaskGroup(ThreadPool* _tp = nullptr)
    : base_type(_tp)
    , m_tbb_task_group(new tbb_task_group_t)
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
        // add the future
        m_task_set.push_back(std::move(_task->get_future()));
        // return
        return _task;
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    task_pointer<_Args...> wrap(_Func&& func, _Args&&... args)
    {
        auto _task = task_pointer<_Args...>(
            new task_type<_Args...>(this, std::forward<_Func>(func),
                                    std::forward<_Args>(args)...));
        return operator+=(_task);
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(_Func&& func, _Args&&... args)
    {
        auto _task = wrap(std::forward<_Func>(func), std::forward<_Args>(args)...);
        auto _lamb = [=]() { (*_task)(); };
        m_tbb_task_group->run(_lamb);
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(_Func&& func, _Args&&... args)
    {
        run(std::forward<_Func>(func), std::forward<_Args>(args)...);
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args, typename _Up = _Tp,
              enable_if_t<std::is_same<_Up, void>::value, int> = 0>
    void parallel_for(uintmax_t nitr, uintmax_t chunks, _Func&& func, _Args&&... args)
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nitr),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for(size_t i = range.begin(); i != range.end(); ++i)
                                  func(std::forward<_Args>(args)...);
                          });
    }

public:
    //------------------------------------------------------------------------//
    // this is not a native Tasking task group
    virtual bool is_native_task_group() const override { return false; }

    //------------------------------------------------------------------------//
    // wait on tbb::task_group, not internal thread-pool
    virtual void wait() override
    {
        base_type::wait();
        m_tbb_task_group->wait();
    }

public:
    using base_type::begin;
    using base_type::cbegin;
    using base_type::cend;
    using base_type::clear;
    using base_type::end;
    using base_type::get_tasks;

    //------------------------------------------------------------------------//
    // wait to finish
    template <typename _Up = _Tp, enable_if_t<!std::is_same<_Up, void>::value, int> = 0>
    inline _Up join(_Up accum = {})
    {
        this->wait();
        for(auto& itr : m_task_set)
            accum = m_join(std::ref(accum), std::forward<ArgTp>(itr.get()));
        this->clear();
        return accum;
    }
    //------------------------------------------------------------------------//
    // wait to finish
    template <typename _Up = _Tp, enable_if_t<std::is_same<_Up, void>::value, int> = 0>
    inline void join()
    {
        this->wait();
        for(auto& itr : m_task_set)
            itr.get();
        m_join();
        this->clear();
    }

protected:
    // Protected variables
    tbb_task_group_t* m_tbb_task_group;
    using base_type:: operator++;
    using base_type:: operator--;
    using base_type::m_join;
    using base_type::m_task_set;
    using base_type::vtask_list;
};

//--------------------------------------------------------------------------------------//
#else
//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Arg = _Tp>
using TBBTaskGroup = TaskGroup<_Tp, _Arg>;

//--------------------------------------------------------------------------------------//
#endif
//--------------------------------------------------------------------------------------//
