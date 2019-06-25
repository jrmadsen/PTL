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
// This file creates the a class for handling a group of tasks that
// can be independently joined
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#pragma once

#include "PTL/Task.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/VTaskGroup.hh"

#include <cstdint>
#include <deque>
#include <future>
#include <list>
#include <vector>

#ifdef PTL_USE_TBB
#    include <tbb/tbb.h>
#endif

class ThreadPool;

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Arg = _Tp>
class TaskGroup
: public VTaskGroup
, public TaskAllocator<TaskGroup<_Tp, _Arg>>
{
protected:
    //----------------------------------------------------------------------------------//
    template <typename _JTp, typename _JArg>
    struct JoinFunction
    {
    public:
        typedef std::function<_JTp(_JTp&, _JArg&&)> Type;

    public:
        template <typename _Func>
        JoinFunction(_Func&& func)
        : m_func(std::forward<_Func>(func))
        {
        }

        template <typename... _Args>
        _JTp& operator()(_Args&&... args)
        {
            return std::move(m_func(std::forward<_Args>(args)...));
        }

    private:
        Type m_func;
    };
    //----------------------------------------------------------------------------------//
    template <typename _JArg>
    struct JoinFunction<void, _JArg>
    {
    public:
        typedef std::function<void()> Type;

    public:
        template <typename _Func>
        JoinFunction(_Func&& func)
        : m_func(std::forward<_Func>(func))
        {
        }

        void operator()() { m_func(); }

    private:
        Type m_func;
    };
    //----------------------------------------------------------------------------------//

public:
    //------------------------------------------------------------------------//
    template <typename _Type>
    using remove_reference_t = typename std::remove_reference<_Type>::type;
    //------------------------------------------------------------------------//
    template <typename _Type>
    using remove_const_t = typename std::remove_const<_Type>::type;
    //------------------------------------------------------------------------//
    template <bool B, class T = void>
    using enable_if_t = typename std::enable_if<B, T>::type;
    //------------------------------------------------------------------------//
    typedef remove_const_t<remove_reference_t<_Arg>>     ArgTp;
    typedef _Tp                                          result_type;
    typedef TaskGroup<_Tp, _Arg>                         this_type;
    typedef std::promise<ArgTp>                          promise_type;
    typedef std::future<ArgTp>                           future_type;
    typedef std::packaged_task<ArgTp()>                  packaged_task_type;
    typedef list_type<future_type>                       task_list_t;
    typedef typename JoinFunction<_Tp, _Arg>::Type       join_type;
    typedef typename task_list_t::iterator               iterator;
    typedef typename task_list_t::reverse_iterator       reverse_iterator;
    typedef typename task_list_t::const_iterator         const_iterator;
    typedef typename task_list_t::const_reverse_iterator const_reverse_iterator;
    //------------------------------------------------------------------------//
    template <typename... _Args>
    using task_type = Task<ArgTp, _Args...>;
    //------------------------------------------------------------------------//

public:
    // Constructor
    template <typename _Func>
    TaskGroup(_Func&& _join, ThreadPool* _tp = nullptr)
    : VTaskGroup(_tp)
    , m_join(std::forward<_Func>(_join))
    {
    }
    template <typename _Up = _Tp, enable_if_t<std::is_same<_Up, void>::value, int> = 0>
    explicit TaskGroup(ThreadPool* _tp = nullptr)
    : VTaskGroup(_tp)
    , m_join([]() {})
    {
    }
    // Destructor
    virtual ~TaskGroup() { this->clear(); }

    // delete copy-construct
    TaskGroup(const this_type&) = delete;
    // define move-construct
    TaskGroup(this_type&& rhs) = default;
    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs) = default;

public:
    //------------------------------------------------------------------------//
    template <typename... _Args>
    task_type<_Args...>* operator+=(task_type<_Args...>* _task)
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
    task_type<_Args...>* wrap(_Func&& func, _Args&&... args)
    {
        return operator+=(new task_type<_Args...>(this, std::forward<_Func>(func),
                                                  std::forward<_Args>(args)...));
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(_Func&& func, _Args&&... args)
    {
        m_pool->add_task(wrap(std::forward<_Func>(func), std::move(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(_Func&& func, _Args&&... args)
    {
        m_pool->add_task(wrap(std::forward<_Func>(func), std::move(args)...));
    }

protected:
    //------------------------------------------------------------------------//
    // shorter typedefs
    typedef iterator               itr_t;
    typedef const_iterator         citr_t;
    typedef reverse_iterator       ritr_t;
    typedef const_reverse_iterator critr_t;

public:
    //------------------------------------------------------------------------//
    // Get tasks with non-void return types
    //
    task_list_t&       get_tasks() { return m_task_set; }
    const task_list_t& get_tasks() const { return m_task_set; }

    //------------------------------------------------------------------------//
    // iterate over tasks with return type
    //
    itr_t   begin() { return m_task_set.begin(); }
    itr_t   end() { return m_task_set.end(); }
    citr_t  begin() const { return m_task_set.begin(); }
    citr_t  end() const { return m_task_set.end(); }
    citr_t  cbegin() const { return m_task_set.begin(); }
    citr_t  cend() const { return m_task_set.end(); }
    ritr_t  rbegin() { return m_task_set.rbegin(); }
    ritr_t  rend() { return m_task_set.rend(); }
    critr_t rbegin() const { return m_task_set.rbegin(); }
    critr_t rend() const { return m_task_set.rend(); }

    //------------------------------------------------------------------------//
    // wait to finish
    template <typename _Up = _Tp, enable_if_t<!std::is_same<_Up, void>::value, int> = 0>
    inline _Up join(_Up accum = {})
    {
        this->wait();
        for(auto& itr : m_task_set)
        {
            using RetType = decltype(itr.get());
            accum = std::move(m_join(std::ref(accum), std::forward<RetType>(itr.get())));
        }
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
    //------------------------------------------------------------------------//
    // clear the task result history
    void clear()
    {
        m_task_set.clear();
        VTaskGroup::clear();
    }

protected:
    // Protected variables
    task_list_t m_task_set;
    join_type   m_join;
};
