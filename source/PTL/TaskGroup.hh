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
, public CountedObject<TaskGroup<_Tp, _Arg>>
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

    typedef _Tp                                          result_type;
    typedef TaskGroup<_Tp, _Arg>                         this_type;
    typedef std::promise<ArgTp>                          promise_type;
    typedef std::future<ArgTp>                           future_type;
    typedef std::packaged_task<ArgTp()>                  packaged_task_type;
    typedef std::tuple<bool, future_type, ArgTp>         data_type;
    typedef list_type<data_type>                         task_list_t;
    typedef std::function<_Tp(_Tp&, _Arg)>               function_type;
    typedef typename task_list_t::iterator               iterator;
    typedef typename task_list_t::reverse_iterator       reverse_iterator;
    typedef typename task_list_t::const_iterator         const_iterator;
    typedef typename task_list_t::const_reverse_iterator const_reverse_iterator;

public:
    // Constructor
    template <typename _Func>
    TaskGroup(const _Func& _join, ThreadPool* tp = nullptr);
    template <typename _Func>
    TaskGroup(int _freq, const _Func& _join, ThreadPool* tp = nullptr);
    // Destructor
    virtual ~TaskGroup();

    // delete copy-construct
    TaskGroup(const this_type&) = delete;
    // define move-construct
    TaskGroup(this_type&& rhs)
    : m_task_set(std::move(rhs.m_task_set))
    , m_promise(std::move(rhs.m_promise))
    , m_join_function(std::move(rhs.m_join_function))
    {
    }

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs)
    {
        if(this != &rhs)
        {
            m_task_set      = std::move(rhs.m_task_set);
            m_promise       = std::move(rhs.m_promise);
            m_join_function = std::move(rhs.m_join_function);
        }
        return *this;
    }

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
        m_task_set.push_back(data_type(false, std::move(_task->get_future()), ArgTp()));
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
        // add the future
        m_task_set.push_back(data_type(false, std::move(_task->get_future()), ArgTp()));
        // return
        return _task;
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    task_pointer<_Args...> wrap(const _Func& func, _Args... args)
    {
        auto _task = task_pointer<_Args...>(
            new task_type<_Args...>(this, func, std::forward<_Args>(args)...));
        return operator+=(_task);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    func_task_pointer wrap(const _Func& func)
    {
        auto   _task = func_task_pointer(new func_task_type(this, func));
        return operator+=(_task);
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(const _Func& func, _Args... args)
    {
        if(m_task_set.size() > 10000)
            func(std::forward<_Args>(args)...);
        else
            m_pool->add_task(wrap(func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void exec(const _Func& func)
    {
        if(m_task_set.size() > 10000)
            func();
        else
            m_pool->add_task(wrap(func));
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(const _Func& func, _Args... args)
    {
        if(m_task_set.size() > 10000)
            func(std::forward<_Args>(args)...);
        else
            m_pool->add_task(wrap(func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void run(const _Func& func)
    {
        if(m_task_set.size() > 10000)
            func();
        else
            m_pool->add_task(wrap(func));
    }

public:
    //------------------------------------------------------------------------//
    // set the join function
    template <typename _Func>
    void set_join_function(const _Func&);

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
    _Tp join(_Tp accum = _Tp());
    //------------------------------------------------------------------------//
    // clear the task result history
    void clear() override
    {
        m_task_set.clear();
        VTaskGroup::clear();
    }

protected:
    //------------------------------------------------------------------------//
    // get a specific task
    ArgTp get(data_type& _data);

protected:
    // Protected variables
    task_list_t   m_task_set;
    promise_type  m_promise;
    function_type m_join_function;
};

//--------------------------------------------------------------------------------------//
// specialization for void type
template <>
class TaskGroup<void, void>
: public VTaskGroup
, public TaskAllocator<TaskGroup<void, void>>
, public CountedObject<TaskGroup<void, void>>

{
public:
    using ArgTp = void;

    template <typename... _Args>
    using task_type = Task<ArgTp, _Args...>;

    template <typename... _Args>
    using task_pointer = std::shared_ptr<task_type<_Args...>>;

    using func_task_type    = Task<ArgTp>;
    using func_task_pointer = std::shared_ptr<func_task_type>;

    typedef void                                         result_type;
    typedef TaskGroup<void, void>                        this_type;
    typedef std::promise<void>                           promise_type;
    typedef std::future<void>                            future_type;
    typedef std::packaged_task<void()>                   packaged_task_type;
    typedef std::tuple<bool, future_type>                data_type;
    typedef list_type<data_type>                         task_list_t;
    typedef std::function<void()>                        function_type;
    typedef typename task_list_t::iterator               iterator;
    typedef typename task_list_t::reverse_iterator       reverse_iterator;
    typedef typename task_list_t::const_iterator         const_iterator;
    typedef typename task_list_t::const_reverse_iterator const_reverse_iterator;

public:
    // Constructor
    explicit TaskGroup(ThreadPool* tp = nullptr)
    : VTaskGroup(tp)
    , m_join_function([]() {})
    {
    }
    template <typename _Func>
    TaskGroup(const _Func& _join, ThreadPool* tp = nullptr)
    : VTaskGroup(tp)
    {
        set_join_function(_join);
    }
    // Destructor
    virtual ~TaskGroup() {}

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
    task_pointer<_Args...>& operator+=(task_pointer<_Args...>& _task)
    {
        // store in list
        vtask_list.push_back(_task);
        // thread-safe increment of tasks in task group
        operator++();
        // add the future
        // m_task_set.push_back(data_type(false, std::move(_task->get_future())));
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
        // add the future
        // m_task_set.push_back(data_type(false, _task->get_future()));
        // return
        return _task;
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    task_pointer<_Args...> wrap(const _Func& func, _Args... args)
    {
        auto _task = task_pointer<_Args...>(
            new task_type<_Args...>(this, func, std::forward<_Args>(args)...));
        return operator+=(_task);
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    func_task_pointer wrap(const _Func& func)
    {
        auto   _task = func_task_pointer(new func_task_type(this, func));
        return operator+=(_task);
    }

public:
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void exec(const _Func& func, _Args... args)
    {
        if(CountedObject<this_type>::live() > 1000)
            func(std::forward<_Args>(args)...);
        else
            m_pool->add_task(wrap(func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void exec(const _Func& func)
    {
        if(CountedObject<this_type>::live() > 1000)
            func();
        else
            m_pool->add_task(wrap(func));
    }
    //------------------------------------------------------------------------//
    template <typename _Func, typename... _Args>
    void run(const _Func& func, _Args... args)
    {
        if(CountedObject<this_type>::live() > 1000)
            func(std::forward<_Args>(args)...);
        else
            m_pool->add_task(wrap(func, std::forward<_Args>(args)...));
    }
    //------------------------------------------------------------------------//
    template <typename _Func>
    void run(const _Func& func)
    {
        if(CountedObject<this_type>::live() > 1000)
            func();
        else
            m_pool->add_task(wrap(func));
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

    //------------------------------------------------------------------------//
    // join function
    template <typename _Func>
    void set_join_function(_Func& _join)
    {
        m_join_function = std::bind<void>(_join);
    }
    //------------------------------------------------------------------------//
    // wait to finish
    void join()
    {
        this->wait();
        m_join_function();

        // if(m_clear_freq.load() > 0 && (++m_clear_count) % m_clear_freq.load() == 0)
        this->clear();
    }
    //------------------------------------------------------------------------//
    // clear the task result history
    void clear() override { VTaskGroup::clear(); }

protected:
    // Private variables
    function_type m_join_function;
};

//--------------------------------------------------------------------------------------//

#include "PTL/TaskGroup.icc"
