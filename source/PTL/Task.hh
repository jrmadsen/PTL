//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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

#ifndef Task_hh_
#define Task_hh_

#include "VTask.hh"
#include "TaskGroup.hh"
#include "TaskAllocator.hh"

#include <stdexcept>
#include <functional>
#include <cstdint>

#define _forward_args_t(_Args, _args) std::forward<_Args>(std::move(_args))...

//============================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename _Ret, typename _Arg, typename... _Args>
class PackagedTask : public VTask
{
public:
    typedef PackagedTask<_Ret, _Arg, _Args...>              this_type;
    typedef _Ret                                            result_type;
    typedef std::function<_Arg(_Args...)>                   function_type;
    typedef TaskGroup<_Ret, _Arg>                           task_group_type;
    typedef typename task_group_type::promise_type          promise_type;
    typedef typename task_group_type::future_type           future_type;
    typedef typename task_group_type::packaged_task_type    packaged_task_type;
    typedef TaskAllocator<this_type>                        allocator_type;

public:
    // pass a free function pointer
    PackagedTask(function_type func, _Args... args)
    : VTask(nullptr),
      m_ptask(std::bind(func, _forward_args_t(_Args, args)))
    { }

    virtual ~PackagedTask() { }

public:
    // execution operator
    virtual void operator()() override
    {
        m_ptask();
    }

    future_type get_future() { return m_ptask.get_future(); }

    virtual bool is_native_task() const override { return true; }

public:
    // define the new operator
    void* operator new(size_type)
    {
        return static_cast<void*>(get_allocator()->MallocSingle());
    }
    // define the delete operator
    void operator delete(void* ptr)
    {
        get_allocator()->FreeSingle(static_cast<this_type*>(ptr));
    }

private:
    // currently disabled due to memory leak found via -fsanitize=leak
    // static function to get allocator
    static allocator_type*& get_allocator()
    {
        typedef allocator_type* allocator_ptr;
        ThreadLocalStatic allocator_ptr _allocator = new allocator_type;
        return _allocator;
    }

private:
    packaged_task_type      m_ptask;
};

//============================================================================//

/// \brief The task class is supplied to thread_pool.
template <typename _Ret, typename _Arg, typename... _Args>
class Task : public VTask
{
public:
    typedef Task<_Ret, _Arg, _Args...>                      this_type;
    typedef _Ret                                            result_type;
    typedef TaskGroup<_Ret, _Arg>                           task_group_type;
    typedef typename task_group_type::ArgTp                 ArgTp;
    typedef typename task_group_type::promise_type          promise_type;
    typedef typename task_group_type::future_type           future_type;
    typedef typename task_group_type::packaged_task_type    packaged_task_type;
    typedef std::function<ArgTp(_Args...)>                  function_type;
    typedef TaskAllocator<this_type>                        allocator_type;

public:
    // pass a free function pointer
    Task(task_group_type* tg, function_type func, _Args... args)
    : VTask(tg),
      m_ptask(std::bind(func, _forward_args_t(_Args, args)))
    {
        //m_tid_bin = tg->add(m_ptask.get_future());
    }

    // pass a free function pointer
    Task(task_group_type& tg, function_type func, _Args... args)
    : VTask(&tg),
      m_ptask(std::bind(func, _forward_args_t(_Args, args)))
    {
        //m_tid_bin = tg.add(m_ptask.get_future());
    }

    virtual ~Task() { }

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

public:
    // define the new operator
    void* operator new(size_type)
    {
        return static_cast<void*>(get_allocator()->MallocSingle());
    }
    // define the delete operator
    void operator delete(void* ptr)
    {
        get_allocator()->FreeSingle(static_cast<this_type*>(ptr));
    }

private:
    // currently disabled due to memory leak found via -fsanitize=leak
    // static function to get allocator
    static allocator_type*& get_allocator()
    {
        typedef allocator_type* allocator_ptr;
        ThreadLocalStatic allocator_ptr _allocator = new allocator_type;
        return _allocator;
    }

private:
    packaged_task_type      m_ptask;
};


//============================================================================//

/// \brief The task class is supplied to thread_pool.
template <>
class Task<void, void> : public VTask
{
public:
    typedef Task<void, void>                                this_type;
    typedef void                                            _Ret;
    typedef _Ret                                            result_type;
    typedef std::function<_Ret()>                           function_type;
    typedef TaskGroup<_Ret, _Ret>                           task_group_type;
    typedef typename task_group_type::promise_type          promise_type;
    typedef typename task_group_type::future_type           future_type;
    typedef typename task_group_type::packaged_task_type    packaged_task_type;
    typedef TaskAllocator<this_type>                        allocator_type;

public:
    // pass a free function pointer
    Task(task_group_type* tg, function_type func)
    : VTask(tg),
      m_ptask(func)
    {
        //m_tid_bin = tg->add(m_ptask.get_future());
    }

    Task(task_group_type& tg, function_type func)
    : VTask(&tg),
      m_ptask(func)
    {
        //m_tid_bin = tg.add(m_ptask.get_future());
    }

    virtual ~Task() { }

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

public:
    // define the new operator
    void* operator new(size_type)
    {
        return static_cast<void*>(get_allocator()->MallocSingle());
    }
    // define the delete operator
    void operator delete(void* ptr)
    {
        get_allocator()->FreeSingle(static_cast<this_type*>(ptr));
    }

private:
    // currently disabled due to memory leak found via -fsanitize=leak
    // static function to get allocator
    static allocator_type*& get_allocator()
    {
        typedef allocator_type* allocator_ptr;
        ThreadLocalStatic allocator_ptr _allocator = new allocator_type;
        return _allocator;
    }

private:
    packaged_task_type      m_ptask;
};

//============================================================================//

// don't pollute
#undef _forward_args_t

#endif
