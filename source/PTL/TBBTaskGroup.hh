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
// Tasking class header file
//
// Class Description:
//
// This file wraps a TBB task_group into a TaskGroup
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Jun 21st 2018)
// ---------------------------------------------------------------

#ifndef g4tbbtaskgroup_hh_
#define g4tbbtaskgroup_hh_

#include "PTL/TaskGroup.hh"

class ThreadPool;

#if defined(PTL_USE_TBB)

#    include <tbb/tbb.h>

//----------------------------------------------------------------------------//

template <typename _Tp, typename _Arg = _Tp>
class TBBTaskGroup : public TaskGroup<_Tp, _Arg>
{
public:
    typedef TBBTaskGroup<_Tp, _Arg>                this_type;
    typedef TaskGroup<_Tp, _Arg>                   base_type;
    typedef typename base_type::result_type        result_type;
    typedef typename base_type::ArgTp              ArgTp;
    typedef typename VTaskGroup::tid_type          tid_type;
    typedef typename base_type::data_type          data_type;
    typedef typename base_type::packaged_task_type packaged_task_type;
    typedef typename base_type::future_type        future_type;
    typedef typename base_type::promise_type       promise_type;
    typedef tbb::task_group                        tbb_task_group_t;

public:
    // Constructor
    template <typename _Func>
    TBBTaskGroup(_Func _join, ThreadPool* tp = nullptr);
    template <typename _Func>
    TBBTaskGroup(int _freq, _Func _join, ThreadPool* tp = nullptr);
    // Destructor
    virtual ~TBBTaskGroup();

    // delete copy-construct
    TBBTaskGroup(const this_type&) = delete;
    // define move-construct
    TBBTaskGroup(this_type&& rhs)
    : m_tbb_task_group(std::move(rhs.m_tbb_task_group))
    {
        m_task_set      = std::move(rhs.m_task_set);
        m_promise       = std::move(rhs.m_promise);
        m_join_function = std::move(rhs.m_join_function);
    }

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs)
    {
        if(this != &rhs)
        {
            m_task_set       = std::move(rhs.m_task_set);
            m_promise        = std::move(rhs.m_promise);
            m_join_function  = std::move(rhs.m_join_function);
            m_tbb_task_group = std::move(rhs.m_tbb_task_group);
        }
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    // add task
    tid_type add(packaged_task_type*);

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
    using base_type::m_tot_task_count;
};

//----------------------------------------------------------------------------//
// specialization for void type
template <> class TBBTaskGroup<void, void> : public TaskGroup<void, void>
{
public:
    typedef TBBTaskGroup<void, void>               this_type;
    typedef TaskGroup<void, void>                  base_type;
    typedef typename base_type::result_type        result_type;
    typedef typename base_type::ArgTp              ArgTp;
    typedef typename VTaskGroup::tid_type          tid_type;
    typedef typename base_type::data_type          data_type;
    typedef typename base_type::packaged_task_type packaged_task_type;
    typedef typename base_type::future_type        future_type;
    typedef typename base_type::promise_type       promise_type;
    typedef tbb::task_group                        tbb_task_group_t;

public:
    // Constructor
    TBBTaskGroup(ThreadPool* _tp = nullptr)
    : base_type(_tp)
    , m_tbb_task_group(new tbb_task_group_t())
    {
    }

    // Destructor
    virtual ~TBBTaskGroup() { delete m_tbb_task_group; }

    // delete copy-construct
    TBBTaskGroup(const this_type&) = delete;
    // define move-construct
    TBBTaskGroup(this_type&& rhs)
    : m_tbb_task_group(std::move(rhs.m_tbb_task_group))
    {
        m_task_set = std::move(rhs.m_task_set);
    }

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs)
    {
        if(this != &rhs)
        {
            m_task_set       = std::move(rhs.m_task_set);
            m_tbb_task_group = std::move(rhs.m_tbb_task_group);
        }
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    // add task
    tid_type add(packaged_task_type* _task)
    {
        tid_type _tid  = this_tid();
        auto     _f    = _task->get_future();
        auto     _func = [=]() { (*_task)(); };
        m_tbb_task_group->run(_func);
        m_task_set.push_back(data_type(false, std::move(_f)));
        return _tid;
    }
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
    using TaskGroup<void, void>::m_task_set;
};

//----------------------------------------------------------------------------//
#else

template <typename _Tp, typename _Arg = _Tp>
using TBBTaskGroup = TaskGroup<_Tp, _Arg>;

#endif

//----------------------------------------------------------------------------//

#include "PTL/TBBTaskGroup.icc"

//----------------------------------------------------------------------------//

#endif
