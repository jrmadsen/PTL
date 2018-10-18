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
// This file creates the a class for handling a group of tasks that
// can be independently joined
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#ifndef TaskGroup_hh_
#define TaskGroup_hh_

#include "PTL/VTaskGroup.hh"

#include <cstdint>
#include <future>
#include <vector>
#include <deque>
#include <list>

#ifdef PTL_USE_TBB
#   include <tbb/tbb.h>
#endif

class ThreadPool;

//----------------------------------------------------------------------------//

template <typename _Tp, typename _Arg = _Tp>
class TaskGroup : public VTaskGroup
{
public:
    typedef typename std::remove_const<
    typename std::remove_reference<_Arg>::type>::type       ArgTp;
    typedef _Tp                                             result_type;
    typedef TaskGroup<_Tp, _Arg>                          this_type;
    typedef std::promise<ArgTp>                             promise_type;
    typedef std::future<ArgTp>                              future_type;
    typedef std::packaged_task<ArgTp()>                     packaged_task_type;
    typedef std::tuple<bool, future_type, ArgTp>            data_type;
    typedef list_type<data_type>                            task_list_t;
    typedef map_type<tid_type, task_list_t>                 task_map_t;
    typedef std::function<_Tp(_Tp&, _Arg)>                  function_type;
    typedef typename task_list_t::iterator                  iterator;
    typedef typename task_list_t::reverse_iterator          reverse_iterator;
    typedef typename task_list_t::const_iterator            const_iterator;
    typedef typename task_list_t::const_reverse_iterator    const_reverse_iterator;

public:
    // Constructor
    template <typename _Func>
    TaskGroup(_Func _join, ThreadPool* tp = nullptr);
    template <typename _Func>
    TaskGroup(int _freq, _Func _join, ThreadPool* tp = nullptr);
    // Destructor
    virtual ~TaskGroup();

    // delete copy-construct
    TaskGroup(const this_type&) = delete;
    // define move-construct
    TaskGroup(this_type&& rhs)
    : m_task_set(std::move(rhs.m_task_set)),
      m_promise(std::move(rhs.m_promise)),
      m_join_function(std::move(rhs.m_join_function))
    { }

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs)
    {
        if(this != &rhs)
        {
            m_task_set = std::move(rhs.m_task_set);
            m_promise = std::move(rhs.m_promise);
            m_join_function = std::move(rhs.m_join_function);
        }
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    // set the join function
    template <typename _Func> void set_join_function(_Func);

protected:
    //------------------------------------------------------------------------//
    // shorter typedefs
    typedef iterator                itr_t;
    typedef const_iterator          citr_t;
    typedef reverse_iterator        ritr_t;
    typedef const_reverse_iterator  critr_t;

public:
    //------------------------------------------------------------------------//
    // Get tasks with non-void return types
    //
    task_list_t& get_tasks()              { return m_task_set; }
    const task_list_t& get_tasks() const  { return m_task_set; }

    //------------------------------------------------------------------------//
    // iterate over tasks with return type
    //
    itr_t   begin()           { return m_task_set.begin();   }
    itr_t   end()             { return m_task_set.end();     }
    citr_t  begin() const     { return m_task_set.begin();   }
    citr_t  end()   const     { return m_task_set.end();     }
    citr_t  cbegin() const    { return m_task_set.begin();   }
    citr_t  cend()   const    { return m_task_set.end();     }
    ritr_t  rbegin()          { return m_task_set.rbegin();  }
    ritr_t  rend()            { return m_task_set.rend();    }
    critr_t rbegin() const    { return m_task_set.rbegin();  }
    critr_t rend()   const    { return m_task_set.rend();    }

    //------------------------------------------------------------------------//
    // add task
    tid_type add(future_type&& _f);
    tid_type add(packaged_task_type*);
    //------------------------------------------------------------------------//
    // wait to finish
    _Tp join(_Tp accum = _Tp());
    //------------------------------------------------------------------------//
    // clear the task result history
    void clear() override { m_task_set.clear(); VTaskGroup::clear(); }

protected:
    //------------------------------------------------------------------------//
    // get a specific task
    ArgTp get(data_type& _data);

protected:
    // Protected variables
    task_list_t         m_task_set;
    promise_type        m_promise;
    function_type       m_join_function;
};

//----------------------------------------------------------------------------//
// specialization for void type
template <>
class TaskGroup<void, void> : public VTaskGroup
{
public:
    typedef void                                            ArgTp;
    typedef void                                            result_type;
    typedef TaskGroup<void, void>                         this_type;
    typedef std::promise<void>                              promise_type;
    typedef std::future<void>                               future_type;
    typedef std::packaged_task<void()>                      packaged_task_type;
    typedef std::tuple<bool, future_type>                   data_type;
    typedef list_type<data_type>                            task_list_t;
    typedef map_type<tid_type, task_list_t>                 task_map_t;
    typedef typename task_list_t::iterator                  iterator;
    typedef typename task_list_t::reverse_iterator          reverse_iterator;
    typedef typename task_list_t::const_iterator            const_iterator;
    typedef typename task_list_t::const_reverse_iterator    const_reverse_iterator;

public:
    // Constructor
    TaskGroup(ThreadPool* tp = nullptr) : VTaskGroup(tp) { }
    // Destructor
    virtual ~TaskGroup() { }

    // delete copy-construct
    TaskGroup(const this_type&) = delete;
    // define move-construct
    TaskGroup(this_type&& rhs)
    : m_task_set(std::move(rhs.m_task_set))
    { }

    // delete copy-assign
    this_type& operator=(const this_type& rhs) = delete;
    // define move-assign
    this_type& operator=(this_type&& rhs)
    {
        if(this != &rhs)
            m_task_set = std::move(rhs.m_task_set);
        return *this;
    }

protected:
    //------------------------------------------------------------------------//
    // shorter typedefs
    typedef iterator                itr_t;
    typedef const_iterator          citr_t;
    typedef reverse_iterator        ritr_t;
    typedef const_reverse_iterator  critr_t;

public:
    //------------------------------------------------------------------------//
    // Get tasks with non-void return types
    //
    task_list_t& get_tasks()              { return m_task_set; }
    const task_list_t& get_tasks() const  { return m_task_set; }

    //------------------------------------------------------------------------//
    // iterate over tasks with return type
    //
    itr_t   begin()           { return m_task_set.begin();   }
    itr_t   end()             { return m_task_set.end();     }
    citr_t  begin() const     { return m_task_set.begin();   }
    citr_t  end()   const     { return m_task_set.end();     }
    citr_t  cbegin() const    { return m_task_set.begin();   }
    citr_t  cend()   const    { return m_task_set.end();     }
    ritr_t  rbegin()          { return m_task_set.rbegin();  }
    ritr_t  rend()            { return m_task_set.rend();    }
    critr_t rbegin() const    { return m_task_set.rbegin();  }
    critr_t rend()   const    { return m_task_set.rend();    }

    //------------------------------------------------------------------------//
    // add task
    tid_type add(future_type&& _f)
    {
        tid_type _tid = this_tid();
        m_task_set.push_back(data_type(false, std::move(_f)));
        return _tid;
    }
    //------------------------------------------------------------------------//
    // add task
    tid_type add(packaged_task_type& _task)
    {
        tid_type _tid = this_tid();
        auto _f = _task.get_future();
        m_task_set.push_back(data_type(false, std::move(_f)));
        return _tid;
    }
    //------------------------------------------------------------------------//
    // wait to finish
    void join()
    {
        this->wait();
        for(auto itr = begin(); itr != end(); ++itr)
            this->get(*itr);

        if(m_clear_freq.load() > 0 &&
           (++m_clear_count) % m_clear_freq.load() == 0)
            this->clear();
    }
    //------------------------------------------------------------------------//
    // clear the task result history
    void clear() override { m_task_set.clear(); VTaskGroup::clear(); }

protected:
    //------------------------------------------------------------------------//
    // get specific task
    void get(data_type& _data)
    {
        if(!std::get<0>(_data))
        {
            std::get<1>(_data).get();
            std::get<0>(_data) = true;
        }
    }

protected:
    // Private variables
    mutable task_list_t  m_task_set;
};

//----------------------------------------------------------------------------//

#include "PTL/TaskGroup.icc"

#endif
