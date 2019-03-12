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
//  ---------------------------------------------------------------
//  Tasking class header
//  Class Description:
//  ---------------------------------------------------------------
//  Author: Jonathan Madsen
//  ---------------------------------------------------------------

#pragma once

#include "PTL/Globals.hh"
#include "PTL/Threading.hh"
#include "PTL/Types.hh"
#include "PTL/VUserTaskQueue.hh"

#include <atomic>
#include <deque>
#include <list>
#include <queue>
#include <random>
#include <set>
#include <stack>

class VTask;
class VTaskGroup;
class TaskSubQueue;  // definition in UserTaskQueue.icc

class UserTaskQueue : public VUserTaskQueue
{
public:
    typedef std::shared_ptr<VTask>             VTaskPtr;
    typedef std::vector<TaskSubQueue*>         TaskSubQueueContainer;
    typedef std::default_random_engine         random_engine_t;
    typedef std::uniform_int_distribution<int> int_dist_t;

public:
    // Constructor and Destructors
    UserTaskQueue(intmax_t nworkers = -1, UserTaskQueue* = nullptr);
    // Virtual destructors are required by abstract classes
    // so add it by default, just in case
    virtual ~UserTaskQueue() override;

public:
    // Virtual  function for getting a task from the queue
    virtual VTaskPtr GetTask(intmax_t subq = -1, intmax_t nitr = -1) override;
    // Virtual function for inserting a task into the queue
    virtual intmax_t InsertTask(VTaskPtr, ThreadData* = nullptr,
                                intmax_t subq = -1) override;

    // Overload this function to hold threads
    virtual void Wait() override {}
    virtual void resize(intmax_t) override;

    virtual bool      empty() const override;
    virtual size_type size() const override;

    virtual size_type bin_size() const override;
    virtual bool      bin_empty() const override;

    inline bool      true_empty() const override;
    inline size_type true_size() const override;

    virtual void ExecuteOnAllThreads(ThreadPool* tp, function_type f) override;

    virtual void ExecuteOnSpecificThreads(ThreadIdSet tid_set, ThreadPool* tp,
                                          function_type f) override;

    virtual VUserTaskQueue* clone() override;

    virtual intmax_t GetThreadBin() const override;

protected:
    template <typename _Tp>
    class binner
    {
    public:
        binner(_Tp tot, _Tp n)
        : m_tot(tot)
        , m_incr((n % 2 == 0) ? -1 : 1)
        , m_idx(m_incr * n)
        , m_base(m_incr * tot)
        , m_last(m_incr * ((m_base - m_idx) % (m_incr * m_tot)))
        {
        }

        _Tp operator()()
        {
            auto _idx = m_base - m_idx;
            m_idx     = (m_idx + 1) % m_tot;
            return (m_last = m_incr * ((_idx) % (m_incr * m_tot)));
        }

        const _Tp& last() const { return m_last; }

    private:
        _Tp m_tot;
        _Tp m_incr;
        _Tp m_idx;
        _Tp m_base;
        _Tp m_last;
    };

protected:
    intmax_t GetInsertBin() const;

private:
    void AcquireHold();
    void ReleaseHold();

private:
    bool                       m_is_clone;
    intmax_t                   m_thread_bin;
    mutable intmax_t           m_insert_bin;
    std::atomic_bool*          m_hold;
    std::atomic_uintmax_t*     m_ntasks;
    Mutex*                     m_mutex;
    TaskSubQueueContainer*     m_subqueues;
    std::vector<int>           m_rand_list;
    std::vector<int>::iterator m_rand_itr;
};

//======================================================================================//

#include "PTL/UserTaskQueue.icc"

//======================================================================================//

inline bool
UserTaskQueue::empty() const
{
    return (m_ntasks->load(std::memory_order_relaxed) == 0);
}

//======================================================================================//

inline UserTaskQueue::size_type
UserTaskQueue::size() const
{
    return m_ntasks->load(std::memory_order_relaxed);
}

//======================================================================================//

inline UserTaskQueue::size_type
UserTaskQueue::bin_size() const
{
    return (*m_subqueues)[GetThreadBin()]->size();
}

//======================================================================================//

inline bool
UserTaskQueue::bin_empty() const
{
    return (*m_subqueues)[GetThreadBin()]->empty();
}

//======================================================================================//

inline bool
UserTaskQueue::true_empty() const
{
    // return !std::any_of(m_subqueues->begin(), m_subqueues->end(), [](TaskSubQueue* itr)
    // { return !itr->empty(); });
    for(const auto& itr : *m_subqueues)
        if(!itr->empty())
            return false;
    return true;
}

//======================================================================================//

inline UserTaskQueue::size_type
UserTaskQueue::true_size() const
{
    // return std::accumulate<TaskSubQueueContainer::iterator, size_type>(
    //    m_subqueues->begin(), m_subqueues->end(), 0,
    //    [](size_type& n, TaskSubQueue* itr) { return n += itr->size(); });
    size_type _n = 0;
    for(const auto& itr : *m_subqueues)
        _n += itr->size();
    return _n;
}

//======================================================================================//
