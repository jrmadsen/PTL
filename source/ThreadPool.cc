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
//  Tasking class implementation
//
// Class Description:
//
// This file creates a class for an efficient thread-pool that
// accepts work in the form of tasks.
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#include "PTL/ThreadPool.hh"
#include "PTL/ThreadData.hh"
#include "PTL/UserTaskQueue.hh"
#include "PTL/VUserTaskQueue.hh"

#include <cstdlib>

#if defined(PTL_USE_GPERF)
#    include <gperftools/heap-checker.h>
#    include <gperftools/heap-profiler.h>
#    include <gperftools/profiler.h>
#endif

//============================================================================//

inline intmax_t
ncores()
{
    return static_cast<intmax_t>(Thread::hardware_concurrency());
}

//============================================================================//

ThreadPool::thread_id_map_t    ThreadPool::f_thread_ids;
ThreadPool::thread_index_map_t ThreadPool::f_thread_indexes;

//============================================================================//

namespace state
{
static const int STARTED = 0;
static const int STOPPED = 1;
static const int NONINIT = 2;
}

//============================================================================//

namespace
{
ThreadData*&
thread_data()
{
    return ThreadData::GetInstance();
}
}

//============================================================================//

bool ThreadPool::f_use_tbb = false;

//============================================================================//
// static member function that calls the member function we want the thread to
// run
void
ThreadPool::start_thread(ThreadPool* tp)
{
    {
        AutoLock lock(TypeMutex<ThreadPool>());
        auto     _idx                            = f_thread_ids.size();
        f_thread_ids[std::this_thread::get_id()] = _idx;
        f_thread_indexes[_idx]                   = std::this_thread::get_id();
        // tp->get_queue()->ThisThreadNumber() = _idx;
    }
    static thread_local std::unique_ptr<ThreadData> _unique_data(
        new ThreadData(tp));
    thread_data() = _unique_data.get();
    tp->execute_thread(thread_data()->current_queue);
}

//============================================================================//
// static member function that initialized tbb library
void
ThreadPool::set_use_tbb(bool enable)
{
#if defined(PTL_USE_TBB)
    f_use_tbb = enable;
#else
    ConsumeParameters<bool>(enable);
#endif
}

//============================================================================//

uintmax_t
ThreadPool::GetThisThreadID()
{
    auto _tid = ThisThread::get_id();
    {
        AutoLock lock(TypeMutex<ThreadPool>());
        if(f_thread_ids.find(_tid) == f_thread_ids.end())
        {
            auto _idx              = f_thread_ids.size();
            f_thread_ids[_tid]     = _idx;
            f_thread_indexes[_idx] = _tid;
        }
    }
    return f_thread_ids[_tid];
}

//============================================================================//

ThreadPool::ThreadPool(const size_type& pool_size, VUserTaskQueue* task_queue,
                       bool _use_affinity, affinity_func_t _affinity_func)
: m_use_affinity(_use_affinity)
, m_tbb_tp(false)
, m_alive_flag(false)
, m_verbose(0)
, m_pool_size(0)
, m_recursive_limit(GetEnv<size_type>("TP_RECURSIVE_LIMIT", 20))
, m_pool_state(state::NONINIT)
, m_master_tid(ThisThread::get_id())
, m_thread_awake(new atomic_int_type(0))
, m_task_queue(task_queue)
, m_tbb_task_group(nullptr)
, m_init_func([]() { return; })
, m_affinity_func(_affinity_func)
{
    m_verbose = GetEnv<int>("PTL_VERBOSE", m_verbose);

    if(!m_task_queue)
        m_task_queue = new UserTaskQueue(pool_size);

    auto master_id = GetThisThreadID();
    if(master_id != 0 && m_verbose > 1)
        std::cerr << "ThreadPool created on non-master slave" << std::endl;

    thread_data() = new ThreadData(this);

    // initialize after GetThisThreadID so master is zero
    this->initialize_threadpool(pool_size);
}

//============================================================================//

ThreadPool::~ThreadPool()
{
    delete m_thread_awake;
    // Release resources
    if(m_pool_state.load() != state::STOPPED)
    {
        size_type ret = destroy_threadpool();
        while(ret > 0)
            ret = stop_thread();
    }

    // wait until thread pool is fully destroyed
    while(is_alive())
        ;

    // delete thread-local allocator and erase thread IDS
    auto      _tid = std::this_thread::get_id();
    uintmax_t _idx = ThreadPool::GetThreadIDs().find(_tid)->second;

    if(f_thread_ids.find(_tid) != f_thread_ids.end())
        f_thread_ids.erase(f_thread_ids.find(_tid));

    if(f_thread_indexes.find(_idx) != f_thread_indexes.end())
        f_thread_indexes.erase(f_thread_indexes.find(_idx));

    // deleted by ThreadData
    // delete m_task_queue;
}

//============================================================================//

bool
ThreadPool::is_initialized() const
{
    return !(m_pool_state.load() == state::NONINIT);
}

//============================================================================//

void
ThreadPool::resize(size_type _n)
{
    if(_n == m_pool_size)
        return;
    initialize_threadpool(_n);
    m_task_queue->resize(static_cast<intmax_t>(_n));
}

//============================================================================//

void
ThreadPool::set_affinity(intmax_t i)
{
    if(!(i < static_cast<intmax_t>(m_main_threads.size())))
        return;

    Thread* _thread = m_main_threads.at(i);

    try
    {
        NativeThread native_thread = _thread->native_handle();
        intmax_t     _pin          = m_affinity_func(i);
        if(m_verbose > 0)
        {
            std::cout << "Setting pin affinity for thread " << _thread->get_id()
                      << " to " << _pin << std::endl;
        }
        Threading::SetPinAffinity(_pin, native_thread);
    }
    catch(std::runtime_error& e)
    {
        std::cout << "Error setting pin affinity" << std::endl;
        std::cerr << e.what() << std::endl;  // issue assigning affinity
    }
}

//============================================================================//

ThreadPool::size_type
ThreadPool::initialize_threadpool(size_type proposed_size)
{
    //--------------------------------------------------------------------//
    // return before initializing
    if(proposed_size < 1)
        return 0;

    //--------------------------------------------------------------------//
    // store that has been started
    if(!m_alive_flag.load())
        m_pool_state.store(state::STARTED);

        //--------------------------------------------------------------------//
        // handle tbb task scheduler
#ifdef PTL_USE_TBB
    if(f_use_tbb)
    {
        m_tbb_tp                               = true;
        m_pool_size                            = proposed_size;
        tbb_task_scheduler_t*& _task_scheduler = tbb_task_scheduler();
        // delete if wrong size
        if(m_pool_size != proposed_size)
        {
            delete _task_scheduler;
            _task_scheduler = nullptr;
        }

        if(!_task_scheduler)
            _task_scheduler =
                new tbb_task_scheduler_t(tbb::task_scheduler_init::deferred);

        if(!_task_scheduler->is_active())
        {
            m_pool_size = proposed_size;
            _task_scheduler->initialize(proposed_size + 1);
            if(m_verbose > 0)
                std::cout << "ThreadPool [TBB] initialized with " << m_pool_size
                          << " threads." << std::endl;
        }
        // create task group (used for async)
        if(!m_tbb_task_group)
            m_tbb_task_group = new tbb_task_group_t();
        return m_pool_size;
    }
    else if(tbb_task_scheduler())
    {
        m_tbb_tp                               = false;
        tbb_task_scheduler_t*& _task_scheduler = tbb_task_scheduler();
        if(_task_scheduler)
        {
            _task_scheduler->terminate();
            delete _task_scheduler;
            _task_scheduler = nullptr;
        }
        // delete task group (used for async)
        if(m_tbb_task_group)
        {
            m_tbb_task_group->wait();
            delete m_tbb_task_group;
            m_tbb_task_group = nullptr;
        }
    }
#endif

    m_alive_flag.store(true);

    //--------------------------------------------------------------------//
    // if started, stop some thread if smaller or return if equal
    if(m_pool_state.load() == state::STARTED)
    {
        if(m_pool_size > proposed_size)
        {
            while(stop_thread() > proposed_size)
                ;
            if(m_verbose > 0)
                std::cout << "ThreadPool initialized with " << m_pool_size
                          << " threads." << std::endl;
            return m_pool_size;
        }
        else if(m_pool_size == proposed_size)
        {
            if(m_verbose > 0)
                std::cout << "ThreadPool initialized with " << m_pool_size
                          << " threads." << std::endl;
            return m_pool_size;
        }
    }

    //--------------------------------------------------------------------//
    // reserve enough space to prevent realloc later
    {
        AutoLock _task_lock(m_task_lock);
        m_main_threads.reserve(proposed_size);
        m_is_joined.reserve(proposed_size);
    }

    for(size_type i = m_pool_size; i < proposed_size; ++i)
    {
        // add the threads
        Thread* tid = new Thread;
        try
        {
            *tid = Thread(ThreadPool::start_thread, this);
            // only reaches here if successful creation of thread
            ++m_pool_size;
            // store thread
            m_main_threads.push_back(tid);
            // list of joined thread booleans
            m_is_joined.push_back(false);
        }
        catch(std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;  // issue creating thread
            continue;
        }
        catch(std::bad_alloc& e)
        {
            std::cerr << e.what() << std::endl;
            continue;
        }

        if(m_use_affinity)
            set_affinity(i);
    }
    //------------------------------------------------------------------------//

    AutoLock _task_lock(m_task_lock);

    // thread pool size doesn't match with join vector
    // this will screw up joining later
    if(m_is_joined.size() != m_main_threads.size())
    {
        std::stringstream ss;
        ss << "ThreadPool::initialize_threadpool - boolean is_joined vector "
           << "is a different size than threads vector: " << m_is_joined.size()
           << " vs. " << m_main_threads.size()
           << " (tid: " << std::this_thread::get_id() << ")";

        throw std::runtime_error(ss.str());
    }

    if(m_verbose > 0)
        std::cout << "ThreadPool initialized with " << m_pool_size
                  << " threads." << std::endl;

    return m_main_threads.size();
}

//============================================================================//

ThreadPool::size_type
ThreadPool::destroy_threadpool()
{
    // Note: this is not for synchronization, its for thread communication!
    // destroy_threadpool() will only be called from the main thread, yet
    // the modified m_pool_state may not show up to other threads until its
    // modified in a lock!
    //------------------------------------------------------------------------//
    m_pool_state.store(state::STOPPED);

    //--------------------------------------------------------------------//
    // handle tbb task scheduler
#ifdef PTL_USE_TBB
    if(m_tbb_tp && tbb_task_scheduler())
    {
        tbb_task_scheduler_t*& _task_scheduler = tbb_task_scheduler();
        delete _task_scheduler;
        _task_scheduler = nullptr;
        m_tbb_tp        = false;
        std::cout << "ThreadPool [TBB] destroyed" << std::endl;
    }
    if(m_tbb_task_group)
    {
        m_tbb_task_group->wait();
        delete m_tbb_task_group;
        m_tbb_task_group = nullptr;
    }
#endif

    if(!m_alive_flag.load())
        return 0;

    //------------------------------------------------------------------------//
    // notify all threads we are shutting down
    m_task_lock.lock();
    CONDITIONBROADCAST(&m_task_cond);
    m_task_lock.unlock();
    //------------------------------------------------------------------------//

    if(m_is_joined.size() != m_main_threads.size())
    {
        std::stringstream ss;
        ss << "   ThreadPool::destroy_thread_pool - boolean is_joined vector "
           << "is a different size than threads vector: " << m_is_joined.size()
           << " vs. " << m_main_threads.size()
           << " (tid: " << std::this_thread::get_id() << ")";

        throw std::runtime_error(ss.str());
    }

    for(size_type i = 0; i < m_is_joined.size(); i++)
    {
        //--------------------------------------------------------------------//
        // if its joined already, nothing else needs to be done
        if(m_is_joined.at(i))
            continue;

        //--------------------------------------------------------------------//
        // join
        if(!(std::this_thread::get_id() == m_main_threads[i]->get_id()))
            m_main_threads[i]->join();

        //--------------------------------------------------------------------//
        // thread id and index
        auto _tid = m_main_threads[i]->get_id();
        auto _idx = f_thread_ids[_tid];

        //--------------------------------------------------------------------//
        // erase thread from thread ID list
        if(f_thread_ids.find(_tid) != f_thread_ids.end())
            f_thread_ids.erase(f_thread_ids.find(_tid));

        //--------------------------------------------------------------------//
        // erase thread from thread index list
        if(f_thread_indexes.find(_idx) != f_thread_indexes.end())
            f_thread_indexes.erase(f_thread_indexes.find(_idx));

        //--------------------------------------------------------------------//
        // it's joined
        m_is_joined.at(i) = true;

        //--------------------------------------------------------------------//
        // try waking up a bunch of threads that are still waiting
        CONDITIONBROADCAST(&m_task_cond);
        //--------------------------------------------------------------------//
    }

    for(auto& itr : m_main_threads)
        delete itr;

    m_main_threads.clear();
    m_is_joined.clear();

    m_alive_flag.store(false);

    std::cout << "ThreadPool destroyed" << std::endl;

    return 0;
}

//============================================================================//

ThreadPool::size_type
ThreadPool::stop_thread()
{
    if(!m_alive_flag.load() || m_pool_size == 0)
        return 0;

    //------------------------------------------------------------------------//
    // notify all threads we are shutting down
    m_task_lock.lock();
    m_is_stopped.push_back(true);
    CONDITIONNOTIFY(&m_task_cond);
    m_task_lock.unlock();
    //------------------------------------------------------------------------//

    // lock up the task queue
    AutoLock _task_lock(m_task_lock);

    while(!m_stop_threads.empty())
    {
        // get the thread
        Thread* t = m_stop_threads.back();
        // let thread finish
        t->join();
        // remove from stopped
        m_stop_threads.pop_back();
        // remove from main
        for(auto itr = m_main_threads.begin(); itr != m_main_threads.end();
            ++itr)
            if((*itr)->get_id() == t->get_id())
            {
                m_main_threads.erase(itr);
                break;
            }
        // remove from join list
        m_is_joined.pop_back();
        // delete thread
        delete t;
    }

    m_pool_size = m_main_threads.size();
    return m_main_threads.size();
}

//============================================================================//

void
ThreadPool::run(task_pointer task)
{
    // check the task_pointer (std::shared_ptr) has a valid pointer
    if(!task.get())
        return;

    // execute task
    (*task)();
}

//============================================================================//

int
ThreadPool::run_on_this(task_pointer task)
{
    auto _func = [=]() { this->run(task); };
    if(m_tbb_tp)
    {
        if(m_tbb_task_group)
            m_tbb_task_group->run(_func);
        else
            _func();
    }
    else  // execute task
        _func();

    // return the number of tasks added to task-list
    return 0;
}

//============================================================================//

int
ThreadPool::insert(task_pointer task, int bin)
{
    ThreadLocalStatic ThreadData* _data = thread_data();

    // pass the task to the queue
    auto ibin = m_task_queue->InsertTask(task, _data, bin);
    notify();
    return ibin;
}

//============================================================================//

ThreadPool::size_type
ThreadPool::add_task(task_pointer task, int bin)
{
    // if not native (i.e. TBB) then return
    if(!task->is_native_task())
        return 0;

    // if we haven't built thread-pool, just execute
    if(!m_alive_flag.load())
        return static_cast<size_type>(run_on_this(task));

    return static_cast<size_type>(insert(task, bin));
}

//============================================================================//

intmax_t
ThreadPool::GetEnvNumThreads(intmax_t _default)
{
    return GetEnv<intmax_t>("PTL_NUM_THREADS", _default);
}

//============================================================================//

void
ThreadPool::execute_thread(VUserTaskQueue* _task_queue)
{
#if defined(PTL_USE_GPERF)
    ProfilerRegisterThread();
#endif

    ++(*m_thread_awake);

    // initialization function
    m_init_func();

    ThreadId tid = ThisThread::get_id();

    ThreadData* data = thread_data();

    assert(data->current_queue != nullptr);
    assert(_task_queue == data->current_queue);

    // essentially a dummy run
    {
        data->within_task = true;
        run(_task_queue->GetTask());
        data->within_task = false;
    }

    // threads stay in this loop forever until thread-pool destroyed
    while(true)
    {
        //--------------------------------------------------------------------//
        // Try to pick a task
        AutoLock _task_lock(m_task_lock, std::defer_lock);
        //--------------------------------------------------------------------//

        // We need to put condition.wait() in a loop for two reasons:
        // 1. There can be spurious wake-ups (due to signal/ENITR)
        // 2. When mutex is released for waiting, another thread can be woken up
        //    from a signal/broadcast and that thread can mess up the condition.
        //    So when the current thread wakes up the condition may no longer be
        //    actually true!
        while(_task_queue->empty())
        {
            // If the thread was waked to notify process shutdown, return from
            // here
            if(m_pool_state.load() == state::STOPPED)
            {
                // has exited.
                if(_task_lock.owns_lock())
                    _task_lock.unlock();
                return;
            }

            // single thread stoppage
            if(m_is_stopped.size() > 0)
            {
                if(!_task_lock.owns_lock())
                    _task_lock.lock();
                if(m_is_stopped.back())
                    m_stop_threads.push_back(get_thread(tid));
                m_is_stopped.pop_back();
                if(_task_lock.owns_lock())
                    _task_lock.unlock();
                // exit entire function
                return;
            }

            if(_task_queue->true_size() == 0)
            {
                if(m_thread_awake->load() > 0)
                    --(*m_thread_awake);

                // lock before sleeping on condition
                if(!_task_lock.owns_lock())
                    _task_lock.lock();

                // Wait until there is a task in the queue
                // Unlocks mutex while waiting, then locks it back when signaled
                // use long duration wait_for to keep overhead low but
                // spuriously wake up and check
                m_task_cond.wait_for(_task_lock, std::chrono::seconds(10));

                // unlock if owned
                if(_task_lock.owns_lock())
                    _task_lock.unlock();

                // notify that is awake
                if(m_thread_awake->load() < m_pool_size)
                    ++(*m_thread_awake);
            }
            else
                break;
        }

        // release the lock
        if(_task_lock.owns_lock())
            _task_lock.unlock();
        //----------------------------------------------------------------//

        // activate guard against recursive deadlock
        data->within_task = true;
        //----------------------------------------------------------------//

        // get the next task and execute the task (will return if nullptr)
        while(!_task_queue->empty())
            run(_task_queue->GetTask());
        //----------------------------------------------------------------//

        // disable guard against recursive deadlock
        data->within_task = false;
        //----------------------------------------------------------------//

        // release the lock
        // if(_task_lock.owns_lock())
        //    _task_lock.unlock();
        //----------------------------------------------------------------//
    }
}

//============================================================================//
