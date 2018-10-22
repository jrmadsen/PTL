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
// Tasking class header file
//
// Class Description:
//
// This file creates an abstract base class for the thread-pool tasking
// system
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#ifndef VTask_hh_
#define VTask_hh_

#include "PTL/Threading.hh"
#include "PTL/AutoLock.hh"
#include "PTL/TaskAllocator.hh"

#include <functional>
#include <utility>
#include <tuple>
#include <cstddef>
#include <string>
#include <future>
#include <thread>
#include <cstdint>
#include <atomic>

class VTaskGroup;
class ThreadPool;

//============================================================================//

/// \brief VTask is the abstract class stored in thread_pool
class VTask
{
public:
    typedef std::thread::id             tid_type;
    typedef size_t                      size_type;
    typedef VTask                       this_type;
    typedef std::atomic_uintmax_t       count_t;
    typedef VTask*                      iterator;
    typedef const VTask*                const_iterator;
    typedef TaskAllocator<this_type>    allocator_type;
    typedef std::function<void()>       void_func_t;

public:
    VTask(VTaskGroup* _group = nullptr);
    virtual ~VTask();
    VTask(void_func_t&& _func, VTaskGroup* _group = nullptr);

public:
    // execution operator
    virtual void operator()()
    {
        m_func();
        // decrements the task-group counter on active tasks
        // when the counter is < 2, if the thread owning the task group is
        // sleeping at the TaskGroup::wait(), it signals the thread to wake
        // up and check if all tasks are finished, proceeding if this
        // check returns as true
        this_type::operator--();
    }

public:
    // used by thread_pool
    void operator++();
    void operator--();
    virtual bool is_native_task() const;
    virtual ThreadPool* pool() const;

public:
    // used by task tree
    iterator begin() { return this; }
    iterator end() { return this+1; }

    const_iterator begin() const { return this; }
    const_iterator end() const { return this+1; }

    const_iterator cbegin() const { return this; }
    const_iterator cend() const { return this+1; }

    intmax_t& depth() { return m_depth; }
    const intmax_t& depth() const { return m_depth; }

    void set_tid_bin(const tid_type& _bin) { m_tid_bin = _bin; }

protected:
    static tid_type this_tid() { return std::this_thread::get_id(); }

protected:
    VTaskGroup* m_vgroup;
    tid_type    m_tid_bin;
    intmax_t    m_depth;
    void_func_t m_func = [](){};

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
        ThreadLocalStatic allocator_ptr _allocator = new allocator_type();
        return _allocator;
    }
};

//============================================================================//

#endif

