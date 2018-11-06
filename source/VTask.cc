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
// ---------------------------------------------------------------
//  Tasking class implementation
//
// Class Description:
//
// This file creates an abstract base class for the thread-pool tasking
// system
//
// ---------------------------------------------------------------
// Author: Jonathan Madsen (Feb 13th 2018)
// ---------------------------------------------------------------

#include "PTL/VTask.hh"
#include "PTL/VTaskGroup.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/ThreadData.hh"

//============================================================================//

VTask::VTask(VTaskGroup* _group)
: m_vgroup(_group),
  m_tid_bin(this_tid()),
  m_depth(0)
{
    //ThreadData* data = ThreadData::GetInstance();
    //if(data && data->within_task)
    //    m_depth = (data->task_depth += 1);
}

//============================================================================//

VTask::~VTask()
{
    //ThreadData* data = ThreadData::GetInstance();
    //if(data && data->within_task)
    //    data->task_depth -= 1;
}

//============================================================================//

void VTask::operator++()
{
    if(m_vgroup)
    {
        m_vgroup->increase(m_tid_bin);
    }
}

//============================================================================//

void VTask::operator--()
{
    if(m_vgroup)
    {
        intmax_t _count = m_vgroup->reduce(m_tid_bin);
        if(_count < 2)
        {
            //AutoLock l(m_vgroup->task_lock());
            //CONDITIONBROADCAST(&m_vgroup->task_cond());
            try
            {
                m_vgroup->task_cond().notify_all();
            }
            catch (std::system_error& e)
            {
                auto tid = ThreadPool::GetThisThreadID();
                AutoLock l(TypeMutex<decltype(std::cerr)>());
                std::cerr << "[" << tid << "] Caught system error: "
                          << e.what() << std::endl;
            }

        }
    }
}

//============================================================================//

bool VTask::is_native_task() const
{
    return (m_vgroup) ? m_vgroup->is_native_task_group() : false;
}

//============================================================================//

ThreadPool* VTask::pool() const
{
    return (m_vgroup) ? m_vgroup->pool() : nullptr;
}

//============================================================================//
