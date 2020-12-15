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
/// \file tasking.cc
/// \brief Example showing the usage of tasking

#include "PTL/PTL.hh"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

using namespace PTL;

static std::mt19937 rng;

//============================================================================//

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
Tp
random_entry(const std::vector<Tp>& v)
{
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

//============================================================================//

int
main(int argc, char** argv)
{
    ConsumeParameters(argc, argv);
    rng.seed(std::random_device()());
    Threading::SetThreadId(0);

    auto hwthreads = std::thread::hardware_concurrency();

    Timer total_timer;
    total_timer.Start();

    // Construct the default run manager
    bool use_tbb     = false;
    auto run_manager = TaskRunManager(use_tbb);
    run_manager.Initialize(hwthreads);

    // the TaskManager is a utility that wraps the function calls into tasks for the
    // ThreadPool
    TaskManager* task_manager = run_manager.GetTaskManager();

    //------------------------------------------------------------------------//
    //                                                                        //
    //                      Asynchronous examples/tests                       //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        long nfib      = 40;
        auto fib_async = task_manager->async<uint64_t>(fibonacci, nfib);
        auto fib_n     = fib_async.get();
        std::cout << "[async test] fibonacci(" << nfib << ") = " << fib_n << std::endl;
        std::cout << std::endl;
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                        TaskGroup examples/tests                        //
    //                                                                        //
    //------------------------------------------------------------------------//
    {
        long     nfib  = 35;
        uint64_t nloop = 100;

        auto join = [](long& lhs, long rhs) {
            std::stringstream ss;
            ss << "thread " << std::setw(4) << PTL::Threading::GetThreadId() << " adding "
               << rhs << " to " << lhs << std::endl;
            std::cout << ss.str();
            return lhs += rhs;
        };

        auto     entry = [](uint64_t n) {
            std::vector<double> v(n * 100, 0);
            for(auto& itr : v)
                itr = std::generate_canonical<double, 12>(rng);
            auto e = random_entry(v);
            std::stringstream ss;
            ss << "random entry from thread " << std::setw(4)
               << PTL::Threading::GetThreadId() << " was : " << std::setw(8)
               << std::setprecision(6) << std::fixed << e << std::endl;
            std::cout << ss.str();
        };

        TaskGroup<long> tgf(join);
        TaskGroup<void> tgv;
        for(uint64_t i = 0; i < nloop; ++i)
        {
            tgf.exec(fibonacci, nfib + (i % 4));
            tgv.exec(consume, 100);
            tgv.exec(do_sleep, 50);
            tgv.exec(entry, i + 1);
        }

        auto ret = tgf.join();
        tgv.join();
        std::cout << "fibonacci(" << nfib << ") * " << nloop << " = " << ret << std::endl;
        std::cout << std::endl;
    }

    //------------------------------------------------------------------------//
    //                                                                        //
    //                          Task-group example/test                       //
    //                                                                        //
    //------------------------------------------------------------------------//

    // print the time for the calculation
    total_timer.Stop();
    std::cout << "Total time: \t" << total_timer << std::endl;
}
