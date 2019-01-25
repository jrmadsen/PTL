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
// Global utility functions
//

#pragma once

#include "PTL/Types.hh"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

//--------------------------------------------------------------------------------------//
// use this function to get rid of "unused parameter" warnings
//
template <typename _Tp, typename... _Args>
void
ConsumeParameters(_Tp, _Args...)
{
}

//--------------------------------------------------------------------------------------//

class EnvSettings
{
public:
    typedef std::mutex                        mutex_t;
    typedef std::string                       string_t;
    typedef std::multimap<string_t, string_t> env_map_t;
    typedef std::pair<string_t, string_t>     env_pair_t;

public:
    static EnvSettings* GetInstance()
    {
        static EnvSettings* _instance = new EnvSettings();
        return _instance;
    }

public:
    template <typename _Tp>
    void insert(const std::string& env_id, _Tp val)
    {
        std::stringstream ss;
        ss << std::boolalpha << val;
        m_mutex.lock();
        m_env.insert(env_pair_t(env_id, ss.str()));
        m_mutex.unlock();
    }

    const env_map_t& get() const { return m_env; }
    mutex_t&         mutex() const { return m_mutex; }

    friend std::ostream& operator<<(std::ostream& os, const EnvSettings& env)
    {
        std::stringstream filler;
        filler.fill('#');
        filler << std::setw(90) << "";
        std::stringstream ss;
        ss << filler.str() << "\n# Environment settings:\n";
        env.mutex().lock();
        for(const auto& itr : env.get())
        {
            ss << "# " << std::setw(35) << std::right << itr.first << "\t = \t"
               << std::left << itr.second << "\n";
        }
        env.mutex().unlock();
        ss << filler.str();
        os << ss.str() << std::endl;
        return os;
    }

private:
    env_map_t       m_env;
    mutable mutex_t m_mutex;
};

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting +
//  a default if not defined, e.g.
//      int num_threads =
//          GetEnv<int>("FORCENUMBEROFTHREADS",
//                          std::thread::hardware_concurrency());
//
template <typename _Tp>
_Tp
GetEnv(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp                var = _Tp();
        iss >> var;
        // record value defined by environment
        EnvSettings::GetInstance()->insert<_Tp>(env_id, var);
        return var;
    }
    // record default value
    EnvSettings::GetInstance()->insert<_Tp>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting +
//  a default if not defined, e.g.
//      int num_threads =
//          GetEnv<int>("FORCENUMBEROFTHREADS",
//                          std::thread::hardware_concurrency());
//
template <>
inline bool
GetEnv(const std::string& env_id, bool _default)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string var = std::string(env_var);
        bool        val = true;
        if(var.find_first_not_of("0123456789") == std::string::npos)
            val = (bool) atoi(var.c_str());
        else
        {
            for(auto& itr : var)
                itr = tolower(itr);
            if(var == "off" || var == "false")
                val = false;
        }
        // record value defined by environment
        EnvSettings::GetInstance()->insert<bool>(env_id, val);
        return val;
    }
    // record default value
    EnvSettings::GetInstance()->insert<bool>(env_id, false);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting +
//  a default if not defined and a message about the setting, e.g.
//      int num_threads =
//          GetEnv<int>("FORCENUMBEROFTHREADS",
//                          std::thread::hardware_concurrency(),
//                          "Forcing number of threads");
//
template <typename _Tp>
_Tp
GetEnv(const std::string& env_id, _Tp _default, const std::string& msg)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp                var = _Tp();
        iss >> var;
        std::cout << "Environment variable \"" << env_id << "\" enabled with "
                  << "value == " << var << ". " << msg << std::endl;
        // record value defined by environment
        EnvSettings::GetInstance()->insert<_Tp>(env_id, var);
        return var;
    }
    // record default value
    EnvSettings::GetInstance()->insert<_Tp>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//

inline void
PrintEnv(std::ostream& os = std::cout)
{
    os << (*EnvSettings::GetInstance());
}

//--------------------------------------------------------------------------------------//
