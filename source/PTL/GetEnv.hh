//
// MIT License
// Copyright (c) 2020 Jonathan R. Madsen
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

#pragma once

#include <cctype>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>  // IWYU pragma: keep
#include <string>
#include <tuple>
#include <utility>

namespace PTL
{
//--------------------------------------------------------------------------------------//
// a non-string environment option with a string identifier
template <typename Tp>
using EnvChoice = std::tuple<Tp, std::string, std::string>;

//--------------------------------------------------------------------------------------//
// list of environment choices with non-string and string identifiers
template <typename Tp>
using EnvChoiceList = std::set<EnvChoice<Tp>>;

//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting +
//  a default if not defined, e.g.
//      int num_threads =
//          GetEnv<int>("FORCENUMBEROFTHREADS",
//                          std::thread::hardware_concurrency());
//
template <typename Tp>
Tp
GetEnv(const std::string& env_id, Tp _default = Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        Tp                 var = Tp();
        iss >> var;
        return var;
    }

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for boolean
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
        return val;
    }

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for GetEnv + message when set
//
template <typename Tp>
Tp
GetEnv(const std::string& env_id, Tp _default, const std::string& msg)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        Tp                 var = Tp();
        iss >> var;
        std::cout << "Environment variable \"" << env_id << "\" enabled with "
                  << "value == " << var << ". " << msg << std::endl;
        return var;
    }

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting from set of choices
//
//      EnvChoiceList<int> choices =
//              { EnvChoice<int>(NN,     "NN",     "nearest neighbor interpolation"),
//                EnvChoice<int>(LINEAR, "LINEAR", "bilinear interpolation"),
//                EnvChoice<int>(CUBIC,  "CUBIC",  "bicubic interpolation") };
//
//      int eInterp = GetEnv<int>("INTERPOLATION", choices, CUBIC);
//
template <typename Tp>
Tp
GetEnv(const std::string& env_id, const EnvChoiceList<Tp>& _choices, Tp _default)
{
    auto asupper = [](std::string var) {
        for(auto& itr : var)
            itr = toupper(itr);
        return var;
    };

    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string str_var = std::string(env_var);
        std::string upp_var = asupper(str_var);
        Tp          var     = Tp();
        // check to see if string matches a choice
        for(const auto& itr : _choices)
        {
            if(asupper(std::get<1>(itr)) == upp_var)
            {
                return std::get<0>(itr);
            }
        }
        std::istringstream iss(str_var);
        iss >> var;
        // check to see if string matches a choice
        for(const auto& itr : _choices)
        {
            if(var == std::get<0>(itr))
            {
                return var;
            }
        }
        // the value set in env did not match any choices
        std::stringstream ss;
        ss << "\n### Environment setting error @ " << __FUNCTION__ << " (line "
           << __LINE__ << ")! Invalid selection for \"" << env_id
           << "\". Valid choices are:\n";
        for(const auto& itr : _choices)
            ss << "\t\"" << std::get<0>(itr) << "\" or \"" << std::get<1>(itr) << "\" ("
               << std::get<2>(itr) << ")\n";
        std::cerr << ss.str() << std::endl;
        abort();
    }

    std::string _name = "???";
    std::string _desc = "description not provided";
    for(const auto& itr : _choices)
        if(std::get<0>(itr) == _default)
        {
            _name = std::get<1>(itr);
            _desc = std::get<2>(itr);
            break;
        }

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
Tp
GetChoice(const EnvChoiceList<Tp>& _choices, const std::string& str_var)
{
    auto asupper = [](std::string var) {
        for(auto& itr : var)
            itr = toupper(itr);
        return var;
    };

    std::string upp_var = asupper(str_var);
    Tp          var     = Tp();
    // check to see if string matches a choice
    for(const auto& itr : _choices)
    {
        if(asupper(std::get<1>(itr)) == upp_var)
        {
            // record value defined by environment
            return std::get<0>(itr);
        }
    }
    std::istringstream iss(str_var);
    iss >> var;
    // check to see if string matches a choice
    for(const auto& itr : _choices)
    {
        if(var == std::get<0>(itr))
        {
            // record value defined by environment
            return var;
        }
    }
    // the value set in env did not match any choices
    std::stringstream ss;
    ss << "\n### Environment setting error @ " << __FUNCTION__ << " (line " << __LINE__
       << ")! Invalid selection \"" << str_var << "\". Valid choices are:\n";
    for(const auto& itr : _choices)
        ss << "\t\"" << std::get<0>(itr) << "\" or \"" << std::get<1>(itr) << "\" ("
           << std::get<2>(itr) << ")\n";
    std::cerr << ss.str() << std::endl;
    abort();
}

//--------------------------------------------------------------------------------------//
}  // namespace PTL
