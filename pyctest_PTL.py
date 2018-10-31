#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for Parallel Tasking Library (PTL)
"""

import os
import sys
import shutil
import platform
import argparse
import traceback
import warnings
import multiprocessing as mp

import pyctest.pyctest as pyctest
import pyctest.pycmake as pycmake
import pyctest.helpers as helpers


#------------------------------------------------------------------------------#
def configure():

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="PTL",
                                    source_dir=os.getcwd(),
                                    binary_dir=os.path.join(os.getcwd(), "build-PTL"),
                                    python_exe=sys.executable,
                                    submit=False)

    build_choices = [ "Release", "RelWithDebInfo", "Debug", "MinSizeRel" ]
    parser.add_argument("--cc", help="C compiler",
                        default=os.environ.get("CC"), type=str)
    parser.add_argument("--cxx", help="C++ compiler",
                        default=os.environ.get("CXX"), type=str)
    parser.add_argument("--build-type", help="C++ compiler",
                        default="Release", type=str,
                        choices=build_choices)
    parser.add_argument("--arch", help="PTL_USE_ARCH=ON",
                        default=False, action='store_true')
    parser.add_argument("--gperf", help="PTL_USE_GPERF=ON",
                        default=False, action='store_true')
    parser.add_argument("--tbb", help="PTL_USE_TBB=ON",
                        default=False, action='store_true')

    args = parser.parse_args()

    if args.cc is not None:
        os.environ["CC"] = "{}".format(args.cc)

    if args.cxx is not None:
        os.environ["CXX"] = "{}".format(args.cxx)

    if os.path.exists(os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt")):
        cm = helpers.FindExePath("cmake")
        cmd = pyctest.command([cm, "--build", pyctest.BINARY_DIRECTORY, "--target", "clean"])
        cmd.SetWorkingDirectory(pyctest.BINARY_DIRECTORY)
        cmd.SetOutputQuiet(True)
        cmd.SetErrorQuiet(True)
        cmd.Execute()
        for f in [ "CMakeCache.txt", "CMakeFiles" ]:
            helpers.RemovePath(os.path.join(pyctest.BINARY_DIRECTORY, f))

    if args.gperf:
        pyctest.copy_files(["gperf_cpu_profile.sh", "gperf_heap_profile.sh"],
            os.path.join(pyctest.SOURCE_DIRECTORY, ".scripts"),
            pyctest.BINARY_DIRECTORY)
        if args.build_type == "Release":
            warnings.warn("Changing build type to 'RelWithDebInfo' when GPerf is enabled")
            args.build_type = "RelWithDebInfo"

    return args


#------------------------------------------------------------------------------#
#
def run_pyctest():

    #--------------------------------------------------------------------------#
    # run argparse, checkout source, copy over files
    #
    args = configure()

    #--------------------------------------------------------------------------#
    # Set the build name
    #
    pyctest.BUILD_NAME = "[{}] [{} {} {}]".format(
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4])

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyctest.CONFIGURE_COMMAND = "${} -DPTL_USE_ARCH={} -DPTL_USE_GPERF={} -DPTL_USE_TBB={} -DCMAKE_BUILD_TYPE={} -DPTL_BUILD_EXAMPLES=ON {}".format(
        "{CTEST_CMAKE_COMMAND}", "ON" if args.arch else "OFF",
        "ON" if args.gperf else "OFF", "ON" if args.tbb else "OFF",
        args.build_type, pyctest.SOURCE_DIRECTORY)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyctest.BUILD_COMMAND = "${} --build {} --target all".format(
        "{CTEST_CMAKE_COMMAND}", pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # find the CTEST_TOKEN_FILE
    #
    if args.pyctest_token_file is None and args.pyctest_token is None:
        home = helpers.GetHomePath()
        if home is not None:
            token_path = os.path.join(
                home, os.path.join(".tokens", "nersc-cdash"))
            if os.path.exists(token_path):
                pyctest.set("CTEST_TOKEN_FILE", token_path)

    #--------------------------------------------------------------------------#
    # construct a command
    #
    def construct_command(cmd, args):
        _cmd = []
        if args.gperf:
            _cmd.append(os.path.join(pyctest.BINARY_DIRECTORY,
                                     "gperf_cpu_profile.sh"))
        _cmd.extend(cmd)
        return _cmd

    #--------------------------------------------------------------------------#
    # create a CTest that wraps "nosetest"
    #
    test = pyctest.test()
    test.SetName("tasking")
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT",
                     "PTL_NUM_THREADS={};CPUPROFILE={}".format(
                         mp.cpu_count(), "cpu-prof-tasking"))
    test.SetCommand(construct_command(["./tasking"], args))

    test = pyctest.test()
    test.SetName("recursive_tasking")
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT",
                     "PTL_NUM_THREADS={};CPUPROFILE={}".format(
                         mp.cpu_count(), "cpu-prof-recursive-tasking"))
    test.SetCommand(construct_command(["./recursive_tasking"], args))

    if args.gperf:
        clobber = True
        for f in [ "tasking", "recursive-tasking",
                   "tasking.cum", "recursive-tasking.cum" ]:
            pyctest.add_note(pyctest.BINARY_DIRECTORY,
                         "cpu-prof-{}.txt".format(f),
                         clobber=clobber)
            clobber = False

    if args.tbb:
        test = pyctest.test()
        test.SetName("tbb_tasking")
        test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
        test.SetProperty("ENVIRONMENT",
            "PTL_NUM_THREADS={};CPUPROFILE={}".format(
                mp.cpu_count(), "cpu-prof-tbb-tasking"))
        test.SetCommand(construct_command(["./tbb_tasking"], args))

        test = pyctest.test()
        test.SetName("recursive_tbb_tasking")
        test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
        test.SetProperty("ENVIRONMENT",
                         "PTL_NUM_THREADS={};CPUPROFILE={}".format(
                             mp.cpu_count(), "cpu-prof-recursive-tbb-tasking"))
        test.SetCommand(construct_command(["./recursive_tbb_tasking"], args))

        if args.gperf:
            for f in ["tbb-tasking", "recursive-tbb-tasking",
                      "tbb-tasking.cum", "recursive-tbb-tasking.cum"]:
                pyctest.add_note(pyctest.BINARY_DIRECTORY,
                                "cpu-prof-{}.txt".format(f))

    pyctest.generate_config(pyctest.BINARY_DIRECTORY)
    pyctest.generate_test_file(pyctest.BINARY_DIRECTORY)
    pyctest.run(pyctest.ARGUMENTS, pyctest.BINARY_DIRECTORY)


#------------------------------------------------------------------------------#
if __name__ == "__main__":

    try:

        run_pyctest()

    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
