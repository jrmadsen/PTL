#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for Parallel Tasking Library (PTL)
"""

import os
import sys
import platform
import traceback
import warnings
import multiprocessing as mp
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers


# --------------------------------------------------------------------------- #
def configure():

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(
        project_name="PTL",
        source_dir=os.getcwd(),
        binary_dir=os.path.join(os.getcwd(), "build-PTL"),
        build_type="Release",
        vcs_type="git",
    )

    parser.add_argument(
        "--arch", help="PTL_USE_ARCH=ON", default=False, action="store_true"
    )
    parser.add_argument(
        "--tbb", help="PTL_USE_TBB=ON", default=False, action="store_true"
    )
    parser.add_argument(
        "--sanitizer",
        help="PTL_USE_SANITIZER=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--static-analysis",
        help="PTL_USE_CLANG_TIDY=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--coverage",
        help="PTL_USE_COVERAGE=ON",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--num-tasks", help="Set the number of tasks", default=65536, type=int
    )

    args = parser.parse_args()

    if os.path.exists(
        os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt")
    ):
        from pyctest import cmake_executable as cm
        from pyctest import version_info as _pyctest_version

        if (
            _pyctest_version[0] == 0
            and _pyctest_version[1] == 0
            and _pyctest_version[2] < 11
        ):
            cmd = pyctest.command(
                [cm, "--build", pyctest.BINARY_DIRECTORY, "--target", "clean"]
            )
            cmd.SetWorkingDirectory(pyctest.BINARY_DIRECTORY)
            cmd.SetOutputQuiet(True)
            cmd.SetErrorQuiet(True)
            cmd.Execute()
        else:
            from pyctest.cmake import CMake

            CMake("--build", pyctest.BINARY_DIRECTORY, "--target", "clean")
        helpers.RemovePath(
            os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt")
        )

    return args


# --------------------------------------------------------------------------- #
#
def run_pyctest():

    # ----------------------------------------------------------------------- #
    # run argparse, checkout source, copy over files
    #
    args = configure()

    # ----------------------------------------------------------------------- #
    # Compiler version
    #
    if os.environ.get("CXX") is None:
        os.environ["CXX"] = helpers.FindExePath("c++")
    cmd = pyctest.command([os.environ["CXX"], "-dumpversion"])
    cmd.SetOutputStripTrailingWhitespace(True)
    cmd.Execute()
    compiler_version = cmd.Output()

    # ----------------------------------------------------------------------- #
    # Set the build name
    #
    pyctest.BUILD_NAME = "[{}] [{} {} {}] [{} {}]".format(
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4],
        os.path.basename(os.path.realpath(os.environ["CXX"])),
        compiler_version,
    )

    # ----------------------------------------------------------------------- #
    #   build specifications
    #
    build_opts = {
        "PTL_USE_ARCH": "OFF",
        "PTL_USE_TBB": "OFF",
        "PTL_USE_SANITIZER": "OFF",
        "PTL_USE_CLANG_TIDY": "OFF",
        "PTL_USE_COVERAGE": "OFF",
    }

    if args.tbb:
        pyctest.BUILD_NAME = "{} [tbb]".format(pyctest.BUILD_NAME)
        build_opts["PTL_USE_TBB"] = "ON"
    if args.arch:
        pyctest.BUILD_NAME = "{} [arch]".format(pyctest.BUILD_NAME)
        build_opts["PTL_USE_ARCH"] = "ON"
    if args.sanitizer:
        pyctest.BUILD_NAME = "{} [asan]".format(pyctest.BUILD_NAME)
        build_opts["PTL_USE_SANITIZER"] = "ON"
    if args.static_analysis:
        build_opts["PTL_USE_CLANG_TIDY"] = "ON"
    if args.coverage:
        gcov_exe = helpers.FindExePath("gcov")
        if gcov_exe is not None:
            pyctest.COVERAGE_COMMAND = "{}".format(gcov_exe)
            build_opts["PTL_USE_COVERAGE"] = "ON"
            warnings.warn(
                "Forcing build type to 'Debug' when coverage is enabled"
            )
            pyctest.BUILD_TYPE = "Debug"

    # default options
    cmake_args = "-DCMAKE_BUILD_TYPE={} -DPTL_BUILD_EXAMPLES=ON".format(
        pyctest.BUILD_TYPE
    )

    # customized from args
    for key, val in build_opts.items():
        cmake_args = "{} -D{}={}".format(cmake_args, key, val)

    # ----------------------------------------------------------------------- #
    # how to build the code
    #
    ctest_cmake_cmd = "${CTEST_CMAKE_COMMAND}"
    pyctest.CONFIGURE_COMMAND = "{} {} {}".format(
        ctest_cmake_cmd, cmake_args, pyctest.SOURCE_DIRECTORY
    )

    # ----------------------------------------------------------------------- #
    # how to build the code
    #
    pyctest.BUILD_COMMAND = "{} --build {} --target all".format(
        ctest_cmake_cmd, pyctest.BINARY_DIRECTORY
    )

    # ----------------------------------------------------------------------- #
    # parallel build
    #
    if not args.static_analysis:
        if platform.system() != "Windows":
            pyctest.BUILD_COMMAND = "{} -- -j{} VERBOSE=1".format(
                pyctest.BUILD_COMMAND, mp.cpu_count()
            )
        else:
            pyctest.BUILD_COMMAND = "{} -- /MP -A x64".format(
                pyctest.BUILD_COMMAND
            )

    # ----------------------------------------------------------------------- #
    # how to update the code
    #
    git_exe = helpers.FindExePath("git")
    pyctest.UPDATE_COMMAND = "{}".format(git_exe)
    pyctest.set("CTEST_UPDATE_TYPE", "git")
    pyctest.set("CTEST_GIT_COMMAND", "{}".format(git_exe))

    # ----------------------------------------------------------------------- #
    # static analysis
    #
    clang_tidy_exe = helpers.FindExePath("clang-tidy")
    if clang_tidy_exe:
        pyctest.set(
            "CMAKE_CXX_CLANG_TIDY", "{};-checks=*".format(clang_tidy_exe)
        )

    # ----------------------------------------------------------------------- #
    # find the CTEST_TOKEN_FILE
    #
    if args.pyctest_token_file is None and args.pyctest_token is None:
        home = helpers.GetHomePath()
        if home is not None:
            token_path = os.path.join(
                home, os.path.join(".tokens", "nersc-cdash")
            )
            if os.path.exists(token_path):
                pyctest.set("CTEST_TOKEN_FILE", token_path)

    # ----------------------------------------------------------------------- #
    # construct a command
    #
    def construct_command(cmd, args):
        _cmd = []
        _cmd.extend(cmd)
        return _cmd

    # ----------------------------------------------------------------------- #
    # standard environment settings for tests, adds profile to notes
    #
    def test_env_settings(prof_fname, clobber=False, extra=""):
        return "PTL_NUM_THREADS={};CPUPROFILE={};{}".format(
            mp.cpu_count(), prof_fname, extra
        )

    # pyctest.set("ENV{GCOV_PREFIX}", pyctest.BINARY_DIRECTORY)
    # pyctest.set("ENV{GCOV_PREFIX_STRIP}", "4")

    # ----------------------------------------------------------------------- #
    # create tests
    #
    tasking_suffix = ""
    if args.num_tasks != 65536:
        tasking_suffix = "_{}".format(args.num_tasks)
    test = pyctest.test()
    test.SetName("tasking{}".format(tasking_suffix))
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty(
        "ENVIRONMENT",
        test_env_settings(
            "cpu-prof-tasking",
            clobber=True,
            extra="NUM_TASKS={}".format(args.num_tasks),
        ),
    )
    test.SetProperty("RUN_SERIAL", "ON")
    test.SetCommand(construct_command(["./tasking"], args))

    test = pyctest.test()
    test.SetName("recursive_tasking")
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty(
        "ENVIRONMENT", test_env_settings("cpu-prof-recursive-tasking")
    )
    test.SetProperty("RUN_SERIAL", "ON")
    test.SetCommand(construct_command(["./recursive_tasking"], args))

    if args.tbb:
        test = pyctest.test()
        test.SetName("tbb_tasking{}".format(tasking_suffix))
        test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
        test.SetProperty(
            "ENVIRONMENT",
            test_env_settings(
                "cpu-prof-tbb-tasking",
                extra="NUM_TASKS={}".format(args.num_tasks),
            ),
        )
        test.SetProperty("RUN_SERIAL", "ON")
        test.SetCommand(construct_command(["./tbb_tasking"], args))

        test = pyctest.test()
        test.SetName("recursive_tbb_tasking")
        test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
        test.SetProperty(
            "ENVIRONMENT", test_env_settings("cpu-prof-tbb-recursive-tasking")
        )
        test.SetProperty("RUN_SERIAL", "ON")
        test.SetCommand(construct_command(["./recursive_tbb_tasking"], args))

    pyctest.generate_config(pyctest.BINARY_DIRECTORY)
    pyctest.generate_test_file(pyctest.BINARY_DIRECTORY)
    pyctest.run(pyctest.ARGUMENTS, pyctest.BINARY_DIRECTORY)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    try:

        run_pyctest()

    except Exception as e:
        print("Error running pyctest - {}".format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
