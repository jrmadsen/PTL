#!/bin/bash

set -o errexit

: ${N:=0}
: ${CPUPROFILE:=""}
: ${CPUPROFBASE:=gperf.cpu.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=250}
: ${INTERACTIVE:=0}

while [ -z "${CPUPROFILE}" ]
do
    TEST_FILE=${CPUPROFBASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        CPUPROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export CPUPROFILE
export MALLOCSTATS
export CPUPROFILE_FREQUENCY

echo -e "\n\t--> Outputting profile to '${CPUPROFILE}'...\n"

eval $@ | tee ${CPU_PROFILE}.log

EXT=so
if [ "$(uname)" = "Darwin" ]; then EXT=dylib; fi
if [ -f "${CPUPROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof --text --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.txt
        pprof --text --cum --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE} | egrep -v ' 0x[0-9]' &> ${CPUPROFILE}.cum.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            pprof --add_lib=libptl.${EXT} ${PPROF_ARGS} ${1} ${CPUPROFILE}
        fi
    fi
fi
