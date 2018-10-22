#!/bin/bash

set -o errexit

: ${N:=0}
: ${CPUPROFILE:=""}
: ${CPUPROFBASE:=gperf.cpu.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${CPUPROFILE_FREQUENCY:=250}

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
#export CPUPROFILE_FREQUENCY

echo -e "\n\t--> Outputting profile to '${CPUPROFILE}'...\n"

eval $@ | tee ${CPU_PROFILE}.log

if [ -f "${CPUPROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof --text ${PPROF_ARGS} ${1} ${CPUPROFILE} &> ${CPUPROFILE}.txt
        pprof --text --cum ${PPROF_ARGS} ${1} ${CPUPROFILE} &> ${CPUPROFILE}.cum.txt
        pprof ${PPROF_ARGS} ${1} ${CPUPROFILE}
    fi
fi
