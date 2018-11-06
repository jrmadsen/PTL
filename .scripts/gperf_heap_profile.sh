#!/bin/bash

set -o errexit

: ${N:=0}
: ${HEAPPROFILE:=""}
: ${HEAPPROFBASE:=gperf.heap.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}
: ${INTERACTIVE:=0}

while [ -z "${HEAPPROFILE}" ]
do
    TEST_FILE=${HEAPPROFBASE}.${N}
    if [ ! -f "${TEST_FILE}" ]; then
        HEAPPROFILE=${TEST_FILE}
    fi
    N=$((${N}+1))
done

export HEAPPROFILE
export MALLOCSTATS

echo -e "\n\t--> Outputting profile to '${HEAPPROFILE}'...\n"

eval $@ | tee ${HEAPPROFILE}.log

if [ -f "${HEAPPROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof --text ${PPROF_ARGS} ${1} ${HEAPPROFILE} &> ${HEAPPROFILE}.txt
        if [ "${INTERACTIVE}" -gt 0 ]; then
            pprof ${PPROF_ARGS} ${1} ${HEAPPROFILE}
        fi
    fi
fi
