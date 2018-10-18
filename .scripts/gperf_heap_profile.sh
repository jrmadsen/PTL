#!/bin/bash

set -o errexit

: ${N:=0}
: ${HEAPPROFILE:=""}
: ${HEAPPROFBASE:=gperf.heap.prof}
: ${PPROF_ARGS:=""}
: ${MALLOCSTATS:=1}

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

eval $@

if [ -f "${HEAPPROFILE}" ]; then
    : ${PPROF:=$(which pprof)}
    if [ -n "${PPROF}" ]; then
        pprof ${PPROF_ARGS} ${1} ${HEAPPROFILE} #| tee ${HEAPPROFILE}.log
    fi
fi
