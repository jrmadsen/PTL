################################################################################
#
#        Creates a 'format' target that runs clang-format
#
################################################################################

find_program(CLANG_FORMATTER
    NAMES
        clang-format-8.0
        clang-format-7.0
        clang-format-6.0
        clang-format)

if(CLANG_FORMATTER)
    set(_Source_DIR     ${PROJECT_SOURCE_DIR}/source)
    set(_Example_DIR    ${PROJECT_SOURCE_DIR}/examples)

    set(_Header_DIR     ${_Source_DIR}/source/PTL)
    set(_Basic_DIR      ${_Example_DIR}/basic)
    set(_Common_DIR     ${_Example_DIR}/common)
    set(_Gpu_DIR        ${_Example_DIR}/gpu)

    file(GLOB headers
        ${_Header_DIR}/*.hh ${_Header_DIR}/*.icc
        ${_Basic_DIR}/*.hh  ${_Common_DIR}/*.hh
        ${_Gpu_DIR}/*.h     ${_Gpu_DIR}/*.hh)

    file(GLOB sources
        ${_Source_DIR}/*.cc
        ${_Basic_DIR}/*.cc  ${_Common_DIR}/*.cc
        ${_Gpu_DIR}/*.cc    ${_Gpu_DIR}/*.cu)

    add_custom_target(format
        COMMAND ${CLANG_FORMATTER} -i ${headers} ${sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${CLANG_FORMATTER}' on '${_Source_DIR}' and '${_Example_DIR}..."
        SOURCES ${headers} ${sources})

endif()
