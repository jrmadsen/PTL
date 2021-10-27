# - Clang-format only for master project
find_program(PTL_CLANG_FORMATTER
    NAMES
        clang-format-6
        clang-format-6.0
        clang-format-mp-6.0 # macports
        clang-format)
mark_as_advanced(PTL_CLANG_FORMATTER)

if(PTL_CLANG_FORMATTER)
    set(_Source_DIR     ${PROJECT_SOURCE_DIR}/source)
    set(_Example_DIR    ${PROJECT_SOURCE_DIR}/examples)

    file(GLOB_RECURSE headers
      ${_Source_DIR}/*.hh ${_Source_DIR}/*.icc
      ${_Example_DIR}/*.hh ${_Example_DIR}/*.h)

    file(GLOB_RECURSE sources
      ${_Source_DIR}/*.cc ${_Source_DIR}/*.c
      ${_Example_DIR}/*.cc ${_Example_DIR}/*.cu)

    # avoid conflicting format targets
    set(FORMAT_NAME format)
    if(TARGET format)
        set(FORMAT_NAME format-ptl)
    endif()

    add_custom_target(${FORMAT_NAME}
        COMMAND ${PTL_CLANG_FORMATTER} -i ${headers} ${sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running '${PTL_CLANG_FORMATTER}' on '${_Source_DIR}' and '${_Example_DIR}..."
        SOURCES ${headers} ${sources})
endif()
