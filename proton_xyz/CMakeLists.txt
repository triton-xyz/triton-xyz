if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "JSON_INCLUDE_DIR is required to build proton_xyz")
endif()

set(PROTON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/triton/third_party/proton")
set(PROTON_XYZ_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/Proton.cpp
    ${PROTON_DIR}/common/lib/TraceDataIO/ByteSpan.cpp
    ${PROTON_DIR}/common/lib/TraceDataIO/CircularLayoutParser.cpp
    ${PROTON_DIR}/common/lib/TraceDataIO/EntryDecoder.cpp
    ${PROTON_DIR}/common/lib/TraceDataIO/Parser.cpp
    ${PROTON_DIR}/common/lib/TraceDataIO/TraceWriter.cpp
    ${PROTON_DIR}/csrc/lib/Context/Context.cpp
    ${PROTON_DIR}/csrc/lib/Context/Python.cpp
    ${PROTON_DIR}/csrc/lib/Context/Shadow.cpp
    ${PROTON_DIR}/csrc/lib/Data/Data.cpp
    ${PROTON_DIR}/csrc/lib/Data/Metric.cpp
    ${PROTON_DIR}/csrc/lib/Data/TraceData.cpp
    ${PROTON_DIR}/csrc/lib/Data/TreeData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/lib/Driver/Device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/lib/Profiler/Cpu/CpuProfiler.cpp
    ${PROTON_DIR}/csrc/lib/Profiler/Profiler.cpp
    ${PROTON_DIR}/csrc/lib/Session/Session.cpp
    ${PROTON_DIR}/csrc/lib/Utility/MsgPackWriter.cpp)

add_library(proton_xyz SHARED ${PROTON_XYZ_SOURCES})
target_link_libraries(proton_xyz PRIVATE Python3::Module pybind11::headers)
target_include_directories(proton_xyz PRIVATE
    ${JSON_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${PROTON_DIR}/common/include
    ${PROTON_DIR}/csrc/include)
set_target_properties(proton_xyz
    PROPERTIES OUTPUT_NAME proton
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
