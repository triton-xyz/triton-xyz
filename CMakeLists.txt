option(TRITON_XYZ_BUILD_PROTON "Build proton_xyz" ON)

if(TRITON_XYZ_BUILD_PROTON AND TRITON_BUILD_PROTON)
  message(FATAL_ERROR "TRITON_XYZ_BUILD_PROTON=ON requires -DTRITON_BUILD_PROTON=OFF.")
endif()

set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

if(TRITON_XYZ_BUILD_PROTON)
  add_compile_definitions(TRITON_XYZ_BUILD_PROTON=1)
endif()

add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_XYZ_BUILD_PROTON)
  add_subdirectory(proton_xyz)
endif()

add_triton_plugin(TritonXyz triton_xyz.cc
    LINK_LIBS TritonDialectXyz TritonSharedAnalysis)
target_link_libraries(TritonXyz PRIVATE Python3::Module pybind11::headers)
