###############################################################################

rm -rf build/CMakeFiles
rm -rf build/CMakeCache.txt
[[ "$(uname)" == "Darwin" ]] && PRESET="osx_lld" || PRESET="osx"
cmake --preset $PRESET -S$PWD/third_party/triton -B$PWD/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
  -DTRITON_PLUGIN_DIRS=$PWD \
  -DCMAKE_INSTALL_PREFIX=$PWD/build/install \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$PWD/build/bin \
  -DTRITON_WHEEL_DIR=$PWD/build/bin \
  -DLLVM_SYSPATH=$PWD/llvm-triton/llvm-project/build \
  -DLLVM_DIR=$PWD/llvm-triton/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$PWD/llvm-triton/llvm-project/build/lib/cmake/mlir \
  -DLLD_DIR=$PWD/llvm-triton/llvm-project/build/lib/cmake/lld \
  -DCMAKE_C_COMPILER=$(which clang) \
  -DCMAKE_CXX_COMPILER=$(which clang++) \
  -DPython3_EXECUTABLE=$(which python)
cmake --build $PWD/build --target all

###############################################################################
