ln -s $PWD/third_party/triton_patch/CMakePresets.json $PWD/third_party/triton/
ln -s $PWD/third_party/triton_patch/patch.patch $PWD/third_party/triton/

pushd $PWD/third_party/triton
git apply patch.patch
popd

[[ "$(uname)" == "Darwin" ]] && PRESET="osx_lld" || PRESET="osx"
cmake --preset $PRESET -S$PWD/third_party/triton -B$PWD/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=$(which clang) \
  -DCMAKE_CXX_COMPILER=$(which clang++) \
  -DPython3_EXECUTABLE=$(which python)
cmake --build $PWD/build --target all
