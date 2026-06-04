###############################################################################

if [[ ! -e "$PWD/third_party/triton/patch.patch" ]]; then
  ln -s $PWD/third_party/triton_patch/CMakePresets.json $PWD/third_party/triton/
  ln -s $PWD/third_party/triton_patch/patch.patch $PWD/third_party/triton/

  pushd $PWD/third_party/triton
  git apply patch.patch
  popd
fi

rm -rf build/CMakeFiles
rm -rf build/CMakeCache.txt
[[ "$(uname)" == "Darwin" ]] && PRESET="osx_lld" || PRESET="osx"
cmake --preset $PRESET -S$PWD/third_party/triton -B$PWD/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=$(which clang) \
  -DCMAKE_CXX_COMPILER=$(which clang++) \
  -DPython3_EXECUTABLE=$(which python)
cmake --build $PWD/build --target all

###############################################################################

uv pip uninstall --system triton
export TRITON_PLUGIN_DIRS=$PWD
pushd third_party/triton
uv pip install --system --no-build-isolation -e . -v
popd

###############################################################################

mkdir -p $PWD/third_party/triton/python/triton/_C
ln -snf $PWD/build/libtriton.so \
  $PWD/third_party/triton/python/triton/_C/libtriton.so
ln -snf $PWD/build/libproton.so \
  $PWD/third_party/triton/python/triton/_C/libproton.so

###############################################################################
