PPWD=$(pwd)
DIR=$PPWD/debug/tmp-pytest_one
mkdir -p $DIR

export PATH="$PWD/build/bin:$PATH"
export TRITON_ALWAYS_COMPILE=1
export TRITON_HOME="$DIR"

export TRITON_DEBUG=1
export MLIR_ENABLE_DUMP=1
export MLIR_ENABLE_DUMP_DIR="$DIR/_pass_dump"

export TTX_PYTEST_QUIET=0
export TTX_PYTEST_GLOBAL_TIMEOUT=100
export TTX_PYTEST_DTYPE="float32"

args=(
  #
  # --disable-warnings
  -W ignore::pytest.PytestUnknownMarkWarning
  #
  -v
  # -q -r fE
  #
  third_party/ascend/unittest/pytest_ut/test_gamma.py
  -k "test_gamma_case[param_list0]"
  #
)
pushd third_party/triton-ascend
pytest "${args[@]}" 2>&1 | grep -v SKIPPED | tee $DIR/pytest_one.log
# pytest "${args[@]}" 2>&1 | tee $DIR/pytest_one.log
popd
