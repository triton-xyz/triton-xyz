PPWD=$(pwd)
DIR=$PPWD/debug/tmp
mkdir -p $DIR

export PATH="$PWD/build/bin:$PATH"
export TRITON_ALWAYS_COMPILE=1
export TRITON_HOME="$DIR"

export TTX_PYTEST_QUIET=0
export TTX_PYTEST_GLOBAL_TIMEOUT=100

# export TTX_PYTEST_DTYPE="float32"
# export TTX_PYTEST_DTYPE="float16"
# export TTX_PYTEST_DTYPE="int32"

args=(
  #
  # --disable-warnings
  -W ignore::pytest.PytestUnknownMarkWarning
  #
  -v
  # -q -r fE
  #
  # -p no:timeout
  #
  # third_party/ascend/unittest/pytest_ut
  #
  third_party/ascend/unittest/pytest_ut
  -k "test_abs.py"
  #
)
pushd third_party/triton-ascend
pytest "${args[@]}" 2>&1 | grep -v SKIPPED | tee $DIR/pytest.log
# pytest "${args[@]}" 2>&1 | tee $DIR/pytest.log
popd
