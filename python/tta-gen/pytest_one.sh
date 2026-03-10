PPWD=$(pwd)
DIR=$PPWD/debug/tmp
mkdir -p $DIR

export PATH="$PWD/build/bin:$PATH"
export TRITON_ALWAYS_COMPILE=1
export TRITON_HOME="$DIR"

export TTX_PYTEST_QUIET=1
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
  third_party/ascend/unittest/generalization_cases/test_abs.py
  -k "test_abs.py"
  #
  # tt.gater
  # third_party/ascend/unittest/generalization_cases/test_general_gather.py
  # -k "test_gather_4d_5d[src_shape0-indices_shape0-0]"
  #
  # api issue
  # third_party/ascend/unittest/generalization_cases/test_broadcast.py
  # -k "test_broadcast_alltype[float32]"
  # third_party/ascend/unittest/generalization_cases/test_general_split.py
  # -k "test_split[float32-shape4]"
  #
)
pushd third_party/triton-ascend
pytest "${args[@]}" 2>&1 | grep -v SKIPPED | tee $DIR/pytest.log
# pytest "${args[@]}" 2>&1 | tee $DIR/pytest.log
popd
