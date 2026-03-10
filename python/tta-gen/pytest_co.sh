PPWD=$(pwd)
DIR=$PPWD/debug/tmp
mkdir -p $DIR

args=(
  #
  # --disable-warnings
  -W ignore::pytest.PytestUnknownMarkWarning
  #
  --collect-only
  #
  third_party/triton-ascend/third_party/ascend/unittest/generalization_cases
  #
)
pytest "${args[@]}" 2>&1 | tee $DIR/pytest.co.log
