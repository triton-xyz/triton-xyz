# use `debug_agent` as tmp dir
DIR="debug_agent/run_kernel_demo"
mkdir -p $DIR

export TRITON_HOME=$DIR
export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1

# check ir dumps in dir `triton_xyz_mlir_dump`
export MLIR_ENABLE_DUMP_DIR=$DIR/triton_xyz_mlir_dump

args=(
  # demo kernel
  python/tests/test_vec_add.py
  #
)
# check log in `compile.log`
python "${args[@]}" 2>&1 | tee $DIR/compile.log
