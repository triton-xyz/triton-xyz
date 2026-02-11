args=(
  #
  python/tb/bench.py
  #
)
python "${args[@]}" 2>&1 | tee python/tb/run.log
