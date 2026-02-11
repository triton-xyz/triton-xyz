args=(
  #
  # -m debugpy --listen 5678
  -m debugpy --listen 5678 --wait-for-client
  #
  python/tb/bench.py
  #
)
python "${args[@]}" 2>&1 | tee python/tb/debugpy.log
