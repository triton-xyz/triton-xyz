import shlex

from tritonbench.utils.parser import get_parser
from tritonbench.operators import load_opbench_by_name

args = "-d cpu --mode fwd_no_grad --dtype fp32"

parser = get_parser()
op_args = parser.parse_args(shlex.split(args))

addmm_bench = load_opbench_by_name("vector_add")(op_args)

# TODO: fix err
addmm_bench.run()
