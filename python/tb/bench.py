import shlex

import _wrap_api  # noqa

from tritonbench.utils.parser import get_parser
from tritonbench.operators import load_opbench_by_name


# args_sh = "--op vector_add --only torch_add"
args_sh = "--op vector_add --only triton_ --only-match-mode prefix-with-baseline"
# args_sh = "--op layer_norm --only triton_fused_layer_norm --only-match-mode exact"
# args_sh = "--op layer_norm --only triton_ --only-match-mode prefix-with-baseline"

args_sh += " -d cpu --mode fwd_no_grad --dtype fp32"

args = shlex.split(args_sh)
parser = get_parser()
op_args = parser.parse_args(args)

addmm_bench = load_opbench_by_name(op_args.op)(op_args)

addmm_bench.run()
