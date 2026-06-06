#ifndef TRITONXYZOPS_IR
#define TRITONXYZOPS_IR

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TTXyz_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])>;

def TT_NopOp : TTXyz_Op<"nop", [Pure, SameOperandsAndResultType]> {
  let summary = "Return a tensor unchanged";

  let description = [{
    Accept a single tensor operand and return it unchanged.
  }];

  let arguments = (ins TT_Tensor:$input);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // TRITONXYZOPS_IR
