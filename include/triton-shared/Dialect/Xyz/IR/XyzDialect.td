#ifndef TRITON_SHARED_XYZ_DIALECT
#define TRITON_SHARED_XYZ_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

def TTA_AddrTypePred : Type<CPred<"::mlir::isa<::mlir::tta::AddrType>($_self)">,
                            "tta.addr type">;
def TTA_IndexTensorLike : AnyTypeOf<[I32Tensor, I64Tensor]>;

def Xyz_Dialect : Dialect {
  let name = "xyz";
  let cppNamespace = "::mlir::xyz";

  let summary = "XYZ semantic operations for backend-friendly memory access";
  let description = [{
    The Xyz dialect carries semantic gather/scatter operations so backends can
    lower them to dedicated hardware instructions (for example, gather/scatter
    DMA) instead of scalarized loop forms.
  }];

  let dependentDialects = [
    "tta::TritonAddressDialect",
    "triton::TritonDialect"
  ];

  let usePropertiesForAttributes = 1;
}

class Xyz_Op<string mnemonic, list<Trait> traits = []> :
    Op<Xyz_Dialect, mnemonic, traits> {
}

def XYZ_GatherOp : Xyz_Op<"gather", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "semantic gather on a single indirect dimension";

  let arguments = (ins
    TTA_AddrTypePred:$address,
    TTA_IndexTensorLike:$indices,
    I32Attr:$gather_dim,
    Optional<TT_BoolLike>:$mask,
    Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other
  );

  let results = (outs TT_Tensor:$result);

  let hasVerifier = 1;
}

def XYZ_ScatterOp : Xyz_Op<"scatter", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "semantic scatter on a single indirect dimension";

  let arguments = (ins
    TTA_AddrTypePred:$address,
    TTA_IndexTensorLike:$indices,
    TT_Tensor:$value,
    I32Attr:$gather_dim,
    Optional<TT_BoolLike>:$mask
  );

  let hasVerifier = 1;
}

#endif
