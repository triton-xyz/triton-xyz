#ifndef TRITON_ADDRESS_DIALECT
#define TRITON_ADDRESS_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

def Triton_Address_Dialect : Dialect {
  let name = "tta";

  let cppNamespace = "::mlir::tta";

  let summary = "Unified Triton address operations";

  let description = [{
    Triton Address Dialect.

    This dialect is intended to become the single address abstraction layer for
    both structured and unstructured pointer semantics.
  }];

  let dependentDialects = [
    "triton::TritonDialect"
  ];

  let usePropertiesForAttributes = 1;
}

class TTA_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Address_Dialect, mnemonic, traits> {
}

def TTA_IndexTensorLike : AnyTypeOf<[I32Tensor, I64Tensor]>;

def TTA_MakeAddrOp
  : TTA_Op<"make_addr", [AttrSizedOperandSegments, Pure]> {
  let summary = "create a unified address descriptor";

  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_shape,
                       DenseI32ArrayAttr:$order);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    `` `,` `order` `` `:` $order
    attr-dict `:` type($base) `to` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      "ArrayRef<int32_t>":$order)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes;
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    bool isBlockPtr() {
      return !getOrder().empty();
    }
    bool isStructuredPtr() {
      return !isBlockPtr() &&
             llvm::all_of(getStaticShape(), [](auto shape) { return shape == 0; });
    }
    bool isSplitPtr() {
      return !isBlockPtr() &&
             !isStructuredPtr();
    }
  }];
}

def TTA_ReindexOp : TTA_Op<"reindex", [AttrSizedOperandSegments, Pure]> {
  let summary = "reindex an address descriptor with affine or indirect indexing";

  let arguments = (ins TT_PtrLike:$address,
                       Optional<TTA_IndexTensorLike>:$indirect_index,
                       OptionalAttr<I32Attr>:$indirect_dim,
                       Variadic<Index>:$offsets,
                       DenseI64ArrayAttr:$static_offsets,
                       Optional<TT_BoolLike>:$mask);

  let results = (outs TT_PtrLike:$result);

  let builders = [
    OpBuilder<(ins "Value":$address,
                   "ArrayRef<OpFoldResult>":$offsets)>,
    OpBuilder<(ins "Value":$address,
                   "Value":$indirectIndex,
                   "int":$indirectDim,
                   "ArrayRef<OpFoldResult>":$offsets)>,
    OpBuilder<(ins "Value":$address,
                   "Value":$indirectIndex,
                   "Value":$mask,
                   "int":$indirectDim,
                   "ArrayRef<OpFoldResult>":$offsets)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
  }];

  let hasVerifier = 1;
}

def TTA_AdvanceOp : TTA_Op<"advance", [Pure]> {
  let summary = "advance an address descriptor by per-dimension deltas";

  let arguments = (ins TT_PtrLike:$address,
                       Variadic<Index>:$deltas,
                       DenseI64ArrayAttr:$static_deltas);

  let results = (outs TT_PtrLike:$result);

  let builders = [
    OpBuilder<(ins "Value":$address,
                   "ArrayRef<OpFoldResult>":$deltas)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedDeltas() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticDeltas(), getDeltas(), b);
    }
  }];

  let hasVerifier = 1;
}

def TTA_AtomicOp : TTA_Op<"atomic", [
  MemoryEffects<[MemRead, MemWrite]>,
  OptionalTypesMatchWith<"mask type matches offset type", "offset", "mask",
                         "triton::getI1SameShape($_self)">,
  TypesMatchWith<"result type matches value type", "value", "result",
                 "$_self">
]> {
  let summary = "perform an atomic memory operation";

  let arguments = (
    ins
    StrAttr:$kind,
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask
  );

  let results = (outs TT_Type:$result);

  let builders = [
    OpBuilder<(ins "StringRef":$kind,
                   "Value":$ptr,
                   "Value":$offset,
                   "Value":$value)>,
    OpBuilder<(ins "StringRef":$kind,
                   "Value":$ptr,
                   "Value":$offset,
                   "Value":$value,
                   "Value":$mask)>,
  ];

  let hasVerifier = 1;
}

def TTA_LoadOp : TTA_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "optionally load data from memory to fill a tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr,
                   "ArrayRef<OpFoldResult>":$mask_dims,
                   "Value":$other)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];
}

def TTA_StoreOp : TTA_Op<"store", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "optionally store data from tensor to memory";

  let arguments = (ins TT_PtrLike:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims);

  let builders = [
    OpBuilder<(ins "Value":$ptr,
                   "Value":$value,
                   "ArrayRef<OpFoldResult>":$dims)>,
  ];

  let extraClassDeclaration = [{
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];
}

#endif
