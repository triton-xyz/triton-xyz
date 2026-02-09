#ifndef TRITON_ADDRESS_TYPES
#define TRITON_ADDRESS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "triton-shared/Dialect/TritonAddress/IR/TritonAddressDialect.td"

class TritonAddressTypeDef<string name, string _mnemonic,
                           list<Trait> traits = []>
    : TypeDef<Triton_Address_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def TTA_AddrType : TritonAddressTypeDef<"Addr", "addr"> {
  let summary = "Unified address descriptor type";

  let description = [{
    Unified address type used by the Triton Address dialect to represent
    abstract addresses independent of legacy pointer encodings.
  }];

  let parameters = (ins "Type":$elementType,
                        "int64_t":$rank,
                        "int":$addressSpace);

  let assemblyFormat = "`<` $elementType `,` $rank `,` $addressSpace `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$elementType,
      "int64_t":$rank,
      "int":$addressSpace), [{
      return $_get(elementType.getContext(), elementType, rank, addressSpace);
    }]>
  ];

  let genVerifyDecl = 1;
}

def TTA_Addr : DialectType<
    Triton_Address_Dialect,
    CPred<"::mlir::isa<::mlir::tta::AddrType>($_self)">,
    "tta address type",
    "::mlir::tta::AddrType">;

#endif
