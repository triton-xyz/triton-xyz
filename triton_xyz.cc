#include "ir.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "triton-shared/Dialect/Triton/IR/TritonDialectXyz.h"

#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

void init_triton_xyz(py::module &&m) {
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::triton::xyz::registerTritonDialectXyzExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def(
      "create_nop",
      [](TritonOpBuilder &builder, mlir::Value input) -> mlir::Value {
        auto inputType =
            mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
        if (!inputType)
          throw std::invalid_argument(
              "xyz.create_nop expects a ranked tensor input");

        return builder.create<mlir::triton::NopOp>(input.getType(), input);
      },
      py::arg("builder"), py::arg("input"));
}
