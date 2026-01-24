import tempfile
import functools
import hashlib
from dataclasses import dataclass
from typing import Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes


@dataclass(frozen=True)
class CPUOptions:
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 1

    arch: str = None
    enable_fp_fusion: bool = True
    backend_name: str = "cpu"
    sanitize_overflow: bool = True

    debug: bool = False
    instrumentation_mode: str = ""
    allowed_dot_input_precisions: tuple[str] = ("ieee",)
    min_dot_size: int = 1

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class XYZBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def get_target_name(self, options) -> str:
        return "cpu"

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f"{version}-{self.target.arch}"

    def parse_options(self, options):
        args = {}
        args.update(
            {
                k: options[k]
                for k in CPUOptions.__dataclass_fields__.keys()
                if k in options
                if options[k] is not None
            }
        )

        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        # TODO
        return {"triton.language.extra.libdevice": None}

    def load_dialects(self, ctx):
        # TODO
        pass

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_linalg(mod, metadata, options):
        # TODO
        return mod

    def add_stages(self, stages, options, language):
        if language == Language.GLUON:
            raise Exception("GLUON is not supported")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["mlir"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        # stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        # stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        # stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)
