import pytest
import torch
import torch_npu  # noqa: F401

import numbers
import os
import re


_DTYPE_ALIAS = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
}
_DTYPE_PATTERN = re.compile(r"^(u?int\d+|float\d+|bfloat16|bf16|fp\d+|bool)([a-z0-9_]*)$")
_INVALID_TEST_PATTERN = re.compile(r"^test_invalid_[a-zA-Z0-9_]*$")
_INVALID_DTYPE_TEST_PATTERN = re.compile(r"^test_[a-zA-Z0-9_]*_invalid_dtype_[a-zA-Z0-9_]*$")


def _normalize_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        if value == torch.bool:
            return "bool"
        if value == torch.int8:
            return "int8"
        if value == torch.int16:
            return "int16"
        if value == torch.int32:
            return "int32"
        if value == torch.int64:
            return "int64"
        if value == torch.uint8:
            return "uint8"
        if hasattr(torch, "uint16") and value == torch.uint16:
            return "uint16"
        if hasattr(torch, "uint32") and value == torch.uint32:
            return "uint32"
        if hasattr(torch, "uint64") and value == torch.uint64:
            return "uint64"
        if value == torch.float16:
            return "float16"
        if value == torch.float32:
            return "float32"
        if hasattr(torch, "float64") and value == torch.float64:
            return "float64"
        if value == torch.bfloat16:
            return "bfloat16"
        return None
    if isinstance(value, str):
        norm = _DTYPE_ALIAS.get(value.strip().lower(), value.strip().lower())
        return norm if _DTYPE_PATTERN.match(norm) else None
    if hasattr(value, "name") and isinstance(value.name, str):
        norm = _DTYPE_ALIAS.get(value.name.strip().lower(), value.name.strip().lower())
        return norm if _DTYPE_PATTERN.match(norm) else None
    return None


def _normalize_shape(value):
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        if all(isinstance(x, numbers.Integral) for x in value):
            return tuple(int(x) for x in value)
    return None


_DTYPE_PARAM_NAMES = {
    "sigtype",
    "data_type",
    "para_type",
    "normal_type",
}

SKIP_TEST_FILES = {
    #
    # `triton.language.extra.cann.libdevice`
    "test_pow.py",
    #
    # `triton.language.math`
    "test_multi_return.py",
    #
    # `triton.language.extra.cann.extension`
    "test_cat_help_func.py",
    "test_gather.py",
    "test_sub_vec_num.py",
    "test_ascend_barrier.py",
    "test_max_constancy.py",
    "test_extract_slice.py",
    "test_sync_block_all.py",
    "test_parallel.py",
    "test_bind_buffer.py",
    "test_insert_slice.py",
    "test_sub_vec_id.py",
    "test_compile_hint.py",
    "test_index_select.py",
    "test_to_tensor.py",
    "test_copy.py",
    "test_scope.py",
    "test_fixpipe.py",
    "test_sync_block.py",
    "test_custom.py",
    "test_alloc.py",
    "test_sort.py",
    "test_subview.py",
    "test_to_buffer.py",
    "test_fusedattention.py",
    #
    # `triton.extension.buffer.language`
    "test_arch.py",
    "test_alloc.py",
    "test_bind_buffer.py",
    "test_copy.py",
    "test_fixpipe.py",
    "test_subview.py",
    "test_to_buffer.py",
    "test_to_tensor.py",
    #
    # `triton.language.extra.kernels`
    "test_gather_simd.py",
    #
    # `einops`
    "test_scalarPointer.py",
    "test_attn_cp.py",
    #
    # needs `triton.runtime.libentry`
    "test_cumsum.py",
    "test_cumprod.py",
    "test_sum_vector.py",
    "test_flip.py",
    #
    # ignored
    "test_add_mindspore.py",
    "test_softmax_mindspore.py",
    "test_downgrade.py",
    "test_index_select_inductor.py",
    "test_matmul_mindspore.py",
    #
}

SKIP_TESTS = [
    "test_advance.py::test_advance_with_boundary_check[shape0-float32]",
    "test_advance.py::test_advance_supplement[shape0-float32]",
    "test_advance.py::test_advance_supplement[shape1-float32]",
    "test_advance.py::test_advance_supplement[shape2-float32]",
    "test_advance.py::test_advance_supplement[shape3-float32]",
]

SUPPORTED_DTYPES = {"float32"}
# SUPPORTED_DTYPES = {"float16"}
SUPPORTED_DTYPES = os.environ.get("TTX_PYTEST_DTYPE", None) or SUPPORTED_DTYPES
SUPPORTED_SHAPES = {
    # 1d
    (1,),
    # (2,),
    (8,),
    # (64,),
    # (256,),
    # 2d
    # (1, 16),
    (8, 32),
    # 3d
    (4, 8, 256),
    # 4d
    (8, 4, 8, 8),
    (2, 2, 4, 8),
    # 5d
    (2, 8, 4, 8, 8),
    (2, 2, 2, 4, 8),
}
SKIP_INVALID_TESTS = True
_EXTRA_TEST_SHAPE5D = (2, 8, 4, 8, 8)

# TODO: check shapes
SUPPORTED_SHAPES = {}


def _is_dtype_param(name):
    return "dtype" in name or name in _DTYPE_PARAM_NAMES


def _is_shape_param(name):
    return "shape" in name or name == "shaape"


def _patch_testutils_shape5d():
    import test_common

    test_utils = test_common.TestUtils
    if _EXTRA_TEST_SHAPE5D in test_utils.test_shape5d:
        return

    test_utils.test_shape5d.append(_EXTRA_TEST_SHAPE5D)
    test_utils.full_shape_4_8d = (
        test_utils.test_shape4d
        + test_utils.test_shape5d
        + test_utils.test_shape6d
        + test_utils.test_shape7d
        + test_utils.test_shape8d
    )


def _extract_dtypes(value):
    dtype = _normalize_dtype(value)
    if dtype:
        return [dtype]
    dtypes = []
    if isinstance(value, (list, tuple, set)):
        for elem in value:
            dtype = _normalize_dtype(elem)
            if dtype:
                dtypes.append(dtype)
    return dtypes


def _extract_shapes(value):
    shape = _normalize_shape(value)
    if shape is not None:
        return [shape]
    shapes = []
    if isinstance(value, (list, tuple, set)):
        for elem in value:
            shape = _normalize_shape(elem)
            if shape is not None:
                shapes.append(shape)
    return shapes


def _collect_item_filters(item: pytest.Item):
    dtypes = []
    shapes = []
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return dtypes, shapes
    for name, value in callspec.params.items():
        lname = name.lower()
        if _is_dtype_param(lname):
            dtypes.extend(_extract_dtypes(value))
        elif _is_shape_param(lname):
            shapes.extend(_extract_shapes(value))
        elif lname == "param_list":
            dtypes.extend(_extract_dtypes(value))
            shapes.extend(_extract_shapes(value))
    return dtypes, shapes


def _get_test_name(item: pytest.Item):
    original_name = getattr(item, "originalname", None)
    if isinstance(original_name, str):
        return original_name
    return item.name.split("[", 1)[0]


def _is_invalid_test_item(item: pytest.Item):
    test_name = _get_test_name(item)
    is_invalid = _INVALID_TEST_PATTERN.match(test_name) is not None
    is_invalid_dtype = _INVALID_DTYPE_TEST_PATTERN.match(test_name) is not None
    return is_invalid or is_invalid_dtype


def _is_skip_test_item(item: pytest.Item):
    if not SKIP_TESTS:
        return False, None

    full_nodeid = item.nodeid
    if full_nodeid in SKIP_TESTS:
        return True, full_nodeid

    short_nodeid = f"{os.path.basename(str(item.fspath))}::{item.name}"
    if short_nodeid in SKIP_TESTS:
        return True, short_nodeid

    return False, None


def pytest_collection_modifyitems(config, items: list[pytest.Item]):
    if SKIP_INVALID_TESTS:
        for item in items:
            if _is_invalid_test_item(item):
                item.add_marker(pytest.mark.skip(reason="skipped by invalid-test filter"))

    if not SUPPORTED_DTYPES and not SUPPORTED_SHAPES and not SKIP_TEST_FILES and not SKIP_TESTS:
        return

    for item in items:
        if SKIP_TESTS:
            is_skip_item, matched_nodeid = _is_skip_test_item(item)
            if is_skip_item:
                item.add_marker(pytest.mark.skip(reason=f"skipped by test filter: {matched_nodeid}"))
                continue

        if SKIP_TEST_FILES:
            filename = os.path.basename(str(item.fspath))
            if filename in SKIP_TEST_FILES:
                item.add_marker(pytest.mark.skip(reason=f"skipped by file filter: {filename}"))
                continue
        dtypes, shapes = _collect_item_filters(item)
        reasons = []
        if SUPPORTED_DTYPES and dtypes:
            unsupported = sorted({d for d in dtypes if d not in SUPPORTED_DTYPES})
            if unsupported:
                reasons.append(f"unsupported dtype(s): {', '.join(unsupported)}")
        if SUPPORTED_SHAPES and shapes:
            unsupported_shapes = [s for s in shapes if s not in SUPPORTED_SHAPES]
            if unsupported_shapes:
                reasons.append(f"unsupported shape(s): {', '.join(str(s) for s in unsupported_shapes)}")
        if reasons:
            # print(f"skip item: {item}")
            item.add_marker(pytest.mark.skip(reason="; ".join(reasons)))

    # print callspec
    if not bool(os.environ.get("TTX_PYTEST_QUIET", 0)):
        for item in items:
            if hasattr(item, "callspec"):
                print(f"{item.nodeid}, [Params]: {item.callspec.params}")
            else:
                print(f"{item.nodeid}")

    GLOBAL_TIMEOUT = int(os.environ.get("TTX_PYTEST_GLOBAL_TIMEOUT", 10))
    for item in items:
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(GLOBAL_TIMEOUT))


# skip for tta-ut
# def pytest_configure(config):
#     _patch_testutils_shape5d()


def pytest_ignore_collect(collection_path, config):
    if not SKIP_TEST_FILES:
        return False
    filename = os.path.basename(str(collection_path))
    return filename in SKIP_TEST_FILES


def pytest_collection_finish(session: pytest.Session):
    print("\n=== Final Selected Tests (Will Run) ===")
    idx = 0
    for item in session.items:
        if item.get_closest_marker("skip") is None:
            print(item.nodeid)
            idx += 1
    print(f"len: {idx}")
    print("=======================================\n")


def pytest_runtest_call(item):
    print(f"\n[DEBUG] Test ID: {item.name}")
    print(f"[DEBUG] Parameters: {item.funcargs}")


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    import torch.cpu

    torch.cpu.set_device(0)
