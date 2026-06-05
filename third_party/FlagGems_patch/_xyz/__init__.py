from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="xyz",
    device_name="cpu",
    device_query_cmd="false",
    triton_extra_name="xyz",
    tune_config_vendor="arm",
    heuristics_config_vendor="arm",
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
