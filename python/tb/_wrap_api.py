import torch


def _wrap_api(func):
    def wrapper(*args, **kwargs):
        if "device" in kwargs and "cuda" in str(kwargs["device"]):
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)

    return wrapper


for name in [
    "arange",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]:
    if hasattr(torch, name):
        setattr(torch, name, _wrap_api(getattr(torch, name)))
