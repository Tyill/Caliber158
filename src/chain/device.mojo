"""Compute device selection for student training."""

from std.os import getenv
from std.sys import has_accelerator


@fieldwise_init
struct DeviceKind(Equatable, ImplicitlyCopyable, Copyable, Movable):
    """Student train device: CPU or CUDA."""

    var _tag: Int  # 0 = cpu, 1 = cuda

    @staticmethod
    def cpu() -> DeviceKind:
        return DeviceKind(0)

    @staticmethod
    def cuda() -> DeviceKind:
        return DeviceKind(1)

    def is_cuda(self) -> Bool:
        return self._tag == 1

    def label(self) -> String:
        if self.is_cuda():
            return "cuda"
        return "cpu"


def cuda_available() -> Bool:
    """Return whether a compatible GPU accelerator is present."""
    return has_accelerator()


def resolve_device_from_env() -> DeviceKind:
    """Resolve CALIBER158_DEVICE: cuda if requested and available, else cpu."""
    var pref = getenv("CALIBER158_DEVICE", "cuda").strip().lower()
    if pref == "cpu":
        return DeviceKind.cpu()
    if pref == "cuda":
        if cuda_available():
            return DeviceKind.cuda()
        print("warning: CALIBER158_DEVICE=cuda but CUDA unavailable; using cpu")
        return DeviceKind.cpu()
    print("warning: unknown CALIBER158_DEVICE=", pref, "; using cpu")
    return DeviceKind.cpu()


def train_backend_from_env() -> String:
    """Return CALIBER158_TRAIN_BACKEND (must be mojo)."""
    var backend = getenv("CALIBER158_TRAIN_BACKEND", "mojo").strip().lower()
    if backend != "mojo":
        print("warning: CALIBER158_TRAIN_BACKEND=", backend, " not supported; using mojo")
        return "mojo"
    return "mojo"
