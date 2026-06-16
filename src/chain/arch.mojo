"""Student micro-network architecture version."""

from std.os import getenv


@fieldwise_init
struct ArchKind(Copyable, Movable):
    """v0: single SwiGLU block; v1: two blocks + hidden residual."""

    var _tag: Int  # 0 = v0, 1 = v1

    @staticmethod
    def v0() -> ArchKind:
        return ArchKind(0)

    @staticmethod
    def v1() -> ArchKind:
        return ArchKind(1)

    def is_v1(self) -> Bool:
        return self._tag == 1

    def label(self) -> String:
        if self.is_v1():
            return "v1"
        return "v0"


def resolve_arch_from_env() -> ArchKind:
    """Resolve CALIBER158_ARCH: v0 or v1 (default v1)."""
    var raw = getenv("CALIBER158_ARCH", "v1").strip().lower()
    if raw == "v0":
        return ArchKind.v0()
    if raw == "v1":
        return ArchKind.v1()
    print("warning: unknown CALIBER158_ARCH=", raw, "; using v1")
    return ArchKind.v1()
