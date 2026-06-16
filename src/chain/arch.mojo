"""Student network architecture selection (domain layer)."""

from std.os import getenv


@fieldwise_init
struct ArchKind(Equatable, ImplicitlyCopyable, Copyable, Movable):
    """Student arch: v0 single SwiGLU, v1 dual SwiGLU + residual."""

    var _tag: Int  # 0 = v0, 1 = v1

    @staticmethod
    def v0() -> ArchKind:
        return ArchKind(0)

    @staticmethod
    def v1() -> ArchKind:
        return ArchKind(1)

    def is_v1(self) -> Bool:
        return self._tag == 1


def arch_from_env() -> ArchKind:
    var raw = getenv("CALIBER158_ARCH", "v0")
    if raw == "v1":
        return ArchKind.v1()
    return ArchKind.v0()


def block2_weight_count(arch: ArchKind, hidden_dim: Int) -> Int:
    if arch.is_v1():
        return hidden_dim * hidden_dim
    return 0


def param_count(arch: ArchKind, input_dim: Int, hidden_dim: Int) -> Int:
    var block1 = 2 * hidden_dim * input_dim
    var block2 = 2 * block2_weight_count(arch, hidden_dim)
    return block1 + block2 + hidden_dim + 1


def arch_label(arch: ArchKind) -> String:
    if arch.is_v1():
        return "v1"
    return "v0"
