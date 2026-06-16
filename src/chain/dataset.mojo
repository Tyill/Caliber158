"""Load teacher dataset from Caliber158 binary format."""

from std.memory import bitcast

from .rng import lcg_next, unit_float

comptime DATASET_MAGIC = "CAL158"
comptime DATASET_VERSION: UInt32 = 1


@fieldwise_init
struct ChainDataset(Copyable, Movable):
    """In-memory (X, Y) pairs for one chain distillation run."""

    var n_samples: Int
    var input_dim: Int
    var x_data: List[Float32]
    var y_data: List[Float32]

    @staticmethod
    def synthetic(n_samples: Int, input_dim: Int) -> ChainDataset:
        """Small random dataset for smoke tests without Python."""
        var x_data = List[Float32](capacity=n_samples * input_dim)
        var y_data = List[Float32](capacity=n_samples)

        var seed: UInt64 = 42
        for _ in range(n_samples * input_dim):
            seed = lcg_next(seed)
            x_data.append(unit_float(seed))

        for _ in range(n_samples):
            seed = lcg_next(seed)
            y_data.append(unit_float(seed) * 2.0 - 1.0)

        return ChainDataset(n_samples, input_dim, x_data^, y_data^)

    @staticmethod
    def load(path: String) raises -> ChainDataset:
        """Read dataset written by python/extract_chain.py."""
        var f = open(path, "r")
        var data = f.read_bytes()
        f.close()

        if len(data) < 18:
            raise Error("dataset file too small")

        _check_magic(data)

        var offset = 6
        var version = _read_u32_le(data, offset)
        offset += 4
        if version != DATASET_VERSION:
            raise Error("unsupported dataset version")

        var n_samples = Int(_read_u32_le(data, offset))
        offset += 4
        var input_dim = Int(_read_u32_le(data, offset))
        offset += 4

        var x_count = n_samples * input_dim
        var x_data = List[Float32](capacity=x_count)
        var y_data = List[Float32](capacity=n_samples)

        for _ in range(x_count):
            x_data.append(_read_f32_le(data, offset))
            offset += 4
        for _ in range(n_samples):
            y_data.append(_read_f32_le(data, offset))
            offset += 4

        return ChainDataset(n_samples, input_dim, x_data^, y_data^)

    def sample_input(self, index: Int) -> List[Float32]:
        """Return one input row (length input_dim)."""
        var row = List[Float32](capacity=self.input_dim)
        var base = index * self.input_dim
        for i in range(self.input_dim):
            row.append(self.x_data[base + i])
        return row^

    def target(self, index: Int) -> Float32:
        return self.y_data[index]


def _check_magic(data: List[UInt8]) raises -> None:
  # b"CAL158"
    if (
        data[0] != 67
        or data[1] != 65
        or data[2] != 76
        or data[3] != 49
        or data[4] != 53
        or data[5] != 56
    ):
        raise Error("invalid dataset magic, expected CAL158")


def _read_u32_le(data: List[UInt8], offset: Int) -> UInt32:
    var out: UInt32 = 0
    for i in range(4):
        out |= UInt32(data[offset + i]) << UInt32(i * 8)
    return out


def _read_f32_le(data: List[UInt8], offset: Int) -> Float32:
    var bits = _read_u32_le(data, offset)
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))[0]
