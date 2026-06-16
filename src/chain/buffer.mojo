"""Dense contiguous buffers for batched chain training."""

from .dataset import ChainDataset


@fieldwise_init
struct ChainData(Copyable, Movable):
    """Row-major dataset storage: X [n_samples × input_dim], Y [n_samples]."""

    var n_samples: Int
    var input_dim: Int
    var x_data: List[Float32]
    var y_data: List[Float32]

    @staticmethod
    def from_dataset(dataset: ChainDataset) -> ChainData:
        var x_data = List[Float32](capacity=len(dataset.x_data))
        var y_data = List[Float32](capacity=len(dataset.y_data))
        for i in range(len(dataset.x_data)):
            x_data.append(dataset.x_data[i])
        for i in range(len(dataset.y_data)):
            y_data.append(dataset.y_data[i])
        return ChainData(dataset.n_samples, dataset.input_dim, x_data^, y_data^)

    def batch_size(self, start: Int, requested: Int) -> Int:
        var end = start + requested
        if end > self.n_samples:
            end = self.n_samples
        return end - start

    def x_offset(self, sample_index: Int) -> Int:
        return sample_index * self.input_dim

    def x_at(self, sample_index: Int, feature_index: Int) -> Float32:
        return self.x_data[self.x_offset(sample_index) + feature_index]

    def y_at(self, sample_index: Int) -> Float32:
        return self.y_data[sample_index]
