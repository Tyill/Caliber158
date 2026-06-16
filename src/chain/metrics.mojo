"""Dataset metrics for train/holdout evaluation."""

from .buffer import ChainData


def variance_y(data: ChainData) -> Float32:
    """Population variance of Y over all samples in data."""
    if data.n_samples == 0:
        return 0.0

    var mean: Float32 = 0.0
    for i in range(data.n_samples):
        mean += data.y_at(i)
    mean /= Float32(data.n_samples)

    var var_acc: Float32 = 0.0
    for i in range(data.n_samples):
        var d = data.y_at(i) - mean
        var_acc += d * d
    return var_acc / Float32(data.n_samples)


def relative_mse(mse: Float32, var_y: Float32) -> Float32:
    """MSE divided by Var(Y); 1.0 ≈ predicting the mean."""
    if var_y <= 0.0:
        return 0.0
    return mse / var_y
