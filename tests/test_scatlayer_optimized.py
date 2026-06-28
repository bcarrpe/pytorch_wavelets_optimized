import torch
from pytorch_wavelets import ScatLayer


def test_scatlayer_rgb_optimized_shape():
    x = torch.randn(2, 3, 64, 64)
    y = ScatLayer(biort="near_sym_a")(x)
    assert tuple(y.shape) == (2, 21, 64, 64)
