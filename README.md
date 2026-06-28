# pytorch_wavelets_optimized

Fork of [`fbcotter/pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets) with an optimized first-order DTCWT scattering path for ScatYOLO integration.

The original package provides differentiable 2D wavelet transforms and DTCWT-based scattering layers in PyTorch. This fork keeps the original package name and import API:

```python
from pytorch_wavelets import ScatLayer
```

## Optimized ScatLayer behavior

Compared with the original `ScatLayer`, this fork preserves the input spatial resolution for J=1 scattering.

| Input | Output |
|---|---|
| `(N, 3, H, W)` | `(N, 21, H, W)` |

For example:

```python
import torch
from pytorch_wavelets import ScatLayer

x = torch.randn(2, 3, 640, 640)
y = ScatLayer(biort="near_sym_a")(x)

print(y.shape)  # torch.Size([2, 21, 640, 640])
```

This behavior is covered by:

```bash
python -m pytest tests/test_scatlayer_optimized.py -q
```

## Installation

```bash
pip install git+https://github.com/bcarrpe/pytorch_wavelets_optimized.git@optimized-scatlayer
```

For editable development:

```bash
git clone https://github.com/bcarrpe/pytorch_wavelets_optimized.git
cd pytorch_wavelets_optimized
git checkout optimized-scatlayer
pip install -e . --no-build-isolation
```

## Research context

This optimized scattering implementation is associated with the ScatYOLOv8+CBAM work on embedded small-ship segmentation and the related PhD thesis on real-time ship recognition and georeferencing for maritime situational awareness.

Please cite:

- Borja Carrillo-Perez, Ángel Bueno Rodríguez, Sarah Barnes, Maurice Stephan, “Enhanced Small Ship Segmentation with Optimized ScatYOLOv8+CBAM on Embedded Systems,” *2024 IEEE International Conference on Real-time Computing and Robotics (RCAR)*, pp. 13–18, 2024. DOI: `10.1109/RCAR61438.2024.10670759`.

- Borja Jesus Carrillo Perez, “Real-time Ship Recognition and Georeferencing for the Improvement of Maritime Situational Awareness,” Dissertation, University of Bremen, 2024. DOI: `10.26092/elib/3265`.

## Upstream

Original project: <https://github.com/fbcotter/pytorch_wavelets>
