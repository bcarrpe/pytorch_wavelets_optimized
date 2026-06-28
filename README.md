# pytorch_wavelets_optimized

`pytorch_wavelets_optimized` is a research fork of [`fbcotter/pytorch_wavelets`](https://github.com/fbcotter/pytorch_wavelets) focused on one practical goal:

> make first-order DTCWT scattering usable as a fast, full-resolution input stem for real-time maritime instance segmentation.

The fork keeps the original Python package name and import API, so existing code can still use:

```python
from pytorch_wavelets import ScatLayer
```

## Why this fork exists

The original `pytorch_wavelets` `ScatLayer` computes first-order DTCWT scattering coefficients and downsamples the spatial resolution. For RGB input, the original behavior is:

| Input | Output |
|---|---|
| `(N, 3, H, W)` | `(N, 21, H/2, W/2)` |

In ScatYOLO, this creates an architectural inconvenience: if the downstream YOLO backbone expects feature maps at the original input resolution, the image must first be upsampled before scattering. That adds redundant computation:

```text
RGB image -> upsample -> ScatLayer downsamples -> full-resolution scattering maps
```

This fork removes that redundancy. The optimized `ScatLayer` preserves the input spatial resolution directly:

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

This enables the simpler and faster ScatYOLO preprocessing path:

```text
RGB image -> optimized ScatLayer -> full-resolution scattering maps
```

## What changed

This fork applies the optimized first-order scattering behavior used in the optimized ScatYOLOv8+CBAM work:

- the DTCWT quadrant-to-complex conversion path is adapted to keep full-resolution subimages;
- the first-order scattering low-pass output is no longer average-pooled by a factor of 2;
- `PyWavelets` is declared as an install dependency;
- a regression test checks that RGB input `(N, 3, H, W)` produces `(N, 21, H, W)`.

The public import API is unchanged.

## Intended use

This fork is intended for models that use fixed wavelet scattering coefficients as an input feature expansion before a CNN detector or segmenter, especially ScatYOLO-style architectures for maritime perception.

Typical use:

```python
import torch.nn.functional as F
from pytorch_wavelets import ScatLayer

scat = ScatLayer(biort="near_sym_a")

def scat_preprocess(x):
    return F.relu(scat(x))
```

For an RGB image tensor with shape `(N, 3, 640, 640)`, this produces a 21-channel tensor with shape `(N, 21, 640, 640)`.

## Installation

Install directly from GitHub:

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

## Test

```bash
python -m pytest tests/test_scatlayer_optimized.py -q
```

Expected result:

```text
1 passed
```

## Research context

This optimized scattering implementation comes from the ScatYOLOv8+CBAM line of work for real-time ship segmentation on embedded systems.

In the original ScatBlock design, the input image was upsampled before scattering to compensate for the resolution decrease caused by first-order scattering. The optimized ScatBlock removes this redundant upsampling/downsampling behavior by modifying the `pytorch_wavelets` scattering path so that the ScatLayer output keeps the original image resolution.

In the reported embedded deployment on NVIDIA Jetson AGX Xavier with TensorRT, the optimized ScatYOLOv8+CBAM nano model reached comparable mask mAP to the previous ScatYOLOv8+CBAM design while reducing inference time from 39.9 ms to 25.3 ms. The same work also shows that optimized ScatYOLOv8+CBAM improves small-ship segmentation compared with standard YOLOv8 across model sizes, and that combining it with SAHI further improves small-object mAP.

Please cite:

- Borja Carrillo-Perez, Ángel Bueno Rodríguez, Sarah Barnes, Maurice Stephan, “Enhanced Small Ship Segmentation with Optimized ScatYOLOv8+CBAM on Embedded Systems,” *2024 IEEE International Conference on Real-time Computing and Robotics (RCAR)*, pp. 13–18, 2024. DOI: `10.1109/RCAR61438.2024.10670759`.

- Borja Jesus Carrillo Perez, “Real-time Ship Recognition and Georeferencing for the Improvement of Maritime Situational Awareness,” Dissertation, University of Bremen, 2024. DOI: `10.26092/elib/3265`.

## Upstream

This repository is a fork of:

<https://github.com/fbcotter/pytorch_wavelets>

The original project provides differentiable 2D wavelet transforms and DTCWT-based scattering layers in PyTorch.
