# GreedyIP
# Brief description
This repository contains the code developed to study greedy point selection (specifically residual-based and error-based strategies) applied to image reconstruction using different inversion methods.
The framework allows you to:

- simulate different flare configurations (e.g., one source, two sources, etc.)

- apply various targeted subsampling strategies

- reconstruct the image using different inversion methods

- compare the resulting reconstruction metrics

# Contents
This repository contains a demo for greedy selection of visibilities and UV interpolation using kernel-based methods. The demo allows simulating different types of solar flares, generating the corresponding visibilities, and performing greedy selection and reconstruction via different reconstruction methods
## Files overview
- "**Kernel_utilities**".py:
Utility functions for kernel interpolation and feature construction:
- "**greedy_utilities.py**":
Implements greedy selection of visibilities
- "**uv_smooth.py**":
Functions for UV interpolation and reconstruction
- "**simulator.py**":
Functions for simulating flare sources and generating visibilities or Fourier matrices for testing and demo purposes.

# Citations
If you use this code please cite

> L. Bruni Bruno, P. Massa, E. Perracchione, M. Trombini, Greedy methods in inverse problems, in preparation 2025
