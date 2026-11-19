# GreedyIP
# Brief description
This repository contains the code developed to study greedy point selection (specifically residual-based and error-based strategies) applied to image reconstruction using different inversion methods.
The framework allows you to:

- simulate different flare configurations (e.g., one source, two sources, etc.)

- apply various targeted subsampling strategies

- reconstruct the image using different inversion methods

- compare the resulting reconstruction metrics

# Example workflow

## 1. Simulated flare configurations

Synthetic solar-flare scenarios generated using the `simulator.py` module.

![Ground truth](GT_images.png)
<p align="center">
  <strong>Ground-truth emission models.</strong>
</p>

## 2. Greedy points selection  
Residual-based and error-based strategies used to select informative points.  

<p align="center">
  <img src="Sel_Points.png" width="70%">
</p>
<p align="center">
  <strong>UV sampling and f-greedy selected points across configurations and strategies.</strong>
</p>

## 3. Reconstruction from selected visibilities  
Reconstruction obtained from full sampling, p-greedy points, and f-greedy points using different inversion methods.  

<p align="center">
  <img src="Loop.png" width="70%">
</p>
<p align="center">
  <strong>Example reconstruction (Loop configuration).</strong>
</p>

## 4. Reconstruction metrics

To evaluate the reconstruction quality across different sampling strategies and inversion methods, the framework computes standard quantitative metrics, including:

- **χ² (Chi-square)** — discrepancy between observed and model visibilities  
- **RMSE (Root Mean Square Error)** — absolute reconstruction error  
- **MRE (Mean Relative Error)** — relative reconstruction error

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

# Testing greedy selection on simulated data

The `Sim_data` folder provides a workflow for testing greedy point-selection strategies with Tikhonov-regularized image reconstruction. It can be used with different test functions, including examples outside astronomical imaging.

<p align="center">
  <img src="Rec_400punti.png" alt="Example Reconstruction" width="600">
  <br>
  <strong>Example Reconstruction.</strong>
</p>

Choose or define the test function in `img_test.py`, then use the script to evaluate the selected greedy strategy and reconstruct the corresponding image. See `img_test.py` for the available options and expected function format.

The workflow supports residual-based and error-based greedy point-selection strategies.

The folder contains:
- **`main_sim.py`**: Main script for running the simulated-data workflow.
- **`aux_function.py`**: Utility functions for interpolation, greedy point selection, image inversion and reconstruction, and reconstruction metrics.
- **`img_test.py`**: Test functions used in the examples. Add your own functions here to test the workflow on other image models, including non-astronomical ones.


# Citations
If you use this code please cite

> L. Bruni Bruno, P. Massa, E. Perracchione, M. Trombini, Greedy methods for inverse problems: application to image reconstruction, under review 2025
