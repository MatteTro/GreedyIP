# =============================================================================
# Toy test: reconstruction of an image from a subset of Fourier samples
# selected with a greedy strategy ('f-greedy' or 'p-greedy', chosen below).
#
# Pipeline:
#   1. Build a test image and its Fourier transform (visibilities).
#   2. Keep only the frequencies inside a radial mask.
#   3. Select a subset of those frequencies with the chosen greedy strategy.
#   4. Interpolate the visibilities from the selected frequencies (VSK-type
#      kernel on the augmented space (u, v, Re(V_tik), Im(V_tik))).
#   5. Reconstruct the image with Tikhonov regularization from the interpolated
#      visibilities.
#   6. Plot results and compute metrics, comparing the greedy reconstruction
#      with the plain Tikhonov reconstruction on the mask.
# =============================================================================

# -------------------------
# IMPORTS
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import aux_functions as f
import img_test as i_test
from aux_functions import Fourier_matrix, tik_svd, tik_solve

# -------------------------
# GLOBAL PARAMETERS
# -------------------------
greedy_selection = 'f-greedy'   # selection strategy: 'f-greedy' or 'p-greedy'

if greedy_selection not in ('f-greedy', 'p-greedy'):
    raise ValueError("greedy_selection must be 'f-greedy' or 'p-greedy'")

N = 64              # the image is N x N pixels
n_sel = 399         # target number of selected frequencies
alpha = 1e-08       # Tikhonov regularization parameter
symm = True         # if True, each selected frequency is paired with its conjugate-symmetric one

function_name = "gaussian_mixture_signed"   # test image (see img_test.get_function)

save_figures = False                        # if True, the figures are saved in `output_folder`
output_folder = "toytest_final_images"

# -------------------------
# TEST IMAGE
# -------------------------
# Pixel grid on [0, 1] x [0, 1]; `matrix` contains the (x, y) coordinates of
# all the pixels, one per row. It is used later as evaluation set of the interpolant.
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
matrix = np.column_stack((X.ravel(), Y.ravel()))

img_function = i_test.get_function(function_name)
img = img_function(X, Y)

# -------------------------
# FOURIER OPERATOR
# -------------------------
# Frequency grid as in Tests_main (uv_min_max = 0.5 => same grid k/(N-1) - 0.5)
pix = 1
uv_min_max = 0.5

u_1d = np.linspace(0, 1, N) * 2 * uv_min_max - uv_min_max
v_1d = np.linspace(0, 1, N) * 2 * uv_min_max - uv_min_max

U_2d, V_2d = np.meshgrid(u_1d, v_1d)

# Fourier matrix (image -> visibilities on the whole frequency grid)
F_grid_2d = Fourier_matrix(U_2d.flatten(), V_2d.flatten(), N, pix)

# Visibilities of the ground-truth image on the whole grid
F_full = np.reshape(F_grid_2d @ img.flatten(), (N, N))

# -------------------------
# RADIAL MASK ON THE FREQUENCIES
# -------------------------
# Only the frequencies inside a disk of radius 0.3 (centered in (0.5, 0.5))
# are considered available.
mask_radial = np.zeros((N, N), dtype=bool)
kx = np.arange(N) / (N - 1)
ky = np.arange(N) / (N - 1)
KX, KY = np.meshgrid(kx, ky)

R = np.sqrt((KX - 0.5)**2 + (KY - 0.5)**2)

mask_radial[R < 0.3] = True

# Fourier operator restricted to the masked frequencies and its SVD
# (factorized once and reused for every Tikhonov solve)
F_mask = F_grid_2d[mask_radial.flatten(), :]
svd_mask = tik_svd(F_mask)

# Visibilities of the ground truth, set to zero outside the mask
F_masked = mask_radial * F_full

# Baseline: Tikhonov reconstruction from ALL the masked visibilities
rec = np.reshape(np.real(tik_solve(svd_mask, F_masked[mask_radial], alpha)), (N, N))

# Coordinates and visibilities of the available (masked) frequencies
u = KX[mask_radial]       # u coordinates
v = KY[mask_radial]       # v coordinates
V = F_full[mask_radial]   # visibilities (true values) at the masked frequencies

# Visibilities of the baseline reconstruction on the whole grid:
# - V_img_rec: on the whole grid (also used in the metrics)
# - V_img_rec_masked: zero outside the mask (extra coordinates of the VSK kernel)
V_img_rec = np.reshape(F_grid_2d @ rec.flatten(), (N, N))
V_img_rec_masked = V_img_rec * mask_radial

# -------------------------
# GREEDY SELECTION
# -------------------------
# Both strategies start from the frequency with the largest modulus.
start_idx = [np.argmax(np.abs(V))]

if greedy_selection == 'f-greedy':
    # f-greedy: the next frequency is the one with the largest residual between
    # the true and the currently reconstructed visibilities
    selected_idx, _ = f.fgreedy_fourier_selection(
        N,
        u, v,
        V,
        matrix,
        selected_idx=start_idx,
        mask=mask_radial,
        n_points=n_sel,
        F_masked=F_masked,
        symm=symm,
        alfa=alpha,   # `alfa` is the keyword name used in aux_functions
        svd_mask=svd_mask,
        F_mask=F_mask
    )
else:
    # p-greedy: the next frequency is the one maximizing the power function
    selected_idx, _ = f.pgreedy_fourier_selection(
        u, v,
        n_points=n_sel,
        eps=0.01,
        selected_idx=start_idx,
        symm=symm,
        plot=True
    )

# Summary of the selection
n_total = len(u)
n_selected = len(selected_idx)

percentage = 100 * n_selected / n_total

print("\n--- GREEDY SUMMARY ---")
print(f"Selection strategy: {greedy_selection}")
print(f"Total available frequencies: {n_total}")
print(f"Selected frequencies: {n_selected}")
print(f"Percentage selected: {percentage:.2f}%")

selected_u = u[selected_idx]
selected_v = v[selected_idx]

# -------------------------
# INTERPOLATION OF THE VISIBILITIES
# -------------------------
# Data sites: (u, v, Re(V), Im(V)) at the selected frequencies
dsites = np.vstack((selected_u, selected_v)).T

dsites_r = np.hstack((dsites, np.real(V[selected_idx].reshape(-1, 1))))
dsites_a = np.hstack((dsites_r, np.imag(V[selected_idx].reshape(-1, 1))))

# Evaluation points: all the pixels of the frequency grid, with the baseline
# reconstruction as extra coordinates (variably scaled kernel)
epoints_r = np.hstack((matrix, np.real(V_img_rec_masked).reshape(-1, 1)))
epoints_a = np.hstack((epoints_r, np.imag(V_img_rec_masked).reshape(-1, 1)))

# Values to interpolate: real and imaginary parts of the selected visibilities
re_selected = np.real(V[selected_idx].reshape(-1, 1))
im_selected = np.imag(V[selected_idx].reshape(-1, 1))

interp_out = f.matern_kernel_interp(dsites_a, epoints_a, np.hstack((re_selected, im_selected)), ep=0.001)

interp_re = interp_out['Pf1'][:, 0]
interp_im = interp_out['Pf1'][:, 1]

interp_re = interp_re.reshape(N, N)
interp_im = interp_im.reshape(N, N)

# Interpolated visibilities, restricted to the mask
V_interp_greedy = interp_re + 1j * interp_im
V_interp_mask = V_interp_greedy * mask_radial

# Matrix showing (in log scale) the modulus of the selected visibilities only
sampled_matrix = np.zeros_like(F_full, dtype=float)

mask_positions = np.argwhere(mask_radial)

for idx in selected_idx:
    i, j = mask_positions[idx]
    sampled_matrix[i, j] = np.log1p(np.abs(V[idx]))

# -------------------------
# IMAGE RECONSTRUCTION
# -------------------------
# Tikhonov reconstruction from the interpolated visibilities
y_tik = V_interp_mask[mask_radial]
rec_greedy = np.reshape(np.real(tik_solve(svd_mask, y_tik, alpha)), (N, N))

# Plot of the selected frequencies inside the mask
plt.figure(figsize=(6, 6))

plt.scatter(u, v, s=5, alpha=0.2, label='Mask')

plt.scatter(
    selected_u,
    selected_v,
    c='red',
    s=20,
    label=greedy_selection.title()
)

plt.axis('equal')
plt.legend()
plt.show()

if save_figures:
    os.makedirs(output_folder, exist_ok=True)

##############################################
################### Plots ####################
##############################################

#-------------------------
# Reconstructed frequencies
#-------------------------
# Layout (log scale):
#   [ full Fourier matrix   | selected frequencies (sampled) ]
#   [ masked Fourier matrix | interpolated visibilities      ]

colormap_freq = plt.colormaps['afmhot'].resampled(256)
colormap_freq.set_bad(color='#000000')

# wspace/hspace give a small uniform gap between the (square) panels
fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

# Column 0: Fourier matrices of the ground truth
data_00 = np.log(1 + np.abs(F_full))
axes[0, 0].imshow(data_00, cmap=colormap_freq, vmin=0, vmax=data_00.max())
axes[0, 0].axis('off')

data_10 = np.log(1 + np.abs(F_masked))
axes[1, 0].imshow(data_10, cmap=colormap_freq, vmin=0, vmax=data_10.max())
axes[1, 0].axis('off')

# Column 1: selected frequencies and interpolated visibilities
# (zeros are masked so that they are shown with the "bad" color, i.e. black)
data_01_masked = np.ma.masked_equal(sampled_matrix, 0)
axes[0, 1].imshow(data_01_masked, cmap=colormap_freq, vmin=0, vmax=sampled_matrix.max())
axes[0, 1].axis('off')

data_11 = np.log(1 + np.abs(V_interp_mask))
data_11_masked = np.ma.masked_equal(data_11, 0)
axes[1, 1].imshow(data_11_masked, cmap=colormap_freq, vmin=0, vmax=data_11.max())
axes[1, 1].axis('off')

# Keep the figure borders close to the panels
plt.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)

if save_figures:
    plt.savefig(os.path.join(output_folder, f"Fourier_{greedy_selection}.png"), dpi=300, bbox_inches='tight')
plt.show()


#-------------------------------------------
# Real and imaginary part of the Fourier transform: original vs reconstructed
#-------------------------------------------

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

# Original (masked) visibilities: real part
im_re_orig = ax[0, 0].imshow(np.log(1 + np.abs(np.real(F_masked))), cmap='viridis')
ax[0, 0].set_title('Original Real Part')
plt.colorbar(im_re_orig, ax=ax[0, 0])

# Original (masked) visibilities: imaginary part
im_im_orig = ax[0, 1].imshow(np.log(1 + np.abs(np.imag(F_masked))), cmap='viridis')
ax[0, 1].set_title('Original Imaginary Part')
plt.colorbar(im_im_orig, ax=ax[0, 1])

# Interpolated visibilities: real part
im_re_rec = ax[1, 0].imshow(np.log(1 + np.abs(np.real(V_interp_mask))), cmap='viridis')
ax[1, 0].set_title('Reconstructed Real Part')
plt.colorbar(im_re_rec, ax=ax[1, 0])

# Interpolated visibilities: imaginary part
im_im_rec = ax[1, 1].imshow(np.log(1 + np.abs(np.imag(V_interp_mask))), cmap='viridis')
ax[1, 1].set_title('Reconstructed Imaginary Part')
plt.colorbar(im_im_rec, ax=ax[1, 1])

fig.suptitle(greedy_selection.title(), fontsize=16, fontweight='bold', x=0.5)
plt.tight_layout()

plt.show()


#----------------------
# Reconstructed images
#----------------------
# Left: ground truth; center: Tikhonov on all the masked visibilities;
# right: Tikhonov on the visibilities interpolated from the selected frequencies.
# All the panels share the same color scale (that of the ground truth).

colormap_img = 'gist_heat'

img_min = img.min()
img_max = img.max()

fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))

im0 = ax[0].imshow(img, cmap=colormap_img, vmin=img_min, vmax=img_max)
ax[0].set_title('Original')

ax[1].imshow(rec, cmap=colormap_img, vmin=img_min, vmax=img_max)
ax[1].set_title('Tikhonov (mask)')

ax[2].imshow(rec_greedy, cmap=colormap_img, vmin=img_min, vmax=img_max)
ax[2].set_title(greedy_selection.title())

cbar = fig.colorbar(im0, ax=ax, location="right", shrink=0.92, pad=0.02)
cbar.ax.tick_params(labelsize=5)

if save_figures:
    plt.savefig(os.path.join(output_folder, f"Rec_{greedy_selection}.png"), dpi=300, bbox_inches='tight')
plt.show()


##############################################
################## Metrics ###################
##############################################

# --- Fourier-space metrics (computed directly on the interpolated visibilities) ---
# Moduli are rescaled with the dynamic range of the true visibilities:
# (|.| - min|V|) / (max|V| - min|V|)
V_abs_min = np.min(np.abs(V))
V_range = np.max(np.abs(V) - V_abs_min)

V_norm = (np.abs(V) - V_abs_min) / V_range

# Greedy: interpolated visibilities on the mask
V_pred_norm = (np.abs(V_interp_mask[mask_radial]) - V_abs_min) / V_range

# Baseline: visibilities of the Tikhonov reconstruction on the mask
V_img_rec_norm = (np.abs(V_img_rec[mask_radial]) - V_abs_min) / V_range

# --- Image-space metrics ---
# Images are rescaled to [0, 1] using the range of the ground truth
img_range = np.max(img - np.min(img))
img_norm = (img - np.min(img)) / img_range
rec_norm = (rec - np.min(img)) / img_range
rec_greedy_norm = (rec_greedy - np.min(img)) / img_range

# Fourier metrics: f.scores returns (chi-like/MSE, RMSE, MRE) -- only RMSE and MRE are used
_, rmse_greedy_fourier, mre_greedy_fourier = f.scores(V_norm, V_pred_norm, sigma=None)
_, rmse_mask_fourier, mre_mask_fourier = f.scores(V_norm, V_img_rec_norm, sigma=None)

# Image metrics
_, rmse_greedy_img, mre_greedy_img = f.scores(img_norm.reshape(-1, 1), rec_greedy_norm.reshape(-1, 1), sigma=None)
_, rmse_mask_img, mre_mask_img = f.scores(img_norm.reshape(-1, 1), rec_norm.reshape(-1, 1), sigma=None)

# SSIM on the rescaled images
ssim_mask = f.ssim(img_norm, rec_norm, data_range=1.0)
ssim_greedy = f.ssim(img_norm, rec_greedy_norm, data_range=1.0)

method_name = greedy_selection.upper()

print("\n" + "=" * 90)
print(f"[Function: {function_name}]  [Selection: {greedy_selection}]")
print("=" * 90)
print(f"{'METHOD':<18} | {'FOURIER RMSE':<12} | {'FOURIER MRE':<12} | {'IMG RMSE':<12} | {'IMG MRE':<12} | {'SSIM':<12}")
print("=" * 90)
print(f"{'MASK':<18} | {rmse_mask_fourier:<12.3e} | {mre_mask_fourier:<12.3e} | {rmse_mask_img:<12.3e} | {mre_mask_img:<12.3e} | {ssim_mask:<12.3e}")
print(f"{method_name:<18} | {rmse_greedy_fourier:<12.3e} | {mre_greedy_fourier:<12.3e} | {rmse_greedy_img:<12.3e} | {mre_greedy_img:<12.3e} | {ssim_greedy:<12.3e}")
print("=" * 90)