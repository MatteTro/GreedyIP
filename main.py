# -------------------------
# IMPORTS
# -------------------------
import simulator as sm                  # Local simulator module: creates Gaussian sources and simulates visibilities
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu        # Astropy units for explicit handling of pixel/arcsec units

# xrayvision: MEM and visibility utilities (used for reconstructions)
from xrayvision.mem import mem, resistant_mean
from xrayvision.visibility import Visibilities#, VisMeta

# CLEAN implementation from xrayvision
from xrayvision.clean import vis_clean

# Project utilities: Kernel_utilities, uv_smooth
import Kernel_utilities as Ku
import uv_smooth as uv_smooth

# Greedy selection utilities implemented in the repository
import greedy_utilities as gr

# -------------------------
# GLOBAL PARAMETERS
# -------------------------
greedy_selection = 'p-greedy'   # selection strategy; 'p-greedy' or 'f-greedy' 

# Parameters for constructing a Fibonacci-like uv sampling
N = 400                          
n = 40                           
phi = (1 + np.sqrt(5)) / 2      
# Radius vector R: STIX radius
R = 0.03408518887061331 * np.sqrt(np.linspace(1/2, N - 1/2, N)) / np.sqrt(N - 1/2)
T = 4 / (1 + np.sqrt(5)) * np.pi * np.arange(1, N + 1)

xc = R * np.cos(T)
yc = R * np.sin(T)
y = np.column_stack((xc, yc))   #
u_fibo = xc.flatten()
v_fibo = yc.flatten()

# Mask: keep only points where u + v > 0. 
# This removes one half of the spiral to avoid complex conjugates.
mask = v_fibo + u_fibo > 0
u = u_fibo[mask]
v = v_fibo[mask]

add_noise = False              # if True, simulated visibilities include noise

# # Image geometry
npix = 128                     # number of pixels
pix = 1                        # pixel size 
imsize = [npix, npix] * apu.pixel               # shape with astropy unit (passed to xrayvision functions)
pixel = [pix, pix] * apu.arcsec / apu.pixel     # pixel size as astropy Quantity (arcsec/pixel)


config = 'Loop' # OneSource or TwoSources or Loop


# -------------------------
# SIMULATE SOURCES AND VISIBILITIES
# -------------------------

if config=='Loop':
    xc = [5.]
    yc = [0.]
    FWHM_max = [25]
    FWHM_min = [18]
    flux = [1000]
    srcpa = [45.]
    loop_angle = [70.]
    gt_im = sm.CreateLoopSources(xc, yc, FWHM_min, FWHM_max, flux, srcpa, loop_angle, npix=npix, pix=pix)
    vis = sm.SimulateLoopConfig(xc, yc, FWHM_min, FWHM_max, flux, srcpa, loop_angle, u, v, add_noise=add_noise)
    
    # vis_stx = sm.SimulateLoopConfig(xc, yc, FWHM_min, FWHM_max, flux, srcpa, loop_angle, u_stix, v_stix, add_noise=add_noise)

    
else:
    if config=='OneSource': 
        xc = [-5]                      # x center(s) in arcsec
        yc = [5]                       # y center(s) in arcsec
        FWHM_max = [25]                # maximum FWHM (arcsec)
        FWHM_min = [15]                # minimum FWHM (arcsec)
        flux = [10000]                 # flux value 
        angle = [30]                   # source orientation angle (degrees)
        
    else:
        
        xc = [15, -15]
        yc = [15, -15]
        FWHM_max = [20, 20]
        FWHM_min = [20, 20]
        flux = [1000, 1000]
        angle = [0, 0]
        
    gt_im = sm.CreateGaussianSources(xc, yc, FWHM_min, FWHM_max, flux, angle, npix=npix, pix=pix)

    # Simulate visibilities for the chosen uv sampling (u, v)
    vis = sm.SimulateConfig(xc, yc, FWHM_max, FWHM_min, flux, angle, u, v, add_noise=add_noise)
    
add_noise = False              # if True, simulated visibilities include noise


# -------------------------
# IMAGE RECONSTRUCTION (CLEAN, UV-SMOOTH, MEM_GE)
# -------------------------
key_map = 'clean'              # base map used by uv_smooth (e.g., 'clean' or 'bp')
threshold_PSI = 0              # threshold parameter used by uv_smooth 
ep = 0.01                      # shape parameter used by uv_smooth in kernel construction
plot_show = 1                  # whether to show plots (1 = show)
beam_width = 12.0              # CLEAN beam width 
# Build clean beam with astropy units 
clean_beam_width = beam_width * apu.arcsec

# Run uv-smooth  on the full visibility set
vis_2 = Ku.duplicate_vis(vis)
uv_map = uv_smooth.uv_smooth(vis_2, key_map=key_map, threshold_PSI=threshold_PSI, ep=ep)
uv_im = np.flipud(uv_map.data)

# Run CLEAN on the full visibility set
clean_map = vis_clean(vis, shape=imsize, pixel_size=pixel, clean_beam_width=clean_beam_width)
clean_im = np.flipud(clean_map[0].data)  

# Run MEM_GE reconstruction over full dataset
snr_value, _ = resistant_mean((np.abs(vis.visibilities) / vis.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90)
mem_map = mem(vis, shape=imsize, pixel_size=pixel, percent_lambda=percent_lambda, map=False)
mem_im = np.flipud(mem_map.data)

# -------------------------
# GREEDY SELECTION: choose a subset of visibilities
# -------------------------

subset_greedy_idx_uv = gr.GreedyIP([0], vis, 'uv', pix, npix, n, plot_show,
                                   greedy_selection=greedy_selection,
                                   ep=ep, key_map=key_map, threshold_PSI=threshold_PSI)

# Build a Visibilities object containing only the selected subset 
vis_greedy_uv = Visibilities(
    visibilities=vis.visibilities[subset_greedy_idx_uv],
    u=vis.u[subset_greedy_idx_uv],
    v=vis.v[subset_greedy_idx_uv],
    amplitude=vis.amplitude[subset_greedy_idx_uv],
    amplitude_uncertainty=vis.amplitude_uncertainty[subset_greedy_idx_uv],
    meta=vis.meta
)


# # If using 'f-greedy', compute separate selections for MEM_GE and CLEAN maps.
if greedy_selection == 'f-greedy':

    subset_greedy_idx_mem_ge = gr.GreedyIP([0], vis, 'MEM_GE', pix, npix, n, plot_show,
                                          greedy_selection=greedy_selection,
                                          ep=ep, key_map=key_map, threshold_PSI=threshold_PSI,)

    vis_greedy_mem_ge = Visibilities(
        visibilities=vis.visibilities[subset_greedy_idx_mem_ge],
        u=vis.u[subset_greedy_idx_mem_ge],
        v=vis.v[subset_greedy_idx_mem_ge],
        amplitude=vis.amplitude[subset_greedy_idx_mem_ge],
        amplitude_uncertainty=vis.amplitude_uncertainty[subset_greedy_idx_mem_ge],
        meta=vis.meta
    )

    subset_greedy_idx_clean = gr.GreedyIP([0], vis, 'clean', pix, npix, n, plot_show,
                                          greedy_selection=greedy_selection,
                                          ep=ep, key_map=key_map, threshold_PSI=threshold_PSI)

    vis_greedy_clean = Visibilities(
        visibilities=vis.visibilities[subset_greedy_idx_clean],
        u=vis.u[subset_greedy_idx_clean],
        v=vis.v[subset_greedy_idx_clean],
        amplitude=vis.amplitude[subset_greedy_idx_clean],
        amplitude_uncertainty=vis.amplitude_uncertainty[subset_greedy_idx_clean],
        meta=vis.meta
    )

else:
    vis_greedy_clean = vis_greedy_uv
    vis_greedy_mem_ge = vis_greedy_uv
    subset_greedy_idx_mem_ge = subset_greedy_idx_uv
    subset_greedy_idx_clean = subset_greedy_idx_uv


# -------------------------
# RECONSTRUCT IMAGES FROM SELECTED SUBSETS
# -------------------------
# uv_smooth on the selected subset
vis_2 = Ku.duplicate_vis(vis_greedy_uv)
uv_map_sel = uv_smooth.uv_smooth(vis_2,key_map = key_map, threshold_PSI=threshold_PSI,ep = ep)
uv_im_gr = np.flipud(uv_map_sel.data)

# MEM_GE on the selected subset
snr_value, _ = resistant_mean((np.abs(vis_greedy_mem_ge.visibilities) / vis_greedy_mem_ge.amplitude_uncertainty).flatten(), 3)
percent_lambda = 2 / (snr_value**2 + 90)
mem_map_sel = mem(vis_greedy_mem_ge, shape=imsize, pixel_size=pixel, percent_lambda=percent_lambda, map=False)
mem_im_gr = np.flipud(mem_map_sel.data)

# CLEAN on selected subset
clean_map_sel = vis_clean(vis_greedy_clean, shape=imsize, pixel_size=pixel, clean_beam_width=clean_beam_width)
clean_im_gr = np.flipud(clean_map_sel[0].data)

# -------------------------
# CHI-SQUARED 
# -------------------------
npix = 128
pix = 1
F = sm.Fourier_matrix_STIX(vis.u.value, vis.v.value, npix, pix)
chi2_uv, rmse_uv, mean_err_uv = gr.compute_metrics(uv_im, vis, F)
chi2_MEM, rmse_MEM, mean_err_MEM = gr.compute_metrics(mem_im, vis, F)
chi2_CLEAN, rmse_clean, mean_err_clean = gr.compute_metrics(clean_im, vis, F)

# Compute chi-squared comparing predicted visibilities from each reconstructed image 
chi2_uv_greedy, rmse_uv_greedy, mean_err_uv_greedy  = gr.compute_metrics(uv_im_gr, vis, F)
chi2_mem_greedy, rmse_mem_greedy, mean_err_mem_greedy = gr.compute_metrics(mem_im_gr, vis, F)
chi2_clean_greedy, rmse_clean_greedy,mean_err_clean_greedy = gr.compute_metrics(clean_im_gr, vis, F)


# -------------------------
# PLOTTING (controlled by plot_show flag)
# -------------------------
if plot_show == 1:

    # Plot uv sampling (spiral) and its mirrored counterpart for visual symmetry
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u, v, marker='o', s=1.5**2, color='blue')
    ax.scatter(-u, -v, marker='o', s=1.5**2, color='blue')
    ax.set_title("(u,v) - Fibonacci Sampling")
    ax.set_xlabel(r"$[\text{arcsec}^{-1}]$")
    ax.set_ylabel(r"$[\text{arcsec}^{-1}]$")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Highlight selected points (from greedy uv-subset) in red on top of the total sampling
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u, v, c='blue', s=1.5**2, label='uv-smooth')
    ax.scatter(vis.u[subset_greedy_idx_uv], vis.v[subset_greedy_idx_uv], c='red', s=1.5**2, label='Selected points')
    ax.scatter(-u, -v, c='blue', s=1.5**2, label='uv-smooth')
    ax.scatter(-vis.u[subset_greedy_idx_uv], -vis.v[subset_greedy_idx_uv], c='red', s=1.5**2, label='Selected points')
    ax.set_title("(u,v) - Fibonacci Sampling Selected Points")
    ax.set_xlabel(r"$[\text{arcsec}^{-1}]$")
    ax.set_ylabel(r"$[\text{arcsec}^{-1}]$")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # If 'f-greedy' was used, plot mem_ge and clean selections similarly (here this block is skipped)
    if greedy_selection == 'f-greedy':
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(u, v, c='blue', s=1.5**2, label='mem_ge')
        ax.scatter(vis.u[subset_greedy_idx_mem_ge], vis.v[subset_greedy_idx_mem_ge], c='red', s=1.5**2, label='Selected points')
        ax.scatter(-u, -v, c='blue', s=1.5**2, label='mem_ge')
        ax.scatter(-vis.u[subset_greedy_idx_mem_ge], -vis.v[subset_greedy_idx_mem_ge], c='red', s=1.5**2, label='Selected points')
        ax.set_title("(u,v) - Fibonacci Sampling Selected Points")
        ax.set_xlabel(r"$[\text{arcsec}^{-1}]$")
        ax.set_ylabel(r"$[\text{arcsec}^{-1}]$")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(u, v, c='blue', s=1.5**2, label='clean')
        ax.scatter(vis.u[subset_greedy_idx_clean], vis.v[subset_greedy_idx_clean], c='red', s=1.5**2, label='Selected points')
        ax.scatter(-u, -v, c='blue', s=1.5**2, label='clean')
        ax.scatter(-vis.u[subset_greedy_idx_clean], -vis.v[subset_greedy_idx_clean], c='red', s=1.5**2, label='Selected points')
        ax.set_title("(u,v) - Fibonacci Sampling Selected Points")
        ax.set_xlabel(r"$[\text{arcsec}^{-1}]$")
        ax.set_ylabel(r"$[\text{arcsec}^{-1}]$")
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# Separate block that displays recovered maps if plotting is enabled
if plot_show == 1:
    # Comparison of reconstructions: ground truth, MEM (full), UV_SMOOTH (full & selected), CLEAN (full & selected)
    plt.figure(figsize=(5, 7))
    plt.suptitle("Comparison of Reconstructions", fontsize=14)

    plt.subplot(4, 2, 1)
    plt.imshow(gt_im, cmap = 'gist_heat')
    plt.title("GROUND TRUTH", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 5)
    plt.imshow(mem_im, cmap = 'gist_heat')
    plt.title("MEM_GE (tot points)", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 3)
    plt.imshow(uv_im, cmap = 'gist_heat')
    plt.title("UV_SMOOTH (tot points)", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 4)
    plt.imshow(uv_im_gr, cmap = 'gist_heat')
    plt.title("UV_SMOOTH (sel points)", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 6)
    plt.imshow(mem_im_gr, cmap = 'gist_heat')
    plt.title("MEM_GE (sel points)", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 7)
    plt.imshow(clean_im, cmap = 'gist_heat')
    plt.title("CLEAN (tot points)", fontsize=9)
    plt.axis('off')

    plt.subplot(4, 2, 8)
    plt.imshow(clean_im_gr, cmap = 'gist_heat')
    plt.title("CLEAN (selected)", fontsize=9)
    plt.axis('off')

    plt.show()
    

# -------------------------
# PRINT CHI-SQUARED SUMMARY
# -------------------------
print("===================== χ² =======================")
print("uv_smooth:                   χ² = {:.4f}".format(chi2_uv))
print("MEM_GE:                      χ² = {:.4f}".format(chi2_MEM))
print("CLEAN:                       χ² = {:.4f}".format(chi2_CLEAN))
print()
print("uv_smooth (greedy):          χ² = {:.4f}".format(chi2_uv_greedy))
print("MEM_GE (greedy):             χ² = {:.4f}".format(chi2_mem_greedy))
print("CLEAN (greedy):              χ² = {:.4f}".format(chi2_clean_greedy))
print()
print("=====================  rmse =======================")
print("uv_smooth:                   rmse = {:.4f}".format(rmse_uv))
print("MEM_GE:                      rmse = {:.4f}".format(rmse_MEM))
print("CLEAN:                       rmse = {:.4f}".format(rmse_clean))
print()
print("uv_smooth (greedy):          rmse = {:.4f}".format(rmse_uv_greedy))
print("MEM_GE (greedy)::            rmse = {:.4f}".format(rmse_mem_greedy))
print("CLEAN (greedy):              rmse = {:.4f}".format(rmse_clean_greedy))
print()
print("===================== mre =======================")
print("uv_smooth:                   mean_err = {:.4f}".format(mean_err_uv))
print("MEM_GE:                      mean_err = {:.4f}".format(mean_err_MEM))
print("CLEAN:                       mean_err = {:.4f}".format(mean_err_clean))
print()
print("uv_smooth (greedy):          mean_err = {:.4f}".format(mean_err_uv_greedy))
print("MEM_GE (greedy):             mean_err = {:.4f}".format(mean_err_mem_greedy))
print("CLEAN (greedy):              mean_err = {:.4f}".format(mean_err_clean_greedy))
