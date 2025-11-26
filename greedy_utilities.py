import numpy as np
import matplotlib.pyplot as plt
from astropy import units as apu
from xrayvision.mem import mem, resistant_mean
from xrayvision.visibility import Visibilities
from xrayvision.clean import vis_clean
import Kernel_utilities as Ku
import uv_smooth as uv_smooth

import simulator as sm

def GreedyIP(starting_index, vis, method, pix, npix, n_points, plot_show,
             greedy_selection, ep, key_map, threshold_PSI):
    """
    GreedyIP - greedy incremental selection of visibility indices.

    The function incrementally selects visibilities according to a selection rule:
      - if greedy_selection == 'f-greedy', predict visibilities from a reconstruction
        (CLEAN, MEM_GE or UV) and pick the test index whose predicted visibility
        deviates most from the measured one;
      - otherwise compute a kernel-based power-function criterion 
        between already-selected uv points and candidate test points.

    Parameters
    ----------
    starting_index : array-like
        Initial indices already included (seed for greedy selection).
    vis : Visibilities object
        Full visibility dataset (must have attributes u, v, visibilities, amplitude, amplitude_uncertainty, meta).
    method : str
        Reconstruction method used inside f-greedy: 'MEM_GE', 'clean', or 'uv'.
    imsize : tuple
        Image size passed to reconstruction routines.
    pixel : tuple or Quantity
        Pixel size passed to reconstructions.
    n_points : int
        Number of visibilities to select (termination condition).
    plot_show : int (0/1)
        If 1, show diagnostic plot of the selection metric over iterations.
    greedy_selection : str
        Choice of greedy strategy ('f-greedy' triggers reconstruction-based selection).
    ep : float
        Kernel/lengthscale parameter used by power-function selection.
    key_map : str
        Map key passed to uv_smooth (if used).
    threshold_PSI : float
        Threshold parameter passed to uv_smooth 

    Returns
    -------
    indices : ndarray
        Array of selected visibility indices (same ordering as vis.u / vis.v).
    """
    imsize = [npix, npix] * apu.pixel               # shape with astropy unit (passed to xrayvision functions)
    pixel = [pix, pix] * apu.arcsec / apu.pixel     # pixel size as astropy Quantity (arcsec/pixel)

    # Initialize training set (selected indices) and test set (candidate indices)
    training_index = starting_index
    test_index = np.setdiff1d(np.arange(vis.u.shape[0]), training_index)
    # Store maximum selection metric value at each iteration 
    max_values = []
    # Temporary index array for iteration count
    test_index_tmp = np.arange(len(test_index))

    # Greedy selection loop: at each iteration pick one new index according to selection_rule
    for k in range(len(test_index_tmp)):
        # Build a Visibilities object containing only currently selected visibilities.
        greedy_vis = Visibilities(
            visibilities=vis.visibilities[training_index],
            u=vis.u[training_index],
            v=vis.v[training_index],
            amplitude=vis.amplitude[training_index],
            amplitude_uncertainty=vis.amplitude_uncertainty[training_index],
            meta=vis.meta
        )

        # If using f-greedy, reconstruct an image from the current subset and use it to predict missing vis
        if greedy_selection == 'f-greedy':
            if method == 'MEM_GE':
                snr_value, _ = resistant_mean(
                    (np.abs(greedy_vis.visibilities) / greedy_vis.amplitude_uncertainty).flatten(), 3)
                percent_lambda = 2 / (snr_value**2 + 90)
                mem_map = mem(greedy_vis, shape=imsize, pixel_size=pixel, percent_lambda=percent_lambda, map=False)
                rec_image = np.flipud(mem_map.data)
            elif method == 'clean':
                beam_width = 12.
                clean_beam_width = beam_width * apu.arcsec
                clean_map = vis_clean(vis, shape=imsize, pixel_size=pixel, clean_beam_width=clean_beam_width)
                rec_image = np.flipud(clean_map[0].data)
            elif method == 'uv':
                vis_2 = Ku.duplicate_vis(greedy_vis)
                uv_map = uv_smooth.uv_smooth(vis_2, key_map=key_map, threshold_PSI=threshold_PSI, ep=ep)
                rec_image = np.flipud(uv_map.data)

        # Compute selection metric depending on strategy:
        if greedy_selection == 'f-greedy':

            fourier_matrix = sm.Fourier_matrix_STIX(vis.u.value[test_index], vis.v.value[test_index], npix, pix)
            
            vis_pred = fourier_matrix @ rec_image.reshape(-1)
            
            # Convert back to complex visibilities
            vis_pred = vis_pred.reshape(-1) * apu.ct / (apu.s * apu.cm**2 * apu.keV)
            vis_pred = vis_pred[0:len(vis_pred)//2] + 1j * vis_pred[len(vis_pred)//2:]

            #diff = np.abs((vis.visibilities.value - vis_pred.visibilities.value) / vis.visibilities.value) #/ vis.amplitude_uncertainty.value
            selection_rule = np.abs((vis.visibilities[test_index].value - vis_pred.value))/ np.abs(vis.visibilities[test_index].value) #/ vis.amplitude_uncertainty.value
            # Selection rule: absolute difference between predicted and measured visibilities (magnitude)
            #selection_rule = abs(vis_pred.visibilities.value - vis.visibilities[test_index].value)
        else:
            # Alternative selection: use kernel-based power function computed on uv coordinates
            vis_2 = Ku.duplicate_vis(greedy_vis)
            vis_uv_pf = np.vstack((vis_2.u, vis_2.v)).value.T
            vis_uv_pf_test = np.vstack((vis.u[test_index], vis.v[test_index])).value.T
            selection_rule = Ku.maternPowerFunction(vis_uv_pf, vis_uv_pf_test, ep)

        # Choose the index that maximizes the selection rule
        index_max = np.argmax(selection_rule)
        max_sel_rule = selection_rule[index_max]
        max_values.append(max_sel_rule)

        # Map the local index back to the global index in the vis dataset
        index_max = test_index[index_max]
        # Append the chosen index to the training set
        training_index = np.hstack((training_index, index_max))
        # Remove newly selected indices from the test set
        test_index = np.setdiff1d(test_index, training_index)
        test_index_tmp = np.arange(len(test_index))

        # Terminate when desired number of points selected
        if len(training_index) == n_points:
            break

    # Final selected indices
    indices = training_index

    if plot_show == 1:
        plt.plot(range(1, len(max_values) + 1), max_values, marker='o', linestyle='-', color='blue')
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel(r'$\chi^2$')
        plt.title('Evolution of the $\chi^2$ as a function of the iterations')
        plt.grid(True)
        plt.show()

    return indices


def compute_metrics(reconstructed_image, vis, fourier_matrix):
    """
    Compute a chi-squared metric between a reconstructed image and observed visibilities.

    Parameters
    ----------
    reconstructed_image : SunPy Map-like or object with `.data` attribute (2D array)
        The image object used for forward-modeling.
    vis : Visibilities
        Observed visibilities with uncertainties.

    Returns
    -------
    chi2 : float
        Computed chi-squared-like scalar (sum of squared normalized residuals / N).
    """
    
    vis_pred = fourier_matrix @ reconstructed_image.reshape(-1)
    
    # Convert back to complex visibilities
    vis_pred = vis_pred.reshape(-1) * apu.ct / (apu.s * apu.cm**2 * apu.keV)
    vis_pred = vis_pred[0:len(vis_pred)//2] + 1j * vis_pred[len(vis_pred)//2:]

    #diff = np.abs((vis.visibilities.value - vis_pred.visibilities.value) / vis.visibilities.value) #/ vis.amplitude_uncertainty.value
    diff = vis.visibilities.value - vis_pred.value
    diff_rel = np.abs(diff) / np.abs(vis.visibilities.value) 
    diff_chi = np.abs(diff) / np.abs(vis.amplitude_uncertainty.value)
    # chi2 as mean squared normalized residual
    chi2 = np.sum((diff_chi**2)) / len(vis.visibilities)
    rmse = np.sqrt(np.sum((diff_rel**2)) / len(vis.visibilities))
    mre = np.mean(np.abs(diff_rel))
    return chi2, rmse, mre


def index_to_subcollimator_name(indices):
    """
    Convert numerical subcollimator indices to STIX-like naming convention.

    The mapping used is:
      - group = 10 - floor(idx / 3)
      - letter cycles among ['a', 'b', 'c'] via idx % 3

    Example:
      index 0 -> '10a'
      index 3 -> '9a'

    Parameters
    ----------
    indices : iterable of ints
        Indices to convert.

    Returns
    -------
    subcollimators : list of str
        Human-readable subcollimator names.
    """
    subcollimators = []
    for idx in indices:
        group = 10 - idx // 3
        letter = ['a', 'b', 'c'][idx % 3]
        subcollimators.append(f"{group}{letter}")

    return subcollimators
