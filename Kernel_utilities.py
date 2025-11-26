import numpy as np
from astropy import units as apu
from xrayvision.imaging import image_to_vis, vis_to_image
from xrayvision.imaging import Visibilities
from xrayvision.clean import vis_clean


def distance_matrix(dsites, ctrs):
    """
    Compute the Euclidean distance matrix between two point sets.

    Parameters
    ----------
    dsites : ndarray, shape (M, s)
        Source points (rows are points, columns are coordinates).
    ctrs : ndarray, shape (N, s)
        Target / center points.

    Returns
    -------
    DM : ndarray, shape (M, N)
        Pairwise Euclidean distances DM[i, j] = || dsites[i,:] - ctrs[j,:] ||_2.
    """
    M, _ = dsites.shape
    N, s = ctrs.shape
    DM = np.zeros((M, N))

    for d in range(s):
        dr, cc = np.meshgrid(dsites[:, d], ctrs[:, d], indexing='ij')
        DM += (dr - cc) ** 2

    return np.sqrt(DM)


def maternPowerFunction(dsites, epoints, ep):
    """
    Evaluate power function at evaluation points.

    Parameters
    ----------
    dsites : ndarray
        Data sites (NxD).
    epoints : ndarray
        Evaluation points (MxD).
    ep : float
        Kernel shape parameter (lengthscale).

    Returns
    -------
    powfun : ndarray, shape (M,1)
        Estimated power-function values at epoints.
    """
    
    dsites = normalize_data(dsites, data1=dsites, second_set=None)
    epoints = normalize_data(epoints, data1=epoints, second_set=None)

    # Distances between evaluation points and data sites (M x N)
    DM_eval = distance_matrix(epoints, dsites)

    # Distances between data sites (N x N)
    DM_data = distance_matrix(dsites, dsites)

    # Exponential kernel matrices (Matérn-like simple exponential)
    IM = np.exp(-ep * DM_data)       # interpolation matrix (N x N)
    EM = np.exp(-ep * DM_eval)       # evaluation matrix (M x N)

    phi0 = 1.0                       
    powfun = np.zeros((np.shape(epoints)[0], 1))

    # Compute power-function expression for kernel interpolation.
    for j in range(np.shape(epoints)[0]):
        row = EM[j, :]                
        z = np.linalg.solve(IM, row.T)  
        val = z @ row                 
        powfun[j] = np.real(np.sqrt(phi0 - val))

    return powfun


def normalize_data(data, data1=None, second_set=None):
    """
    Normalize/rescale `data` to [0,1] using specified reference sets.

    """
    if data1 is None:
        data1 = data
        if second_set is not None:
            # scale data to the dynamic range of second_set
            data_rescaled = data * np.max(second_set - np.min(second_set)) + np.min(second_set)
        else:
            # standard min-max normalization using data1
            data_rescaled = (data - np.min(data1)) / np.max(data1 - np.min(data1))
    else:
        if second_set is not None:
            data_rescaled = data * np.max(second_set - np.min(second_set)) + np.min(second_set)
        else:
            data_rescaled = (data - np.min(data1)) / np.max(data1 - np.min(data1))

    return data_rescaled


def matern_kernel_interp(dsites, epoints, rhs, ep, regparam=None, multi_set=None):
    """
    Kernel interpolation using an exponential kernel phi(r) = exp(-ep * r).

    Solves IM * coef = rhs for interpolation coefficients and evaluates the interpolant
    at new points.

    Parameters
    ----------
    dsites : ndarray (N, d)
        Data locations where rhs is known.
    epoints : ndarray (M, d)
        Evaluation locations where we want the interpolant.
    rhs : ndarray (N, nvec)
        Right-hand sides (e.g., real and imaginary parts stacked column-wise).
    ep : float
        Kernel shape parameter.
    regparam : ndarray or scalar, optional
        Diagonal regularization added to IM for numerical stability (Tikhonov-like).
    multi_set : ndarray, optional
        Additional set of points at which to evaluate the interpolant (returned as 'Pf2').

    Returns
    -------
    Pf : dict
        Dictionary containing:
          - 'Pf1' : interpolated values at epoints (shape M x nvec)
          - 'Pf2' : interpolated values at multi_set (if provided)
    """
    
    if multi_set is None:
        multi_set = 0
    if regparam is None:
        regparam = np.zeros(dsites.shape[0])

    # Build kernel matrix between data sites
    DM_kernel = distance_matrix(dsites, dsites)
    IM = np.exp(-ep * DM_kernel)

    # Add diagonal regularization (modified behavior)
    IM += np.diag(regparam)

    # Solve linear systems 
    coef = np.zeros((IM.shape[1], rhs.shape[1]))
    for i in range(rhs.shape[1]):
        coef[:, i] = np.linalg.solve(IM, rhs[:, i])

    # Build evaluation kernel between data sites and evaluation points
    DM = distance_matrix(dsites, epoints)   
    EM = np.exp(-ep * DM)                   

    # Pf1 will contain interpolated values at epoints
    Pf_1 = np.zeros((EM.shape[1], rhs.shape[1]))  

    # For each RHS column, evaluate EM^T @ coef
    for nsys in range(rhs.shape[1]):
        Pf_1[:, nsys] = EM.T @ coef[:, nsys]

    Pf = {'Pf1': Pf_1}

    # If multi_set is provided, evaluate interpolant at those points as well
    if np.max(multi_set) != 0:
        NN1 = multi_set.shape[0]
        Pf_2 = np.zeros((NN1, rhs.shape[1]))
        DM = distance_matrix(dsites, multi_set)
        EM = np.exp(-ep * DM)
        for nsys in range(rhs.shape[1]):
            Pf_2[:, nsys] = EM.T @ coef[:, nsys]
        Pf = {'Pf1': Pf_1, 'Pf2': Pf_2}

    return Pf


def uv_smooth_augmented_feature(map_coarse_tres, Ulimit, grid_matrix, vis_matrix, ep, dx, dy, vis_wh):
    """
    Construct augmented feature matrices for VSK interpolation.

    Parameters
    ----------
    map_coarse_tres : ndarray
        Thresholded coarse image (2D).
    Ulimit : float
        Maximum u (positive) coordinate for the small uv-grid.
    grid_matrix : ndarray
        Coordinates of the full grid (flattened pairs).
    vis_matrix : ndarray
        Measured uv coordinates (N x 2).
    ep : float
        Kernel parameter forwarded to matern interpolation.
    dx, dy : float
        Pixel sizes (unused here except passed through).
    vis_wh : visibility object
        Original visibility object (used to compute image_to_vis).
    """
    
    N_tmp = 24
    u_vals = np.repeat(-Ulimit + np.arange(N_tmp) * (2 * Ulimit / N_tmp), N_tmp)
    v_vals = np.tile(-Ulimit + np.arange(N_tmp) * (2 * Ulimit / N_tmp), N_tmp)
    uv_grid = np.vstack((u_vals, v_vals)).T / apu.arcsec

    # Compute visibilities of the thresholded coarse map at the uv_grid locations
    vis_data = image_to_vis(map_coarse_tres * apu.ct / apu.arcsec**2, u=uv_grid[:, 0], v=uv_grid[:, 1])

    # Normalize real and imaginary parts
    real_norm = normalize_data(vis_data.visibilities.real)
    imag_norm = normalize_data(vis_data.visibilities.imag)
    rhs = np.vstack((real_norm.value, imag_norm.value)).T

    # Interpolate using matern kernel
    interpolated = matern_kernel_interp((uv_grid.value), (grid_matrix), rhs, ep, multi_set=(vis_matrix.value))

    # Build augmented feature matrices:
    # - grid_matrix_vsk: original spatial coordinates + normalized predicted Pf1 values
    # - vis_matrix_vsk: original vis coordinates + normalized predicted Pf2 values
    grid_matrix_vsk = np.column_stack(((grid_matrix), normalize_data(interpolated['Pf1'], data1=interpolated['Pf1'])))
    vis_matrix_vsk = np.column_stack(((vis_matrix.value), normalize_data(interpolated['Pf2'], data1=interpolated['Pf1'])))

    return {
        'vis_matrix_vsk': vis_matrix_vsk,
        'grid_matrix_vsk': grid_matrix_vsk
    }


def uv_smooth_vsk(vis, usampl, N, Ulimit, threshold_PSI, ep, imsize, pixel, pix, key_map):
    """
    Interpolate sparse visibilities onto a regular u-v grid using an augmented VSK approach.

    Parameters
    ----------
    vis : visibility-like object
    usampl : 1D array
        u sampling points for the uniform grid
    N : int
        number of grid points per axis
    Ulimit : float
    threshold_PSI : float
    ep : float
    imsize : tuple (nx, ny)
    pixel : tuple
    pix : float
    key_map : str
    """
   
    vis_matrix = np.column_stack((vis.u, vis.v))

    # Normalize real and imaginary parts of measured visibilities
    vis_real = normalize_data(vis.visibilities.real)
    vis_imag = normalize_data(vis.visibilities.imag)

    # Create grid matrix (flattened coordinates) of size N*N x 2
    X, Y = np.meshgrid(usampl, usampl)
    grid_matrix = np.column_stack((X.ravel(), Y.ravel()))

    # Regularization parameters for the kernel system (zero by default)
    regparam = np.zeros(vis_matrix.shape[0])

    # Convert imsize and pixel to astropy Quantities for xrayvision routines
    image_size = imsize * apu.pixel
    pixel_scale = [pix, pix] * apu.arcsec / apu.pixel
    beam_width = 12.
    clean_beam_width = beam_width * apu.arcsec

    # Compute coarse_map either from CLEAN or from direct inversion (vis_to_image)
    if key_map != 'bp':
        threshold_PSI = 0.
        clean_map = vis_clean(vis, shape=image_size, pixel_size=pixel_scale, clean_beam_width=clean_beam_width)
        coarse_map = clean_map[1].data
    else:
        coarse_map = vis_to_image(vis, image_size, pix * pixel_scale, "natural").data

    # Apply thresholding to keep strong features (threshold_PSI fraction of maximum)
    threshold = threshold_PSI * np.max(coarse_map)
    map_thresh = np.where(np.abs(coarse_map) >= threshold, coarse_map, 0)

    # Build augmented features using the coarse thresholded map
    features = uv_smooth_augmented_feature(map_thresh, Ulimit, grid_matrix, vis_matrix, ep, pixel[0], pixel[0], vis)

    # Perform kernel interpolation
    interp = matern_kernel_interp(
        features['vis_matrix_vsk'],
        features['grid_matrix_vsk'],
        np.column_stack((vis_real, vis_imag)),
        ep,
        regparam=regparam
    )

    
    reintz = normalize_data(interp['Pf1'][:, 0], second_set=vis.visibilities.real.value.reshape((-1, 1))) / (4 * np.pi**2)
    imintz = normalize_data(interp['Pf1'][:, 1], second_set=vis.visibilities.imag.value.reshape((-1, 1))) / (4 * np.pi**2)

    # Reshape into N x N grid and combine real + imaginary parts into complex visibility grid.
    vis_new = (reintz.reshape((N, N)).T + 1j * imintz.reshape((N, N)).T)
    return vis_new


def duplicate_vis(vis):
    """
    Duplicate a visibility object by concatenating conjugate visibilities and mirrored uv coordinates.

    This is useful when algorithms expect symmetric (u, v) coverage (positive and negative frequencies).

    Parameters
    ----------
    vis : xrayvision-like Visibilities object
        Expected fields: vis.visibilities, vis.amplitude_uncertainty, vis.u, vis.v, vis.meta

    Returns
    -------
    out_vis : Visibilities
        New Visibilities object containing [V, conj(V)] and [u, -u], [v, -v] etc.
    """
    obsvis = vis.visibilities
    sigamp = vis.amplitude_uncertainty
    u = vis.u
    v = vis.v
    meta = vis.meta

  
    obsvis_dup = np.concatenate([obsvis, np.conj(obsvis)])
    u_dup = np.concatenate([u, -u])
    v_dup = np.concatenate([v, -v])
    sigamp_dup = np.concatenate([sigamp, sigamp])

    # Create a new Visibilities object 
    out_vis = Visibilities(
        obsvis_dup,
        u=u_dup,
        v=v_dup,
        amplitude=np.abs(obsvis_dup),
        amplitude_uncertainty=sigamp_dup,
        meta=meta,
    )

    return out_vis
