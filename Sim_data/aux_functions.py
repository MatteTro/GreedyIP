import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def uniform_nodes(n):
    """
    Generate a uniform grid of points in the unit square [0, 1] x [0, 1].

    Parameters
    ----------
    n : int
        Approximate total number of points. The grid has round(sqrt(n)) points
        per side (endpoints included), so the actual number of points is
        round(sqrt(n))**2.

    Returns
    -------
    u, v : ndarray, shape (round(sqrt(n))**2,)
        Flattened first and second coordinates of the grid points.
    """

    uv = np.linspace(0, 1, round(np.sqrt(n)), endpoint=True)
    u, v = np.meshgrid(uv, uv)

    return u.flatten(), v.flatten()


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


def k_matrix_matern(dsites, epoints, ep = 0.01):
    """
   Parameters
    ----------
    dsites : ndarray (N, d)
        Data locations where rhs is known.
    epoints : ndarray (M, d)
        Evaluation locations where we want the interpolant.
    ep : float
        Kernel shape parameter.
    """

    # Build kernel matrix between data sites
    DM_kernel = distance_matrix(dsites, epoints)
    K = np.exp(-ep * DM_kernel).T

    return K

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


def Fourier_matrix(u, v, n_pix, pix_size):
    """
    Function for creating the Fourier matrix to be used for computing the STIX visibility values from an image
    
    INPUTS:
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX
    
    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    n_pix: int
        number of pixels (rows/columns) used to discretize the considered Field-of-View (FOV)
    
    pix_size: float
        pixel size (in arcsec)
    """
    
    x = np.linspace(-(n_pix - 1)/2, (n_pix - 1)/2, num=n_pix)*pix_size
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, n_pix, axis=0)

    y = np.linspace((n_pix - 1)/2, -(n_pix - 1)/2, num=n_pix)*pix_size
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, n_pix, axis=1)
    
    dim = len(u)
    F = np.zeros((dim, n_pix*n_pix)).astype(np.complex128)
    
    for i in range(dim):
        
        phase = -2*np.pi*(x * u[i] + y * v[i])
        F[i, :]    = np.reshape(np.cos(phase), (n_pix*n_pix,)) + 1j * np.reshape(np.sin(phase), (n_pix*n_pix,))
        
    return F * pix_size**2


def Fourier_matrix_adjoint(u, v, n_pix, pix_size, pix_size_uv):
    """
    Function for creating the Fourier matrix to be used for computing the STIX visibility values from an image
    
    INPUTS:
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX
    
    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    n_pix: int
        number of pixels (rows/columns) used to discretize the considered Field-of-View (FOV)
    
    pix_size: float
        pixel size (in arcsec)
    """
    
    x = np.linspace(-(n_pix - 1)/2, (n_pix - 1)/2, num=n_pix)*pix_size
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, n_pix, axis=0)

    y = np.linspace((n_pix - 1)/2, -(n_pix - 1)/2, num=n_pix)*pix_size
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, n_pix, axis=1)
    
    dim = len(u)
    F = np.zeros((dim, n_pix*n_pix)).astype(np.complex128)
    
    for i in range(dim):
        
        phase = 2*np.pi*(x * u[i] + y * v[i])
        F[i, :]    = np.reshape(np.cos(phase), (n_pix*n_pix,)) + 1j * np.reshape(np.sin(phase), (n_pix*n_pix,))
        
    return F * pix_size_uv**2


def greedy_sure(u, v, re_V_noisy, im_V_noisy, re_V, im_V, nodes_grid, k_matrix, K_grid_all, sigma_re, sigma_im, norm_f_re, norm_f_im, 
                re_V_all, im_V_all, ep=10, alpha=1e-6):
    """
    P-greedy selection of nodes combined with Tikhonov-regularized kernel
    interpolation of noisy visibilities, monitoring SURE (Stein's Unbiased Risk
    Estimate), theoretical error bounds and errors w.r.t. the ground truth at
    every greedy iteration.

    At step j the interpolant is built on the j+1 nodes selected so far; the next
    node is the one (among the not yet selected) that maximizes the power function.

    Parameters
    ----------
    u, v : ndarray, shape (n_nodes,)
        Coordinates of all the available nodes (frequencies).
    re_V_noisy, im_V_noisy : ndarray, shape (n_nodes,)
        Noisy real and imaginary parts of the visibilities at the nodes.
    re_V, im_V : ndarray, shape (n_nodes,)
        Noise-free (ground truth) real and imaginary parts at the nodes.
    nodes_grid : ndarray, shape (n_grid**2, 2)
        Points of the fine grid; only its size is used here, to infer the number
        of points per side (n_grid).
    k_matrix : callable
        Function k_matrix(nodes_1, nodes_2, ep=...) returning the kernel matrix
        between two point sets (e.g. k_matrix_matern).
    K_grid_all : ndarray, shape (n_grid**2, n_nodes)
        Precomputed kernel matrix between the fine grid and all the nodes
        (the columns of the selected nodes are used at each step).
    sigma_re, sigma_im : float
        Standard deviation of the noise on the real and imaginary parts.
    norm_f_re, norm_f_im : float
        Norms of the real and imaginary parts of the target function, used in the
        power-function term of the error bound.
    re_V_all, im_V_all : ndarray, shape (n_grid, n_grid)
        Ground-truth real and imaginary parts on the fine grid.
    ep : float, optional
        Kernel shape parameter.
    alpha : float, optional
        Tikhonov regularization parameter added to the kernel matrix diagonal.

    Returns
    -------
    dict with keys:
        - 'sure' : SURE value at each iteration (real + imaginary parts), shape (n_nodes,)
        - 'error_bound' : total error bound (power function + Lebesgue terms)
        - 'error_bound_power' : power-function term of the error bound
        - 'error_bound_lebesgue' : Lebesgue-constant term of the error bound
        - 'mse_gt_nodes' : MSE w.r.t. the ground truth on the nodes
        - 'interp_func_re', 'interp_func_im' : interpolants on the fine grid at
          each iteration, shape (n_nodes, n_grid, n_grid)
        - 'selected_idx' : indices of the nodes in the order they were selected
        - 'mse_gt_grid' : MSE w.r.t. the ground truth on the fine grid
    """

    n_nodes = len(u) # number of nodes
    nodes_all = np.vstack((u, v)).T # All nodes

    n_nodes_grid = round(np.sqrt(np.shape(nodes_grid)[0]))

    # SURE
    residual_sure_re = np.zeros(n_nodes)
    residual_sure_im = np.zeros(n_nodes)
    trace_sure_re = np.zeros(n_nodes)
    trace_sure_im = np.zeros(n_nodes)

    #--------> TBD: error bound
    # ERROR BOUND
    error_bound_power = np.zeros(n_nodes)
    error_bound_lebesgue = np.zeros(n_nodes)

    # COMPARISON WITH GT IMAGE
    mse_gt_nodes = np.zeros(n_nodes) # on the set of nodes
    mse_gt_grid = np.zeros(n_nodes) # on a fine grid of points
    # ssim_gt_grid = np.zeros(n_nodes)

    # INTERPOLATED FUNCTIONS
    interp_func_re = np.zeros((n_nodes,n_nodes_grid,n_nodes_grid))
    interp_func_im = np.zeros((n_nodes,n_nodes_grid,n_nodes_grid))

    # INITIALIZE SET OF NODES FOR GREEDY
    training_index = [0]

    for j in range(n_nodes):

        if (j+1) % 50 == 0:
            print("Node #", j+1)

        # Define indices of test set of nodes
        test_index = np.setdiff1d(np.arange(n_nodes), training_index)
        # nodes_test = np.vstack((u[test_index], v[test_index])).T

        # Interpolation points in training set
        nodes_training = np.vstack((u[training_index], v[training_index])).T
        
        # Noisy interpolation values in training set
        re_V_noisy_training = re_V_noisy[training_index]
        im_V_noisy_training = im_V_noisy[training_index]
        
        # Distance matrices
        K_interp = k_matrix(nodes_training, nodes_training, ep = ep) # Only w.r.t. interpolation nodes
        K_tot = k_matrix(nodes_training, nodes_all, ep = ep) # W.r.t. all nodes
        K_grid = K_grid_all[:, training_index] #k_matrix(nodes_training, nodes_grid, ep = ep) # W.r.t. fine grids

        # Tikhonov matrix
        K_alpha = K_interp + alpha * np.eye(j+1)

        # Regularized interpolation
        coeff_re = np.linalg.solve(K_alpha, re_V_noisy_training)
        coeff_im = np.linalg.solve(K_alpha, im_V_noisy_training)

        R = np.linalg.solve(K_alpha, np.eye(K_interp.shape[0]))
        
        if j==0:
            pred_interp_nodes_re = K_tot[:,0] * coeff_re[0]
            pred_interp_nodes_im = K_tot[:,0] * coeff_im[0]
            trace_re = 2 * sigma_re**2 * np.sum(K_interp * R)
            trace_im = 2 * sigma_im**2 * np.sum(K_interp * R)
        
        else:
            pred_interp_nodes_re = K_tot @ coeff_re
            pred_interp_nodes_im = K_tot @ coeff_im
            trace_re = 2 * sigma_re**2 * np.sum( np.diag(K_interp @ R) )
            trace_im = 2 * sigma_im**2 * np.sum( np.diag(K_interp @ R) )

        
        # SURE
        residual_sure_re[j] = np.sum( (pred_interp_nodes_re - re_V_noisy)**2 ) / n_nodes
        residual_sure_im[j] = np.sum( (pred_interp_nodes_im - im_V_noisy)**2 ) / n_nodes

        trace_sure_re[j] = trace_re / n_nodes
        trace_sure_im[j] = trace_im / n_nodes

        # COMPARISON WITH GT IMAGE ON NODES
        mse_gt_nodes[j] =  np.sum( (pred_interp_nodes_re - re_V)**2 ) / n_nodes + np.sum( (pred_interp_nodes_im - im_V)**2 ) / n_nodes

        # Compute Power function and Lebesgue constant
        phi0 = 1.0                       
        power_func = np.zeros(n_nodes)
        lebesgue = np.zeros(n_nodes)

        tmp = R @ K_tot.T

        val1 = K_tot @ tmp

        val2 = tmp.T @ (K_interp @ tmp) 

        arg_sqrt_pow = phi0 - 2 * np.diag(val1) + np.diag(val2)
        power_func = np.real(np.sqrt(np.abs(arg_sqrt_pow)))
        lebesgue = np.real(np.sqrt(np.sum(tmp**2, axis=0)))

        ## ------------------->
        ## --------------------------------------> CHECK
        ## ------------------->

        error_bound_power[j] = 2 * norm_f_re**2 * np.sum( power_func**2 ) / n_nodes + 2 * norm_f_im**2 * np.sum( power_func**2 ) / n_nodes
        error_bound_lebesgue[j] = 2 * (j+1) * sigma_re**2 * np.sum(lebesgue**2) / n_nodes + 2 * (j+1) * sigma_im**2 * np.sum(lebesgue**2) / n_nodes
        
        # Compute interpolation error on finer grid    
        if j==0:
            pred_interp_fine_re = np.reshape(K_grid[:,0] * coeff_re[0], (n_nodes_grid,n_nodes_grid))
            pred_interp_fine_im = np.reshape(K_grid[:,0] * coeff_im[0], (n_nodes_grid,n_nodes_grid))
        
        else:
            pred_interp_fine_re = np.reshape(K_grid @ coeff_re, (n_nodes_grid,n_nodes_grid))
            pred_interp_fine_im = np.reshape(K_grid @ coeff_im, (n_nodes_grid,n_nodes_grid))
        
        interp_func_re[j,:,:] = pred_interp_fine_re
        interp_func_im[j,:,:] = pred_interp_fine_im

        mse_gt_grid[j] = np.sum( (pred_interp_fine_re - re_V_all)**2 ) / n_nodes + np.sum( (pred_interp_fine_im - im_V_all)**2 ) / n_nodes
        
        # Selection rule
        selection_rule = power_func[test_index]

        if (j+1) < n_nodes:
            # Choose the index that maximizes the selection rule
            index_max = np.argmax(selection_rule)

            # Map the local index back to the global index in the vis dataset
            index_max = test_index[index_max]
            # Append the chosen index to the training set
            training_index = np.hstack((training_index, index_max))
            # Remove newly selected indices from the test set
            test_index = np.setdiff1d(test_index, training_index)

    # SURE
    sure = (residual_sure_re + trace_sure_re - sigma_re**2) + (residual_sure_im + trace_sure_im - sigma_im**2)
    error_bound = error_bound_power + error_bound_lebesgue

    return {"sure": sure, 
            "error_bound": error_bound,
            "error_bound_power": error_bound_power,
            "error_bound_lebesgue": error_bound_lebesgue,
            "mse_gt_nodes": mse_gt_nodes,
            "interp_func_re": interp_func_re,
            "interp_func_im": interp_func_im,
            "selected_idx": training_index, 
            "mse_gt_grid": mse_gt_grid} 



def vsk_first_step(U, V, u_mask, v_mask, selected_idx_sure, re_v_selected, im_v_selected, F_grid_2d_masked, x_tik_alpha, N, mask):
    """
    First step of the VSK (Variably Scaled Kernels) reconstruction, with masking.

    The visibilities at the selected nodes are interpolated with a kernel defined
    on the augmented space (u, v, Re(V_tik), Im(V_tik)), where the two extra
    coordinates (the "scaling" functions) are the real and imaginary parts of the
    visibilities obtained from a preliminary Tikhonov reconstruction. The result
    is then restricted to the mask.

    Parameters
    ----------
    U, V : ndarray
        Coordinates of the evaluation points on the (N x N) frequency grid
        (flattened, N*N elements each).
    u_mask, v_mask : ndarray
        Coordinates of the frequencies inside the mask (candidate nodes).
    selected_idx_sure : array-like of int
        Indices (w.r.t. u_mask, v_mask) of the nodes selected by the greedy/SURE procedure.
    re_v_selected, im_v_selected : ndarray
        Real and imaginary parts of the visibilities at the selected nodes.
    F_grid_2d_masked : ndarray
        Fourier matrix mapping the image onto the frequency grid (used to compute
        the Tikhonov visibilities at all grid points).
    x_tik_alpha : ndarray
        Preliminary Tikhonov reconstruction of the image (flattened internally).
    N : int
        Number of grid points per side.
    mask : ndarray, shape (N, N)
        Mask (boolean/0-1) applied to the interpolated visibilities.

    Returns
    -------
    V_interp_mask_f : ndarray (complex), shape (N, N)
        Interpolated visibilities on the grid, set to zero outside the mask.
    """

    matrix =  np.column_stack((U, V))

    selected_u_f = u_mask[selected_idx_sure]
    selected_v_f = v_mask[selected_idx_sure]

    dsites = np.vstack((selected_u_f, selected_v_f)).T

    dsites_r =  np.hstack((dsites, re_v_selected.reshape(-1,1)))
    dsites_a = np.hstack((dsites_r, im_v_selected.reshape(-1,1)))

    v_tik_noisy = F_grid_2d_masked @ x_tik_alpha.flatten()

    tmp_real = np.real(v_tik_noisy) 
    tmp_imag = np.imag(v_tik_noisy) 
    epoints_r =  np.hstack((matrix,tmp_real.reshape(-1,1)))
    epoints_a = np.hstack((epoints_r,tmp_imag.reshape(-1,1)))

    ris1 = re_v_selected.reshape(-1,1)
    ris2 = im_v_selected.reshape(-1,1)

    regparam=1e-6 + np.zeros(dsites.shape[0])
    AAA = matern_kernel_interp(dsites_a, epoints_a, np.hstack((ris1,ris2)), ep=0.001, regparam=regparam)

    img_R_im = AAA['Pf1'][:,0]
    img_I_im = AAA['Pf1'][:,1]

    img_R_im = img_R_im.reshape(N, N)
    img_I_im = img_I_im.reshape(N, N)

    V_interp_greedy = img_R_im+1j*img_I_im

    V_interp_mask_f = V_interp_greedy*mask

    return V_interp_mask_f


def vsk_first_step_no_mask(U, V, u_mask, v_mask, selected_idx_sure, re_v_selected, im_v_selected, F_grid_2d, x_tik_alpha, N):
    """
    First step of the VSK (Variably Scaled Kernels) reconstruction, without masking.

    Same procedure as `vsk_first_step`: the visibilities at the selected nodes are
    interpolated with a kernel on the augmented space (u, v, Re(V_tik), Im(V_tik)),
    where the extra coordinates come from a preliminary Tikhonov reconstruction.
    Here the interpolated visibilities are returned on the whole grid, without
    applying any mask.

    Parameters
    ----------
    U, V : ndarray
        Coordinates of the evaluation points on the (N x N) frequency grid
        (flattened, N*N elements each).
    u_mask, v_mask : ndarray
        Coordinates of the frequencies from which the nodes were selected.
    selected_idx_sure : array-like of int
        Indices (w.r.t. u_mask, v_mask) of the nodes selected by the greedy/SURE procedure.
    re_v_selected, im_v_selected : ndarray
        Real and imaginary parts of the visibilities at the selected nodes.
    F_grid_2d : ndarray
        Fourier matrix mapping the image onto the whole frequency grid (used to
        compute the Tikhonov visibilities at all grid points).
    x_tik_alpha : ndarray
        Preliminary Tikhonov reconstruction of the image (flattened internally).
    N : int
        Number of grid points per side.

    Returns
    -------
    V_interp_mask_f : ndarray (complex), shape (N, N)
        Interpolated visibilities on the whole frequency grid (no mask applied).
    """

    matrix =  np.column_stack((U, V))

    selected_u_f = u_mask[selected_idx_sure]
    selected_v_f = v_mask[selected_idx_sure]

    dsites = np.vstack((selected_u_f, selected_v_f)).T

    dsites_r =  np.hstack((dsites, re_v_selected.reshape(-1,1)))
    dsites_a = np.hstack((dsites_r, im_v_selected.reshape(-1,1)))

    v_tik_noisy = F_grid_2d @ x_tik_alpha.flatten()

    tmp_real = np.real(v_tik_noisy) 
    tmp_imag = np.imag(v_tik_noisy) 
    epoints_r =  np.hstack((matrix,tmp_real.reshape(-1,1)))
    epoints_a = np.hstack((epoints_r,tmp_imag.reshape(-1,1)))

    ris1 = re_v_selected.reshape(-1,1)
    ris2 = im_v_selected.reshape(-1,1)

    regparam=1e-6 + np.zeros(dsites.shape[0])
    AAA = matern_kernel_interp(dsites_a, epoints_a, np.hstack((ris1,ris2)), ep=0.001, regparam=regparam)

    img_R_im = AAA['Pf1'][:,0]
    img_I_im = AAA['Pf1'][:,1]

    img_R_im = img_R_im.reshape(N, N)
    img_I_im = img_I_im.reshape(N, N)

    V_interp_mask_f = img_R_im+1j*img_I_im

    return V_interp_mask_f


def tik_svd(A):
    """Fattorizza una volta sola l'operatore (le righe nulle vengono scartate)."""
    rows = np.flatnonzero(np.any(A != 0, axis=1))
    U_, s, Vh = np.linalg.svd(A[rows], full_matrices=False)
    return {"rows": rows, "U": U_, "s": s, "V": Vh.conj().T}

def tik_solve(fac, y, alpha):
    """x = (A^H A + alpha I)^-1 A^H y, per qualunque alpha."""
    b = fac["U"].conj().T @ y[fac["rows"]]
    return fac["V"] @ (fac["s"] / (fac["s"]**2 + alpha) * b)








































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

def pgreedy_fourier_selection(u, v, n_points, eps=1.0, selected_idx=None, plot=False, symm=True):
    """
    p-greedy selection in frequency domain.

    Parameters
    ----------
    u, v : 1D arrays
        Frequency coordinates (from FFT grid)
    n_points : int
    eps : float
        RBF parameter (or Matern surrogate)
    start_idx : array-like
        initial seed
    """

    coords = np.vstack([u, v]).T

    N = len(coords)
    
    if selected_idx is None:
        selected_idx = []
    
        seed = 0  
        selected_idx.append(seed)
        
    if symm==True:
        selected_u = u[selected_idx]
        selected_v = v[selected_idx]
        
        conj_candidates = np.where(
            (np.isclose(u, 1-selected_u)) &
            (np.isclose(v, 1-selected_v))
        )[0]
        
        for j in conj_candidates:
            j = int(j)
            if j not in selected_idx:
                selected_idx.append(j)

    test_idx = list(set(range(N)) - set(selected_idx))

    history = []
    k = 0
    if symm==False:
        while len(selected_idx) < n_points:

            X_train = coords[selected_idx]
            X_test = coords[test_idx]

            P = maternPowerFunction(X_train, X_test, eps)

            k_max = np.argmax(P)

            best_idx = test_idx[k_max]
            
            u_best = u[best_idx]
            v_best = v[best_idx]
            
            history.append(P[k_max])

            selected_idx.append(best_idx)
            
            if best_idx in test_idx:
                test_idx.remove(best_idx)
            
            k = k+1
            print(
                f"[P - Greedy step {k}] "
                f"[Points selected: {len(selected_idx)}]"
                f"selected {best_idx:4d} -> ({u[best_idx]:.4f},{v[best_idx]:.4f}) ") 
            if k % 10 == 0:
    
                selected_u_f = u[selected_idx]
                selected_v_f = v[selected_idx]
            
                plt.figure(figsize=(6,6))
            
                plt.scatter(u, v, s=5, alpha=0.2, label='Mask')
            
                plt.scatter(
                    selected_u_f,
                    selected_v_f,
                    c='red',
                    s=20,
                    label='p-Greedy'
                )
            
                plt.axis('equal')
                plt.legend()
            
                plt.title(f"p-Greedy selection - step {k} (selected: {len(selected_idx)})")
            
                plt.show()
    if symm==True:
        while len(selected_idx) < n_points:
    
            X_train = coords[selected_idx]
            X_test = coords[test_idx]
    
            P = maternPowerFunction(X_train, X_test, eps)
    
            k_max = np.argmax(P)
    
            best_idx = test_idx[k_max]
            
            u_best = u[best_idx]
            v_best = v[best_idx]
            
            conj_candidates = np.where(
                (np.isclose(u, 1-u_best)) &
                (np.isclose(v, 1-v_best))
            )[0]
    
            history.append(P[k_max])
    
            selected_idx.append(best_idx)
            
            if best_idx in test_idx:
                test_idx.remove(best_idx)
            
            for conj_idx in conj_candidates:
            
                conj_idx = int(conj_idx)
            
                if conj_idx != best_idx:
            
                    if conj_idx not in selected_idx:
                        selected_idx.append(conj_idx)
            
                    if conj_idx in test_idx:
                        test_idx.remove(conj_idx)
            k = k+1
            print(
                f"[P - Greedy step {k}] "
                f"[Points selected: {len(selected_idx)}]"
                f"selected {best_idx:4d} -> ({u[best_idx]:.4f},{v[best_idx]:.4f}) "
                f"conjugate {conj_idx:4d} -> ({u[conj_idx]:.4f},{v[conj_idx]:.4f})")
            if k % 10 == 0:
    
                selected_u_f = u[selected_idx]
                selected_v_f = v[selected_idx]
            
                plt.figure(figsize=(6,6))
            
                plt.scatter(u, v, s=5, alpha=0.2, label='Mask')
            
                plt.scatter(
                    selected_u_f,
                    selected_v_f,
                    c='red',
                    s=20,
                    label='p-Greedy'
                )
            
                plt.axis('equal')
                plt.legend()
            
                plt.title(f"p-Greedy selection - step {k} (selected: {len(selected_idx)})")
            
                plt.show()
    return selected_idx, history




def fgreedy_fourier_selection( 
        N,
        u,
        v,
        V,
        matrix,
        selected_idx,
        n_points,
        mask,
        F_masked,
        symm=True,
        alfa=0.00000001,
        svd_mask=None,
        F_mask=None):
    """
    f-greedy selection in the frequency domain.

    At each iteration the visibilities at the nodes selected so far are
    interpolated with a VSK-type kernel (augmented space (u, v, Re(F_masked),
    Im(F_masked))), the result is inverted with a Tikhonov-regularized solver to
    obtain an image, and the image is mapped back to the frequency domain. The next
    node is the not yet selected one with the largest relative residual between
    the observed and the predicted visibilities.

    Parameters
    ----------
    N : int
        Number of grid points per side (the image is N x N).
    u, v : ndarray, shape (M,)
        Frequency coordinates of the candidate nodes.
    V : ndarray (complex), shape (M,)
        Observed visibilities at the candidate nodes.
    matrix : ndarray, shape (N*N, 2)
        Coordinates (u, v) of the evaluation points of the interpolant.
    selected_idx : list of int or None
        Indices of the nodes already selected. If None, the selection starts
        from the seed node with index 0.
    n_points : int
        Target number of selected nodes (the loop stops when it is reached).
    mask : ndarray, shape (N, N)
        Boolean mask applied to the interpolated visibilities.
    F_masked : ndarray (complex)
        Preliminary visibilities (e.g. from a Tikhonov reconstruction) whose real
        and imaginary parts are used as additional coordinates of the kernel.
    symm : bool, optional
        If True, the conjugate-symmetric node (u, v) -> (1-u, 1-v) is added
        together with each selected node.
    alfa : float, optional
        Tikhonov regularization parameter used in the image reconstruction step.
    svd_mask : dict
        SVD factorization of the Fourier operator, as returned by `tik_svd`.
    F_mask : ndarray (complex)
        Fourier matrix mapping the image to the visibilities at the candidate nodes.

    Returns
    -------
    selected_idx : list of int
        Indices of the selected nodes, in the order of selection.
    res_vec : list
        Empty list (never filled in this function; the residuals of each step are
        stored in a local `history` list that is not returned).
    """
    
    M = len(V)

    res_vec = []
    history = [] 

    if selected_idx is None:
        selected_idx = []
    
        seed = 0  
        selected_idx.append(seed)
        
    if symm==True:
        selected_u = u[selected_idx]
        selected_v = v[selected_idx]
        
        conj_candidates = np.where(
            (np.isclose(u, 1-selected_u)) &
            (np.isclose(v, 1-selected_v))
        )[0]
        
        for j in conj_candidates:
            j = int(j)
            if j not in selected_idx:
                selected_idx.append(j)
        
    test_idx = list(np.setdiff1d(np.arange(M),selected_idx))

    k = 0    

    if symm==False:
        while len(selected_idx) < n_points:

            selected_u = u[selected_idx]
            selected_v = v[selected_idx]
        
            dsites = np.vstack((selected_u, selected_v)).T
        
            dsites_r =  np.hstack((dsites, np.real(V[selected_idx].reshape(-1,1))))
            dsites_a = np.hstack((dsites_r,np.imag(V[selected_idx].reshape(-1,1))))
        
            epoints_r =  np.hstack((matrix,np.real(F_masked).reshape(-1,1)))
            epoints_a = np.hstack((epoints_r,np.imag(F_masked).reshape(-1,1)))
        
            ris1 = np.real(V[selected_idx].reshape(-1,1))
            ris2 = np.imag(V[selected_idx].reshape(-1,1))
        
            AAA = matern_kernel_interp(dsites_a, epoints_a, np.hstack((ris1,ris2)), ep=0.001)
        
            img_R_im = AAA['Pf1'][:,0]
            img_I_im = AAA['Pf1'][:,1]
            
            img_R_im = img_R_im.reshape(N, N)
            img_I_im = img_I_im.reshape(N, N)
        
        
            V_interp_greedy = img_R_im+1j*img_I_im
        
            V_interp_mask = V_interp_greedy*mask
            
            y_tik = V_interp_mask[mask]
            img_greedy_tik = np.reshape(np.real(tik_solve(svd_mask, y_tik, alfa)), (N, N))
                    
            V_step = F_mask @ img_greedy_tik.flatten()
            
            
            res = np.abs(V[test_idx] - V_step[test_idx])/np.abs(V[test_idx])
            
            k_max = np.argmax(res)
            
            best_idx = test_idx[k_max]

            history.append(res[k_max])

            selected_idx.append(best_idx)
            
            if best_idx in test_idx:
                test_idx.remove(best_idx)
            k = k+1
            print(
                f"[F - Greedy step {k}] "
                f"[Points selected: {len(selected_idx)}]"
                f"selected {best_idx:4d} -> ({u[best_idx]:.4f},{v[best_idx]:.4f}) ")
            if k % 10 == 0:

                selected_u_f = u[selected_idx]
                selected_v_f = v[selected_idx]
            
                plt.figure(figsize=(6,6))
            
                plt.scatter(u, v, s=5, alpha=0.2, label='Mask')
            
                plt.scatter(
                    selected_u_f,
                    selected_v_f,
                    c='red',
                    s=20,
                    label='f-Greedy'
                )
            
                plt.axis('equal')
                plt.legend()
            
                plt.title(f"f-Greedy selection - step {k} (selected: {len(selected_idx)})")
            
                plt.show()
    if symm==True:
        while len(selected_idx) < n_points:
    
            selected_u = u[selected_idx]
            selected_v = v[selected_idx]
        
            dsites = np.vstack((selected_u, selected_v)).T
        
            dsites_r =  np.hstack((dsites, np.real(V[selected_idx].reshape(-1,1))))
            dsites_a = np.hstack((dsites_r,np.imag(V[selected_idx].reshape(-1,1))))
        
            epoints_r =  np.hstack((matrix,np.real(F_masked).reshape(-1,1)))
            epoints_a = np.hstack((epoints_r,np.imag(F_masked).reshape(-1,1)))
        
            ris1 = np.real(V[selected_idx].reshape(-1,1))
            ris2 = np.imag(V[selected_idx].reshape(-1,1))
        
            AAA = matern_kernel_interp(dsites_a, epoints_a, np.hstack((ris1,ris2)), ep=0.001)
        
            img_R_im = AAA['Pf1'][:,0]
            img_I_im = AAA['Pf1'][:,1]
            
            img_R_im = img_R_im.reshape(N, N)
            img_I_im = img_I_im.reshape(N, N)
        
        
            V_interp_greedy = img_R_im+1j*img_I_im
        
            V_interp_mask = V_interp_greedy*mask
            
            y_tik = V_interp_mask[mask]
            img_greedy_tik = np.reshape(np.real(tik_solve(svd_mask, y_tik, alfa)), (N, N))
                    
            V_step = F_mask @ img_greedy_tik.flatten()
            
            
            res = np.abs(V[test_idx] - V_step[test_idx])/np.abs(V[test_idx])
            
            k_max = np.argmax(res)
            
            best_idx = test_idx[k_max]
            
            
            u_best = u[best_idx]
            v_best = v[best_idx]
            
            conj_candidates = np.where(
                (np.isclose(u, 1-u_best)) &
                (np.isclose(v, 1-v_best))
            )[0]
    
            history.append(res[k_max])
    
            selected_idx.append(best_idx)
            
            if best_idx in test_idx:
                test_idx.remove(best_idx)
            
            for conj_idx in conj_candidates:
            
                conj_idx = int(conj_idx)
            
                if conj_idx != best_idx:
            
                    if conj_idx not in selected_idx:
                        selected_idx.append(conj_idx)
            
                    if conj_idx in test_idx:
                        test_idx.remove(conj_idx)
            k = k+1
            print(
                f"[F - Greedy step {k}] "
                f"[Points selected: {len(selected_idx)}]"
                f"selected {best_idx:4d} -> ({u[best_idx]:.4f},{v[best_idx]:.4f}) "
                f"conjugate {conj_idx:4d} -> ({u[conj_idx]:.4f},{v[conj_idx]:.4f})")
            if k % 10 == 0:
    
                selected_u_f = u[selected_idx]
                selected_v_f = v[selected_idx]
            
                plt.figure(figsize=(6,6))
            
                plt.scatter(u, v, s=5, alpha=0.2, label='Mask')
            
                plt.scatter(
                    selected_u_f,
                    selected_v_f,
                    c='red',
                    s=20,
                    label='f-Greedy'
                )
            
                plt.axis('equal')
                plt.legend()
            
                plt.title(f"f-Greedy selection - step {k} (selected: {len(selected_idx)})")
            
                plt.show()
    return selected_idx, res_vec



def scores(V, V_pred, sigma=None):
    """
    Compute chi-square (or residual MSE), relative RMSE and MRE.

    Parameters
    ----------
    V : array-like (complex)
        True/observed visibilities.
    V_pred : array-like (complex)
        Predicted visibilities.
    sigma : None, scalar or array-like, optional
        Uncertainty on visibilities.

        If provided:
            chi_metric = reduced chi-square-like quantity

        If None:
            chi_metric = mean squared residual (MSE) in data space

    Returns
    -------
    chi_metric : float
        Reduced chi-square-like metric if sigma is provided,
        otherwise residual MSE.
    rmse : float
        Relative RMSE.
    mre : float
        Mean Relative Error.
    """
    
    # E = 1e-15

    V = np.asarray(V)
    V_pred = np.asarray(V_pred)

    # Residuals
    diff = V - V_pred

    diff_rel = np.abs(diff)#/np.maximum(np.abs(V),E)

    # Relative RMSE
    rmse = np.sqrt(np.mean(diff_rel**2))

    # Mean Relative Error
    mre = np.mean(diff_rel)

    # Chi-square or fallback metric
    if sigma is not None:
        sigma = np.asarray(sigma)
        chi_metric = np.mean((np.abs(diff) / sigma)**2)
    else:
        # No noise model available:
        # use residual MSE in Fourier/data space
        chi_metric = np.mean(np.abs(diff)**2)

    return chi_metric, rmse, mre


def compute_ssim(img_true, img_rec):
    """
    Compute the Structural Similarity Index (SSIM) between a reference image and
    a reconstructed one.

    Parameters
    ----------
    img_true : array-like
        Ground-truth (reference) image.
    img_rec : array-like
        Reconstructed image, same shape as `img_true`.

    Returns
    -------
    float
        SSIM value (1 means identical images). The data range is set to the
        dynamic range of `img_true` (max - min), as required by skimage.
    """
    
    img_true = np.asarray(img_true, dtype=float)
    img_rec = np.asarray(img_rec, dtype=float)

    # range dinamico richiesto da skimage
    data_range = img_true.max() - img_true.min()

    return ssim(img_true, img_rec, data_range=data_range)