import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import Kernel_utilities as Ku
from xrayvision.imaging import generate_header
from astropy import units as apu
from sunpy.map import Map

def uv_smooth(vis, imsize=(128, 128), pixel=(1., 1.), ep=0.1, threshold_PSI = 0.85, NOPLOT=1,  pix=1, key_map = 'bp'):
    """
    uv_smooth - reconstructs an image from sparse visibilities using:
      - interpolation (VSK-based) onto a regular u-v grid,
      - zero-padding/subsampling,
      - a projected Landweber iterative solver in the Fourier domain,
      - positivity constraint,
      - returns a SunPy Map containing the reconstructed image 

    Parameters
    ----------
    vis : object
        Visibility object expected to have attributes `u`, `v`, `visibilities`,
        `amplitude_uncertainty`, and metadata used by `generate_header`.
        u and v are expected in inverse-arcsec units (astropy Quantity or floats).
    imsize : tuple (nx, ny), optional
        Target image size in pixels (default (128,128)).
    pixel : tuple (px, py), optional
        Pixel size in arcsec (default (1., 1.)).
    ep : float, optional
        Shape parameter used in VSK interpolation (passed to Kernel_utilities).
    threshold_PSI : float, optional
        Threshold parameter used by the VSK interpolation (empirical).
    NOPLOT : int (0 or 1), optional
        If 0, show diagnostic plotting of the sampling; if 1, suppress plotting.
    key_map : str, optional
        Key describing the reference map to use inside the uv_smooth_vsk routine.

    Returns
    -------
    sunpy.map.Map
        A SunPy Map containing the reconstructed image
    """

    # -------------------------
    # Default numerical parameters (tuned to RHESSI-like setups)
    # -------------------------
    pixel_uv = 0.0005    # grid spacing in the u-v plane (units: arcsec^{-1})
    Nnew = 1920          # size of the zero-padded Fourier grid 
    Nmask = 15           # subsampling factor used when downsampling from the zero-padded grid
    iterLand = 50        # maximum Landweber iterations 
    tau = 0.5            # relaxation parameter for Landweber updates 
    N = 320              # size of the working u-v grid 

    if not NOPLOT:
        plt.figure(figsize=(5, 5))
        plt.scatter(vis.u, vis.v, c='blue', label='Visibilities', s=10)
        plt.xlabel('u (arcsec$^{-1}$)')
        plt.ylabel('v (arcsec$^{-1}$)')
        plt.title('Visibility Sampling in u-v plane')
        plt.legend()
        plt.show()

    # -------------------------
    # Define a uniform u-v sampling grid for interpolation
    # -------------------------
    Ulimit = (N / 2 - 1) * pixel_uv + pixel_uv / 2
    usampl = -Ulimit + np.arange(N) * pixel_uv
    vsampl = usampl

    # -------------------------
    # Interpolate measured visibilities onto the regular grid using kernels 
    # -------------------------
    # Ku.uv_smooth_vsk is expected to return a complex-valued N x N array representing the
    # interpolated visibility function on the regular grid (u,v).
    visnew = Ku.uv_smooth_vsk(vis, usampl, N, Ulimit, threshold_PSI, ep, imsize, pixel, pix, key_map)

    # -------------------------
    # Do not allow extrapolation outside the maximum baseline radius:
    # Do not allow function to extrapolate outside the disk
    Rmax = np.max(np.sqrt(vis.u**2 + vis.v**2)).value
    for i in range(N):
        for k in range(N):
            if np.sqrt(usampl[i]**2 + vsampl[k]**2) > Rmax:
                visnew[i, k] = 0. + 0j 

    # -------------------------
    # Zero-padding and subsampling preparation
    # -------------------------
    # Build a larger grid (Nnew x Nnew) and place the N x N visnew at its center
    Ulimit = (Nnew / 2 - 1) * pixel_uv + pixel_uv / 2
    xpix = -Ulimit + np.arange(Nnew) * pixel_uv
    ypix = xpix

    # Create zero-padded complex array and center the visnew inside it
    intzpadd = np.zeros((Nnew, Nnew), dtype=complex)
    start = (Nnew - N) // 2
    intzpadd[start:start + N, start:start + N] = visnew

    # -------------------------
    # Subsampling (masking) the zero-padded Fourier plane to build a lower-resolution image grid
    # -------------------------
    im_new = Nnew // Nmask                           # resulting image dimension after masking
    intznew = np.zeros((im_new, im_new), dtype=complex)
    xpixnew = np.zeros(im_new)                       # new coordinates after subsampling
    ypixnew = np.zeros(im_new)

    # Fill subsampled arrays by taking every Nmask-th point from the zero-padded grid
    for i in range(im_new):
        xpixnew[i] = xpix[Nmask * i]
        ypixnew[i] = ypix[Nmask * i]
        for j in range(im_new):
            intznew[i, j] = intzpadd[Nmask * i, Nmask * j]

    # -------------------------
    # Compute field-of-view (FOV) and sampling distances
    # -------------------------
    deltaomega = (xpixnew[im_new - 1] - xpixnew[0]) / im_new

    # -------------------------
    # Characteristic function chi (disk) in the Fourier domain
    # -------------------------
    chi = np.zeros((im_new, im_new), dtype=complex)
    for i in range(im_new):
        for j in range(im_new):
            if np.sqrt(xpixnew[i]**2 + ypixnew[j]**2) <= Rmax:
                chi[i, j] = complex(1., 0.)

    # -------------------------
    # Prepare arrays for the projected Landweber iteration
    # -------------------------
    map_actual = np.zeros((im_new, im_new), dtype=complex)   # current image estimate in Fourier domain
    map_iteration = np.zeros((iterLand, im_new, im_new))     # store iterates for possible backtracking
    map_solution = np.zeros((im_new, im_new))                # final real-valued solution
    descent = np.zeros(iterLand - 1)                        # relative descent measure between iterations
    normAf_g = np.zeros(iterLand)                           # residual norms for stopping criterion

    # -------------------------
    # Initialize iteration: inverse FFT of the initial map
    # -------------------------
    map_shifted = np.roll(map_actual, shift=[-im_new // 2, -im_new // 2], axis=(0, 1))
    # Compute the inverse Fourier transform. The normalization factor depends on the
    # chosen Fourier conventions
    F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi * np.pi * deltaomega * deltaomega)
    F_Trasf = np.roll(F_Trasf_shifted, shift=[im_new // 2, im_new // 2], axis=(0, 1))

    # -------------------------
    # Landweber iterative solver
    # -------------------------
    for iter in range(iterLand):
        # Landweber update in the Fourier domain: F^{k+1} = F^{k} + tau * (intznew - chi * F^{k})
        F_Trasf_up = F_Trasf + tau * (intznew - chi * F_Trasf)
        F_Trasf = F_Trasf_up
        
        F_Trasf_shifted = np.roll(F_Trasf, shift=[-im_new // 2, -im_new // 2], axis=(0, 1))
        # Forward transform 
        map_shifted = fft2(F_Trasf_shifted) * 4 * np.pi * np.pi * deltaomega * deltaomega
        map_actual = np.roll(map_shifted, shift=[im_new // 2, im_new // 2], axis=(0, 1))

        # Enforce positivity
        for i in range(im_new):
            minzero = np.where(map_actual[0:im_new, i].real < 0)[0]
            if minzero.size > 0:
                map_actual[minzero, i] = complex(0., 0.)

        # Store real part of iterate 
        map_iteration[iter, :, :] = map_actual.real

        map_shifted = np.roll(map_actual, shift=[-im_new // 2, -im_new // 2], axis=(0, 1))
        F_Trasf_shifted = ifft2(map_shifted) / (4 * np.pi * np.pi * deltaomega * deltaomega)
        F_Trasf = np.roll(F_Trasf_shifted, shift=[im_new // 2, im_new // 2], axis=(0, 1))

        # Compute residual Af_g = chi * F_Trasf - intznew and its norm for stopping criterion
        Af_g = chi * F_Trasf - intznew
        normAf_g[iter] = np.sqrt(np.sum(np.abs(Af_g)**2))

        # Compute relative descent and stop early 
        if iter >= 1:
            descent[iter - 1] = (normAf_g[iter - 1] - normAf_g[iter]) / normAf_g[iter - 1]
            if descent[iter - 1] < 0.02:
                break
    # -------------------------
    if iter == iterLand:
        map_solution[:, :] = map_iteration[14, :, :]
    else:
        map_solution = map_actual.real

    # -------------------------
    # Build a SunPy Map for the reconstructed image and generate header metadata
    # -------------------------
    
    im = map_solution
    shape = imsize * apu.pixel
    pixel_size = pixel * apu.arcsec / apu.pixel
    header = generate_header(vis, shape=shape, pixel_size=pixel_size)

    return Map((im.T, header))
