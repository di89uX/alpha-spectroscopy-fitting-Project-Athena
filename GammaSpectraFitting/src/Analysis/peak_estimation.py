# File: src/analysis/peak_estimation.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def estimate_initial_params(x, y, **kwargs):
    """
    Estimates initial guess parameters (A0, mu0, sigma0) for a peak 
    by finding the FWHM of the most prominent peak in the spectrum.
    
    Returns: mu0, A0, sigma0, fwhm
    """
    # --- Begin of function body ---
    peak_index = kwargs.get('peak_index', None)
    smooth_sigma = kwargs.get('smooth_sigma', 2)
    min_sep = kwargs.get('min_sep', 1)
    min_energy_threshold = kwargs.get('min_energy_threshold', 200)
    prominence_factor = kwargs.get('prominence_factor', 0.1)
    height_factor = kwargs.get('height_factor', 0.05)
    
    # Ensure numpy arrays and ascending x
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    
    # Smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        y_s = gaussian_filter1d(y, smooth_sigma)
    else:
        y_s = y.copy()

    if peak_index is None:
        # Mask low-energy region
        mask = x > min_energy_threshold
        if not np.any(mask):
            raise ValueError(f"No data above {min_energy_threshold} keV. Adjust threshold.")
        
        # Find peaks in smoothed counts
        prominence = prominence_factor * np.max(y_s[mask])
        height = height_factor * np.max(y_s[mask])
        peaks, properties = find_peaks(y_s, prominence=prominence, height=height)
        
        if len(peaks) == 0:
            raise ValueError("No peaks found. Adjust prominence_factor or height_factor.")
        
        # Select the tallest peak
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        peak_index = peaks[sorted_indices[0]]
    
    mu0 = x[peak_index]
    A0 = y_s[peak_index] - np.median(y) 
    half = (y_s[peak_index] + np.median(y)) / 2.0 

    x_left_cross = None
    x_right_cross = None

    try:
        left_ix = np.where(x < mu0)[0]
        right_ix = np.where(x > mu0)[0]
        if left_ix.size > 1:
            xl = x[left_ix]
            yl = y_s[left_ix]
            mask = yl <= half
            if mask.any():
                idx = np.where(mask)[0][-1]
                if idx < len(xl)-1:
                    f = interp1d(yl[idx:idx+2], xl[idx:idx+2])
                    x_left_cross = float(f(half))
        if right_ix.size > 1:
            xr = x[right_ix]
            yr = y_s[right_ix]
            mask = yr <= half
            if mask.any():
                idx = np.where(mask)[0][0]
                if idx > 0:
                    f = interp1d(yr[idx-1:idx+1], xr[idx-1:idx+1])
                    x_right_cross = float(f(half))
    except Exception:
        x_left_cross = x_right_cross = None

    if x_left_cross is None:
        i = peak_index
        while i > 0 and y_s[i] > half:
            i -= 1
        if i >= 0 and i < peak_index:
            x0, x1 = x[i], x[i+1]
            y0, y1 = y_s[i], y_s[i+1]
            if (y1 - y0) != 0:
                x_left_cross = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    if x_right_cross is None:
        i = peak_index
        N = len(x)
        while i < N-1 and y_s[i] > half:
            i += 1
        if i > peak_index and i < N:
            x0, x1 = x[i-1], x[i]
            y0, y1 = y_s[i-1], y_s[i]
            if (y1 - y0) != 0:
                x_right_cross = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    if (x_left_cross is not None) and (x_right_cross is not None) and (x_right_cross - x_left_cross > min_sep * (x[1]-x[0])):
        fwhm = x_right_cross - x_left_cross
        sigma0 = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) 
    else:
        mask = (x >= mu0 - 10*(x[1]-x[0])) & (x <= mu0 + 10*(x[1]-x[0]))
        if mask.sum() > 3:
            ypeak = y_s[mask]
            xpeak = x[mask]
            mean = np.sum(xpeak * ypeak) / np.sum(ypeak)
            var = np.sum(ypeak * (xpeak - mean)**2) / np.sum(ypeak)
            sigma0 = np.sqrt(var) if var > 0 else max(1.0, (x[1]-x[0]))
            fwhm = sigma0 * 2.35482
        else:
            sigma0 = max(1.0, (x[1]-x[0]))
            fwhm = sigma0 * 2.35482

    return mu0, A0, sigma0, fwhm

# --- End of function body ---
