import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def _calculate_params_for_index(x, y_s, peak_index, y_median, min_sep, min_width):
    """
    HELPER FUNCTION: Calculates mu0, A0, sigma0, fwhm for a single, known peak index.
    (This is the FWHM and sigma estimation block from the original code.)
    """
    mu0 = x[peak_index]
    A0 = y_s[peak_index] - y_median  # Use median passed from main function
    half = (y_s[peak_index] + y_median) / 2.0 

    x_left_cross = None
    x_right_cross = None

    # --- FWHM Estimation (Interpolation and Fallback) ---
    try:
        # Array-based search for half-max crossing
        left_ix = np.where(x < mu0)[0]
        right_ix = np.where(x > mu0)[0]

        # Left side
        if left_ix.size > 1:
            xl, yl = x[left_ix], y_s[left_ix]
            mask = yl <= half
            if mask.any() and np.where(mask)[0][-1] < len(xl)-1:
                idx = np.where(mask)[0][-1]
                f = interp1d(yl[idx:idx+2], xl[idx:idx+2])
                x_left_cross = float(f(half))

        # Right side
        if right_ix.size > 1:
            xr, yr = x[right_ix], y_s[right_ix]
            mask = yr <= half
            if mask.any() and np.where(mask)[0][0] > 0:
                idx = np.where(mask)[0][0]
                f = interp1d(yr[idx-1:idx+1], xr[idx-1:idx+1])
                x_right_cross = float(f(half))
    except Exception:
        x_left_cross = x_right_cross = None

    # Fallback/refinement using linear search and linear interpolation (simplified)
    if x_left_cross is None:
        i = peak_index
        while i > 0 and y_s[i] > half: i -= 1
        if i >= 0 and i < peak_index and (y_s[i+1] - y_s[i]) != 0:
            x0, x1 = x[i], x[i+1]
            y0, y1 = y_s[i], y_s[i+1]
            x_left_cross = x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    if x_right_cross is None:
        i = peak_index
        N = len(x)
        while i < N-1 and y_s[i] > half: i += 1
        if i > peak_index and i < N and (y_s[i] - y_s[i-1]) != 0:
            x0, x1 = x[i-1], x[i]
            y0, y1 = y_s[i-1], y_s[i]
            x_right_cross = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            
    # --- Sigma Estimation ---
    min_width = x[1] - x[0]
    if (x_left_cross is not None) and (x_right_cross is not None) and (x_right_cross - x_left_cross > min_sep * min_width):
        fwhm = x_right_cross - x_left_cross
        sigma0 = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) 
    else:
        # Fallback: Weighted moment calculation
        mask = (x >= mu0 - 10 * min_width) & (x <= mu0 + 10 * min_width)
        if mask.sum() > 3:
            xpeak, ypeak = x[mask], y_s[mask]
            mean = np.sum(xpeak * ypeak) / np.sum(ypeak)
            var = np.sum(ypeak * (xpeak - mean)**2) / np.sum(ypeak)
            sigma0 = np.sqrt(var) if var > 0 else min_width
            fwhm = sigma0 * 2.35482
        else:
            sigma0 = min_width
            fwhm = sigma0 * 2.35482
            
    sigma0 = max(sigma0, 1e-6) # Ensure sigma is positive
    
    return mu0, A0, sigma0, fwhm


def find_all_peak_initial_params(x, y, **kwargs):
    """
    MAIN FUNCTION: Detects all prominent peaks and returns a list of initial parameters 
    [(mu0, A0, sigma0), ...] for multi-peak fitting.
    """
    smooth_sigma = kwargs.get('smooth_sigma', 2)
    min_sep = kwargs.get('min_sep', 1)
    min_energy_threshold = kwargs.get('min_energy_threshold', 200)
    prominence_factor = kwargs.get('prominence_factor', 0.1)
    height_factor = kwargs.get('height_factor', 0.05)
    
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    
    y_median = np.median(y)
    min_width = x[1] - x[0] # Channel width

    # Smoothing
    y_s = gaussian_filter1d(y, smooth_sigma) if smooth_sigma > 0 else y.copy()

    # Mask low-energy region
    mask = x > min_energy_threshold
    if not np.any(mask): 
        return []
    
    # Find ALL peaks in smoothed counts
    prominence = prominence_factor * np.max(y_s[mask])
    height = height_factor * np.max(y_s[mask])
    
    # We remove the "select the tallest peak" logic here
    peaks, _ = find_peaks(y_s, prominence=prominence, height=height)
    
    if len(peaks) == 0:
        return []

    all_peak_params = []
    
    # --- Iterate Over ALL Detected Peaks ---
    for peak_index in peaks:
        try:
            # Call the helper function for each peak index
            mu0, A0, sigma0, _ = _calculate_params_for_index(
                x, y_s, peak_index, y_median, min_sep, min_width
            )
            # Only return the required fitting parameters
            all_peak_params.append((mu0, A0, sigma0))
        except Exception as e:
            # Skip the peak if parameter estimation fails
            print(f"Warning: Failed to estimate parameters for peak at x={x[peak_index]}. Skipping. Error: {e}")
            continue

    return all_peak_params
