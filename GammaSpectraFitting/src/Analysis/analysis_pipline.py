import warnings
import numpy as np
from scipy.optimize import curve_fit
from src.Analysis.peak_detection import find_all_peak_initial_params
from src.models.multi_peak_model import multi_peak_model

def fit_engine(x, y, initial_params_list, background_param_count=3,
               mu_window_factor=5, c2_upper=0.01, verbose=False):
    """
    Executes the simultaneous curve fitting for N peaks plus background.
    
    Args:
        x (np.ndarray): Independent variable data (e.g., channels/energy).
        y (np.ndarray): Dependent variable data (e.g., counts).
        initial_params_list (list): List of initial (mu0, A0, sigma0) tuples for peaks.
        background_param_count (int): Number of background parameters (e.g., 3 for c0, c1, c2).

    Returns:
        popt (np.ndarray): Optimized fit parameters.
        pcov (np.ndarray): Covariance matrix.
    """
    if not initial_params_list:
        raise ValueError("Cannot fit: initial_params_list is empty.")

    # Basic input validation
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 2:
        raise ValueError("x and y must contain at least two points")

    # For now, the pipeline and multi_peak_model assume 3 background params.
    if background_param_count != 3:
        raise ValueError("background_param_count must be 3 for the current model implementation")
        
    # --- 1. Prepare Initial Guesses (p0) for the multi_peak_model ---
    # The initial guesses MUST match the model's signature: [c0, c1, c2, A1, mu1, sigma1, ...]

    # Initial Background Guess (Simple assumption: constant + small exp term)
    B0 = np.median(y)
    c0_0 = float(B0)          # Initial guess for constant term
    c1_0 = 1.0                # Default guess for exp amplitude
    c2_0 = 1e-6               # Default guess for exp decay rate

    p0 = [c0_0, c1_0, c2_0]
    
    # Append initial peak parameters (A, mu, sigma)
    # Append initial peak parameters (A, mu, sigma)
    for mu0, A0, sigma0 in initial_params_list:
        # Ensure A0 and sigma0 are positive; warn if clipping occurs
        A_clipped = max(A0, 1e-6)
        sigma_clipped = max(sigma0, 1e-6)
        if A_clipped != A0 and verbose:
            warnings.warn(f"Clipped initial amplitude A0={A0} to {A_clipped}")
        if sigma_clipped != sigma0 and verbose:
            warnings.warn(f"Clipped initial sigma0={sigma0} to {sigma_clipped}")
        p0.extend([A_clipped, mu0, sigma_clipped])

    # --- 2. Prepare Bounds ---
    # Bounds for background parameters (c0 >= 0, c1 >= 0, 0 <= c2 <= c2_upper)
    lower_bounds = [0.0, 0.0, 1e-12]
    upper_bounds = [np.inf, np.inf, float(c2_upper)]
    
    # Set reasonable bounds for peak parameters
    # Allow mu to move +/- mu_window_factor * sample spacing
    sample_spacing = float(x[1] - x[0])
    mu_window = sample_spacing * float(mu_window_factor)
    
    for mu0, _, _ in initial_params_list:
        # Peak Bounds: [A_min, mu_min, sigma_min]
        lower_bounds.extend([0.0, float(mu0) - mu_window, 1e-6])
        # Peak Bounds: [A_max, mu_max, sigma_max]
        upper_bounds.extend([np.inf, float(mu0) + mu_window, np.inf])

    # --- 3. Perform Fit (using Poisson error model) ---
    # Estimate weights (sigma) based on Poisson counting statistics (sqrt(y))
    sigma_y = np.sqrt(np.where(y > 0, y, 1.0))

    try:
        popt, pcov = curve_fit(
            multi_peak_model,
            x,
            y,
            p0=p0,
            sigma=sigma_y,
            absolute_sigma=True,   # sigma_y represents the true uncertainty (Poisson counting)
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
    except RuntimeError as e:
        # Re-raise with more context
        raise RuntimeError(f"curve_fit failed: {e}")

    # Post-fit: warn if parameters are very close to bounds (possible clipping)
    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)
    popt_arr = np.asarray(popt, dtype=float)
    eps = 1e-8
    # Expand lb/ub arrays to match popt length if needed
    if lb.size < popt_arr.size:
        lb = np.pad(lb, (0, popt_arr.size - lb.size), constant_values=(-np.inf,))
    if ub.size < popt_arr.size:
        ub = np.pad(ub, (0, popt_arr.size - ub.size), constant_values=(np.inf,))

    close_to_lb = np.isclose(popt_arr, lb, atol=eps, rtol=0)
    close_to_ub = np.isclose(popt_arr, ub, atol=eps, rtol=0)
    if verbose:
        if close_to_lb.any() or close_to_ub.any():
            warnings.warn(f"Some optimized parameters are very close to their bounds. close_to_lb={close_to_lb}, close_to_ub={close_to_ub}")

    return popt, pcov

def full_multi_peak_analysis(x, y, fit_kwargs=None, **detection_kwargs):
    """
    Orchestrates the full pipeline: detection, fitting, and parameter analysis.
    """
    
    # 1. Detect Peaks
    initial_params_list = find_all_peak_initial_params(x, y, **detection_kwargs)
    
    if not initial_params_list:
        return {"error": "No significant peaks were found."}
        
    # 2. Execute Fitting Engine
    try:
        if fit_kwargs is None:
            fit_kwargs = {}
        popt, pcov = fit_engine(x, y, initial_params_list, **fit_kwargs)
    except (RuntimeError, ValueError) as e:
        return {"error": f"Fitting failed to converge or input error: {e}"}

    # 3. Analyze Results (Uncertainties & Goodness of Fit)
    
    # Standard Errors (Uncertainties)
    try:
        perr = np.sqrt(np.diag(pcov))
    except ValueError:
        perr = np.full_like(popt, np.nan) # Covariance matrix could not be inverted

    # Chi-Squared Calculation
    y_fit = multi_peak_model(x, *popt)
    sigma_y_sq = np.where(y > 0, y, 1) # Variance is counts
    chi_squared = np.sum((y - y_fit)**2 / sigma_y_sq)
    
    # Degrees of Freedom (DoF)
    n_params = len(popt)
    dof = len(x) - n_params
    reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan
    
    # 4. Format and Return Report
    
    # Background results are the first 3 parameters
    bkg_opt = {'c0': popt[0], 'c1': popt[1], 'c2': popt[2]}
    bkg_err = {'c0': perr[0], 'c1': perr[1], 'c2': perr[2]}
    
    peak_results = []
    N_peaks = (len(popt) - 3) // 3
    
    for i in range(N_peaks):
        start_idx = 3 + i * 3
        
        peak_results.append({
            'A': popt[start_idx], 'A_err': perr[start_idx],
            'mu': popt[start_idx + 1], 'mu_err': perr[start_idx + 1],
            'sigma': popt[start_idx + 2], 'sigma_err': perr[start_idx + 2]
        })
        
    return {
        'refined_params': {
            'Background': {'opt': bkg_opt, 'err': bkg_err},
            'Peaks': peak_results
        },
        'initial_params': initial_params_list,
        'goodness_of_fit': {
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'dof': dof
        }
    }