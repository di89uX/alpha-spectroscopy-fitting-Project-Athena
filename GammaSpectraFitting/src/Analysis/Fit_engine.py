# File: src/analysis/fit_engine.py
import numpy as np
from scipy.optimize import curve_fit
from src.models.gamma_peak import total_model 

def run_peak_fit(x, y, p0, lower_bounds=None, upper_bounds=None):
    """
    Performs the least squares fit using the combined model.
    
    Args:
        x, y: Energy and Counts data.
        p0: Initial guess parameters.
        lower_bounds, upper_bounds: Optional bounds for curve_fit.
        
    Returns: popt, pcov
    """
    # Weight by sqrt(y) to approximate Poisson errors (avoid zeros)
    sigma_weights = np.sqrt(np.clip(y, 1, np.inf))
    
    try:
        popt, pcov = curve_fit(
            total_model, 
            x, 
            y, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds) if lower_bounds and upper_bounds else (-np.inf, np.inf),
            sigma=sigma_weights, 
            absolute_sigma=True
        )
        return popt, pcov
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None