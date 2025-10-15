# File: src/models/multi_peak_model.py

import numpy as np

def gaussian(x, A, mu, sigma):
    """Simple Gaussian function for a single peak."""
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

def background(x, c0, c1, c2):
    """Exponential plus constant background model (c0 + c1 * exp(-c2 * x))."""
    return c0 + c1 * np.exp(-c2 * x)

def multi_peak_model(x, *params):
    """
    Combined model for N Gaussian peaks plus an exponential background.
    Parameters: [c0, c1, c2, A1, mu1, sigma1, A2, mu2, sigma2, ...]
    """
    if len(params) < 3:
        # If there are not enough parameters for background, return zero array.
        return np.zeros_like(x)
    
    # --- 1. Extract Background Parameters ---
    c0, c1, c2 = params[0:3]
    
    # --- FIX: Initialize y_model immediately with the background value ---
    y_model = background(x, c0, c1, c2)  # <-- ASSIGNMENT GUARANTEED
    
    # --- 2. Extract and Sum Gaussian Components ---
    peak_params = params[3:]
    N_peaks = len(peak_params) // 3
    
    if N_peaks * 3 != len(peak_params):
        # This check is good but should ideally use an exception for bad input length
        # raise ValueError("Parameter list length is invalid.")
        pass

    for i in range(N_peaks):
        A = peak_params[i * 3]
        mu = peak_params[i * 3 + 1]
        sigma = peak_params[i * 3 + 2]
        
        # Add the Gaussian component. y_model is now guaranteed to exist.
        y_model += gaussian(x, A, mu, sigma)

    return y_model # <-- y_model is now always defined