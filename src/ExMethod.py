import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd  # For reading CSV

# Fitting function for a single peak (Eq. 5a-d from the paper)
def fitting_function(x, params):
    """
    Computes the fitting function F = F1 + F2, where F1 is the peak (Gaussian + exponential tail)
    and F2 is the quadratic background.

    Parameters:
    - x: array of channel numbers
    - params: list or array [Xc, G, omega, X1, C0, C1, C2]

    Returns:
    - F: array of fitted values
    """
    Xc, G, omega, X1, C0, C1, C2 = params
    F = np.zeros_like(x, dtype=float) # Initialize F as float array to avoid type errors

    # Gaussian part for Xi >= Xc - X1
    mask_gauss = x >= (Xc - X1)
    F[mask_gauss] += G * np.exp(-((x[mask_gauss] - Xc)**2) / (2 * omega**2))

    # Exponential tail for Xi < Xc - X1
    mask_exp = x < (Xc - X1)
    F[mask_exp] += G * np.exp(X1 * (2 * x[mask_exp] - 2 * Xc + X1) / (2 * omega**2))

    # Quadratic background
    bg = C0 + C1 * (x - Xc) + C2 * (x - Xc)**2
    F += bg
    return F

# Residual S (Eq. 1), using weights Wi = 1 / y (clipped to avoid division by zero)
def compute_S(y, f, weights=None):
    """
    Computes the least-squares residual S.

    Parameters:
    - y: observed counts
    - f: fitted values
    - weights: optional array of weights (default: 1 / y.clip(1))

    Returns:
    - S: scalar residual
    """
    if weights is None:
        weights = 1.0 / np.clip(y, 1, None)  # Poisson-like weights
    return np.sum(weights * (y - f)**2)

# Simple initial parameter estimates (as described in the paper)
def initial_estimates(x, y):
    """
    Provides rough initial estimates for parameters.

    Parameters:
    - x: channels
    - y: counts

    Returns:
    - params_init: list [Xc, G, omega, X1, C0, C1, C2]
    """
    # Approximate linear background: average of first and last few points
    bg_left = np.mean(y[:5])
    bg_right = np.mean(y[-5:])
    bg_slope = (bg_right - bg_left) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0
    bg_intercept = bg_left
    bg = bg_intercept + bg_slope * x

    # Subtract background
    y_sub = y - bg
    y_sub = np.clip(y_sub, 0, None)  # Avoid negative values

    # Peak center: channel with max counts after subtraction
    max_idx = np.argmax(y_sub)
    Xc = x[max_idx]

    # Height G: max count after subtraction
    G = y_sub[max_idx]

    # Width omega: approximate sigma from FWHM / 2.355
    half_max = G / 2
    left_idx = np.argmin(np.abs(y_sub[:max_idx] - half_max)) if max_idx > 0 else 0
    right_idx = np.argmin(np.abs(y_sub[max_idx:] - half_max)) + max_idx if max_idx < len(y_sub) - 1 else len(y_sub) - 1
    fwhm = x[right_idx] - x[left_idx] if right_idx > left_idx else 1.0
    omega = fwhm / (2 * np.sqrt(2 * np.log(2)))  # sigma ≈ FWHM / 2.355
    omega = max(omega, 0.1)  # Prevent zero or negative

    # X1: junction point, rough guess as ~1 sigma
    X1 = omega

    # Background params at Xc, with quadratic term 0
    C0 = bg_intercept + bg_slope * Xc  # bg at Xc
    C1 = bg_slope
    C2 = 0.0

    return [Xc, G, omega, X1, C0, C1, C2]

# Define wide parameter bounds (as per paper)
def get_bounds(params_init, x):
    """
    Sets lower and upper bounds for parameters.

    Returns:
    - bounds_lower, bounds_upper: arrays
    """
    Xc, G, omega, X1, C0, C1, C2 = params_init
    bounds_lower = [x.min(), 0.1 * G, 0.1 * omega, 0.1 * X1, 0, -np.inf, -np.inf]
    bounds_upper = [x.max(), 10 * G, 10 * omega, 10 * X1, np.inf, np.inf, np.inf]
    return np.array(bounds_lower), np.array(bounds_upper)

# Clamp parameters to bounds
def clamp_params(params, bounds_lower, bounds_upper):
    return np.clip(params, bounds_lower, bounds_upper)

# Random search algorithm (based on paper's description)
def random_search(x, y, params_init, bounds_lower, bounds_upper, max_iter=10000, Nf=140, fr=0.8, delta_min_factor=0.002):
    """
    Performs the random search optimization.

    Parameters:
    - x, y: data
    - params_init: initial parameters
    - bounds_lower, bounds_upper: parameter constraints
    - max_iter: max iterations per step size
    - Nf: max successive failures for convergence (paper suggests 10-20 * n_params)
    - fr: step reduction factor (0 < fr < 1)
    - delta_min_factor: min step as fraction of initial

    Returns:
    - params_best: optimized parameters
    - S_best: final residual
    """
    params_best = np.array(params_init)
    S_best = compute_S(y, fitting_function(x, params_best))

    # Initial step sizes (delta_a_j = (upper - lower) / Nd, Nd=20)
    delta_a = (bounds_upper - bounds_lower) / 20.0
    delta_a_min = delta_min_factor * delta_a

    while np.any(delta_a > delta_a_min):
        fault_count = 0
        iter_count = 0
        prev_delta = None  # To handle the backtrack logic more accurately
        while iter_count < max_iter and fault_count < Nf:
            # Generate random direction vector (+1 or -1 for each parameter)
            direction = np.array([random.choice([-1, 1]) for _ in params_best])

            # Displacement vector
            delta = direction * delta_a

            # Trial parameters
            params_trial = params_best + delta
            params_trial = clamp_params(params_trial, bounds_lower, bounds_upper)

            # Compute S for trial
            S_trial = compute_S(y, fitting_function(x, params_trial))

            if S_trial < S_best:
                params_best = params_trial
                S_best = S_trial
                fault_count = 0
                prev_delta = None
            else:
                # Backtrack: try inverse direction only if not already backtracked
                if prev_delta is not None and np.all(prev_delta == -delta):
                    # Already tried inverse, skip to next random
                    fault_count += 1
                    prev_delta = None
                else:
                    params_trial_inv = params_best - delta
                    params_trial_inv = clamp_params(params_trial_inv, bounds_lower, bounds_upper)
                    S_trial_inv = compute_S(y, fitting_function(x, params_trial_inv))

                    if S_trial_inv < S_best:
                        params_best = params_trial_inv
                        S_best = S_trial_inv
                        fault_count = 0
                        prev_delta = None
                    else:
                        fault_count += 1
                        prev_delta = delta  # Mark for next iteration if needed

            iter_count += 1

        # Reduce step sizes for next stage
        delta_a *= fr

    return params_best, S_best

# Function to load data from CSV
def load_csv_data(file_path, channel_col='Channel', counts_col='Counts'):
    """
    Loads gamma spectrum data from CSV file.

    Parameters:
    - file_path: path to CSV
    - channel_col: name of channel column
    - counts_col: name of counts column

    Returns:
    - x: channels (np.array)
    - y: counts (np.array)
    """
    df = pd.read_csv(csv_file)
    x = df[channel_col].values
    y = df[counts_col].values
    return x, y


if __name__ == "__main__":
    csv_file = 'data\synthetic\spectrum_Cs-137.csv' 
    x, y = load_csv_data(csv_file)

    # Get initial estimates
    params_init = initial_estimates(x, y)
    print("Initial parameters:", params_init)

    # Get bounds
    bounds_lower, bounds_upper = get_bounds(params_init, x)

    # Perform fitting
    params_fit, S_fit = random_search(x, y, params_init, bounds_lower, bounds_upper)
    print("Fitted parameters:", params_fit)
    print("Final residual S:", S_fit)

    # Plot results
    y_fit = fitting_function(x, params_fit)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', label='Observed Data')
    plt.plot(x, y_fit, 'r-', label='Fitted Curve')
    plt.xlabel('Channel Number')
    plt.ylabel('Counts')
    plt.title('Gaussian Spectrum fitt experiment method 3')
    plt.legend()
    plt.show()