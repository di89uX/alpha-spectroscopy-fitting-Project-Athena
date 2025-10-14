# File: run_fit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the specific functions needed from your structured modules
from src.models.gamma_peak import total_model, background
from src.Analysis.peak_estimation import estimate_initial_params
from src.Analysis.Fit_engine import run_peak_fit

def main():
    # --- 1. Data Loading (The only place you read the file) ---
    FILE_PATH = r'GammaSpectraFitting\data\raw\synthetic\Cs137_spectrum.csv'
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {FILE_PATH}. Check your path.")
        return

    x = df['Energy_keV'].values
    y = df['Counts'].values
    
    # --- 2. Initial Parameter Estimation (Reusable function call) ---
    mu0, A0, sigma0, fwhm = estimate_initial_params(x, y)
    print("Initial guess: mu0=%.2f, A0=%.1f, sigma0=%.3f (FWHM=%.3f)" % (mu0, A0, sigma0, fwhm))

    # Background parameter estimates
    c0_0 = np.median(y[(x<100) | (x>1200)])
    c1_0 = (np.median(y[x<200]) - c0_0)
    c2_0 = 0.001
    
    p0 = [A0, mu0, sigma0, c0_0, c1_0, c2_0]
    
    # Define bounds (can be put in a separate config file later)
    lower = [0, mu0 - 20, 0.1, 0, -np.inf, 0]
    upper = [np.inf, mu0 + 20, np.inf, np.inf, np.inf, np.inf]

    # --- 3. Run Fitting Engine (Reusable function call) ---
    popt, pcov = run_peak_fit(x, y, p0, lower_bounds=lower, upper_bounds=upper)
    
    if popt is None:
        print("Analysis terminated due to fit failure.")
        return

    # --- 4. Results and Analysis ---
    y_fit = total_model(x, *popt)
    residuals = y - y_fit
    perr = np.sqrt(np.diag(pcov))
    param_names = ["A", "mu", "sigma", "c0", "c1", "c2"]

    print("\nFit Results (with 1σ uncertainties):")
    for name, val, err in zip(param_names, popt, perr):
        print(f"{name:5s} = {val:.3f} ± {err:.3f}")

    # Reduced Chi-square calculation
    dof = len(y) - len(popt)
    chi2 = np.sum((residuals / np.sqrt(np.clip(y, 1, np.inf)))**2)
    chi2_red = chi2 / dof
    print(f"Reduced Chi² = {chi2_red:.3f}")
    
    # --- 5. Plotting (Could be moved to a plotting utility file if repetitive) ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    # ... [Insert the complete plotting code from your original script here] ...
    
    # 1. Fit vs Data
    axes[0].scatter(x, y, label="Data", color="blue", s=10)
    axes[0].plot(x, y_fit, label="Fitted Model", color="red", linewidth=1)
    # ... other settings
    
    # 2. Initial Guess vs Data
    y_init = total_model(x, *p0)
    axes[1].plot(x, y, label="Data")
    axes[1].plot(x, y_init, label="Initial Guess", linestyle='--')
    # ... other settings

    # 3. Residuals
    axes[2].scatter(x, residuals, s=10, color="black")
    axes[2].axhline(0, color="red", linestyle="--")
    # ... other settings

    plt.tight_layout()
    # Save the figure to the 'results/figures' directory
    # plt.savefig('results/figures/ba133_peak_fit.png') 
    plt.show()

if __name__ == "__main__":
    main()