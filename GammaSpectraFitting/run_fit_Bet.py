# File: run_fit.py (Sits at the project root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the specific functions needed from your structured modules
from src.models.gamma_peak import total_model, gaussian, background 
from src.Analysis.peak_estimation import estimate_initial_params # Assumed to be updated for auto-selection
from src.Analysis.Fit_engine import run_peak_fit
# Import argparse if you are using the CLI version, otherwise hardcode paths

# --- Configuration (Using hardcoded paths for this example) ---
FILE_PATH = r'GammaSpectraFitting\data\raw\synthetic\K40_spectrum.csv'
# Replace the above with argparse logic from previous steps if using CLI!

def main():
    # --- 1. Data Loading ---
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {FILE_PATH}. Check your path.")
        return

    x = df['Energy_keV'].values
    y = df['Counts'].values
    
    # Placeholder for simple initial estimation (using default rank 1)
    # NOTE: You must update estimate_initial_params to return target_mu
    # We will simulate the returns here for simplicity.
    mu0, A0, sigma0, fwhm = estimate_initial_params(x, y) 
    target_mu = mu0 # Use the estimated center for bounds
    
    # Background parameter estimates
    c0_0 = np.median(y[(x<100) | (x>1200)])
    c1_0 = (np.median(y[x<200]) - c0_0)
    c2_0 = 0.001
    
    p0 = [A0, mu0, sigma0, c0_0, c1_0, c2_0]
    
    # Define bounds 
    lower = [0, target_mu - 20, 0.1, 0, -np.inf, 0]
    upper = [np.inf, target_mu + 20, np.inf, np.inf, np.inf, np.inf]

    # --- 2. Run Fitting Engine ---
    popt, pcov = run_peak_fit(x, y, p0, lower_bounds=lower, upper_bounds=upper)
    
    if popt is None:
        print("Analysis terminated due to fit failure.")
        return

    A_fit, mu_fit, sigma_fit, c0_fit, c1_fit, c2_fit = popt

    # --- 3. Results and Chi-Square Calculation ---
    y_fit = total_model(x, *popt)
    
    # Use np.clip to prevent division by zero in the denominator
    chi_residuals = (y - y_fit) / np.sqrt(np.clip(y_fit, 1, np.inf)) 
    
    dof = len(y) - len(popt)
    chi2 = np.sum(chi_residuals**2)
    chi2_red = chi2 / dof
    
    print(f"\nReduced Chi² = {chi2_red:.3f} (DOF={dof})")

    # --- 4. Academic Plotting (Two-Panel Rigorous View) ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1]})

    # --- TOP PANEL: Fit Decomposition ---
    
    # Calculate components
    gaussian_fit = gaussian(x, A_fit, mu_fit, sigma_fit)
    background_fit = background(x, c0_fit, c1_fit, c2_fit)

    # Data Plot (with error bars)
    axes[0].errorbar(x, y, yerr=np.sqrt(np.clip(y, 1, np.inf)), 
                     fmt='.', color='blue', label= r'Data ($\pm\sqrt{N}$)', 
                     capsize=2, markersize=3, alpha=0.5)
    
    # Fitted Model and Components
    axes[0].plot(x, y_fit, label='Total Fitted Model', color='red', linewidth=1.5)
    axes[0].plot(x, gaussian_fit, label='Gaussian Peak', color='green', linestyle='--', linewidth=1)
    axes[0].plot(x, background_fit, label='Fitted Background', color='orange', linestyle=':', linewidth=1)

    axes[0].set_title(r'Gamma Spectrum Fit: Peak $\mu$ = {mu_fit:.3f} keV'.format(mu_fit=mu_fit))
    axes[0].set_ylabel('Counts / Channel')
    axes[0].legend(loc='best', frameon=True)
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # --- BOTTOM PANEL: Normalized Residuals ---

    axes[1].scatter(x, chi_residuals, s=10, color="black", alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--")
    
    # Show +/- 1 Sigma lines for visual reference
    axes[1].axhline(1, color="gray", linestyle="-", alpha=0.7)
    axes[1].axhline(-1, color="gray", linestyle="-", alpha=0.7)
    
    axes[1].text(0.98, 0.90, r'Reduced $\chi^2$ = {chi2_red:.3f}'.format(chi2_red=chi2_red), 
                 transform=axes[1].transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    axes[1].set_xlabel('Energy (keV)')
    axes[1].set_ylabel(r'$\frac{N_{data} - N_{fit}}{\sqrt{N_{fit}}}$')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].set_ylim([-4, 4]) 
    
    # Hide x-axis label on top plot for cleaner look
    plt.setp(axes[0].get_xticklabels(), visible=False)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0) # Remove space between subplots
    plt.show()

    # Logarithmic View
    fig_log, ax_log = plt.subplots(figsize=(8, 4))
    ax_log.plot(x, y, label="Data", color="blue", linewidth=1)
    ax_log.plot(x, y_fit, label="Fitted Model", color="red", linewidth=1.5)
    ax_log.set_yscale('log') 
    ax_log.set_title("Spectrum Logarithmic View (Detailing Background)")
    ax_log.set_xlabel("Energy (keV)")
    ax_log.set_ylabel("Counts / Channel (log scale)")
    ax_log.legend(loc='upper right')
    ax_log.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()