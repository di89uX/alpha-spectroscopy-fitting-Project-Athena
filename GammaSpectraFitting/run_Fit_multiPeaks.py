# File: run_fit_Bet.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
# --- Import specific functions needed from structured modules ---
# Assuming these imports work based on previous conversation fixes:
from src.models.multi_peak_model import multi_peak_model, gaussian, background
from src.Analysis.peak_detection import find_all_peak_initial_params # New Multi-Peak Detector
from src.Analysis.analysis_pipline import fit_engine # New Multi-Peak Engine

# --- If available, import the full_multi_peak_analysis function ---
try:
    from src.Analysis.analysis_pipline import full_multi_peak_analysis
except ImportError:
    # Define a placeholder or raise an error if not implemented
    def full_multi_peak_analysis(*args, **kwargs):
        return {'error': 'full_multi_peak_analysis function is not implemented or imported.'}

# --- Configuration ---
FILE_PATH = r'GammaSpectraFitting\data\raw\synthetic\Na22_spectrum.csv'
BACKGROUND_PARAMS_COUNT = 3 # (c0, c1, c2)


def main():
    # --- 1. Data Loading ---
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {FILE_PATH}. Check your path.")
        return

    x = df['Energy_keV'].values
    y = df['Counts'].values
    
    # --- 2. Initial Multi-Peak Estimation ---
    
    # Note: Using find_all_peak_initial_params, which returns a list of (mu0, A0, sigma0)
    try:
        initial_params_list = find_all_peak_initial_params(x, y, 
                                                            smooth_sigma=3, 
                                                            prominence_factor=0.08)
    except ValueError as e:
        print(f"Peak detection failed: {e}")
        return

    if not initial_params_list:
        print("Analysis terminated: No significant peaks found.")
        return
        
    N_peaks = len(initial_params_list)
    print(f"Detected {N_peaks} peaks. Starting multi-peak fit...")
    
    # --- 3. Run Fitting Engine ---
    
    # The fit_engine function now handles parameter list construction and bounds internally.
    try:
        popt, pcov = fit_engine(x, y, initial_params_list, 
                                background_param_count=BACKGROUND_PARAMS_COUNT)
    except RuntimeError as e:
        print(f"Analysis terminated due to fit failure: {e}")
        return

    if popt is None:
        print("Analysis terminated due to fit failure.")
        return

    # --- 4. Results and Chi-Square Calculation ---
    
    # Use the generalized model function with the flattened popt list
    y_fit = multi_peak_model(x, *popt)
    
    # Chi-square calculation remains the same
    # Use np.clip(y_fit) for stable denominator in residuals
    chi_residuals = (y - y_fit) / np.sqrt(np.clip(y_fit, 1, np.inf)) 
    
    dof = len(y) - len(popt)
    chi2 = np.sum(chi_residuals**2)
    chi2_red = chi2 / dof
    
    print(f"\nReduced Chi² = {chi2_red:.3f} (DOF={dof})")

    # --- 5. Academic Plotting (Two-Panel Rigorous View) ---
    
    # Extract fit parameters (first 3 are background, rest are peaks)
    bkg_params = popt[:BACKGROUND_PARAMS_COUNT]
    peak_params_flat = popt[BACKGROUND_PARAMS_COUNT:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})

    # --- TOP PANEL: Fit Decomposition ---
    
    # Calculate components
    background_fit = background(x, *bkg_params)
    
    # Sum all Gaussian components
    total_gaussian_fit = np.zeros_like(x, dtype=float)
    for i in range(N_peaks):
        A_fit, mu_fit, sigma_fit = peak_params_flat[i*3 : i*3 + 3]
        total_gaussian_fit += gaussian(x, A_fit, mu_fit, sigma_fit)
        
        # Optional: Plot individual peaks (too messy for production, but good for testing)
        # axes[0].plot(x, background_fit + gaussian(x, A_fit, mu_fit, sigma_fit), 
        #              label=f'Peak {i+1} Fit', linestyle='-.', linewidth=0.5)

    # Data Plot (with error bars)
    axes[0].errorbar(x, y, yerr=np.sqrt(np.clip(y, 1, np.inf)), 
                     fmt='.', color='blue', label= r'Data ($\pm\sqrt{N}$)', 
                     capsize=2, markersize=3, alpha=0.5)
    
    # Fitted Model and Components
    axes[0].plot(x, y_fit, label=f'Total Fitted Model ({N_peaks} Peaks)', color='red', linewidth=1.5)
    axes[0].plot(x, total_gaussian_fit + background_fit[0], label='Total Gaussian Peaks', color='green', linestyle='--', linewidth=1)
    axes[0].plot(x, background_fit, label='Fitted Background', color='orange', linestyle=':', linewidth=1)

    # Set title based on the first or most prominent peak
    title_mu = initial_params_list[0][0] if initial_params_list else 'N/A'
    axes[0].set_title(r'Gamma Spectrum Multi-Peak Fit: Total Peaks Fitted = {N_peaks}'.format(N_peaks=N_peaks))
    axes[0].set_ylabel('Counts')
    axes[0].legend(loc='upper right', frameon=True)
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
    
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.show()

    # --- 6. Logarithmic View ---
    
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