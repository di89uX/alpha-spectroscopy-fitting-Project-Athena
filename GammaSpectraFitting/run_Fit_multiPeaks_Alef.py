# File: run_fit_Bet.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate # New Import for clear table printing
import locale # New Import for better string formatting

# Set locale for better number formatting (optional but good practice)
try:
    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
except locale.Error:
    # Fallback if the system doesn't have the locale
    pass 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.models.multi_peak_model import multi_peak_model, gaussian, background
from src.Analysis.peak_detection import find_all_peak_initial_params
from src.Analysis.analysis_pipline import fit_engine

# ---import the full_multi_peak_analysis function ---
try:
    from src.Analysis.analysis_pipline import full_multi_peak_analysis
except ImportError:
    # Define a placeholder or raise an error if not implemented
    def full_multi_peak_analysis(*args, **kwargs):
        return {'error': 'full_multi_peak_analysis function is not implemented or imported.'}

# --- Configuration ---
FILE_PATH = r'GammaSpectraFitting\data\raw\synthetic\Eu-152_spectrum.csv'
BACKGROUND_PARAMS_COUNT = 3 # (c0, c1, c2)

# Function to format a measurement as Value ± Uncertainty
def format_measurement(value, error, precision=3):
    """Formats a value and its error in the standard (v ± e) format."""
    return f"{value:.{precision}f} $\\pm$ {error:.{precision}f}"


def main():
    # ... (Steps 1, 2, 3, 4 remain unchanged) ...
    # --- 1. Data Loading ---
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {FILE_PATH}. Check your path.")
        return

    x = df['Energy_keV'].values
    y = df['Counts'].values
    
    # --- 2. Initial Multi-Peak Estimation ---
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
    y_fit = multi_peak_model(x, *popt)
    chi_residuals = (y - y_fit) / np.sqrt(np.clip(y_fit, 1, np.inf)) 
    dof = len(y) - len(popt)
    chi2 = np.sum(chi_residuals**2)
    chi2_red = chi2 / dof
    
    print(f"\nReduced Chi² = {chi2_red:.3f} (DOF={dof})")

    # --- 5.Parameter Report Generation (UPDATED) ---
    
    # Calculate 1-sigma uncertainties (standard errors) from the diagonal of the covariance matrix
    perr = np.sqrt(np.diag(pcov))
    
    # --- Organize Parameters ---
    bkg_params = popt[:BACKGROUND_PARAMS_COUNT]
    bkg_perr = perr[:BACKGROUND_PARAMS_COUNT]
    peak_params_flat = popt[BACKGROUND_PARAMS_COUNT:]
    peak_perr_flat = perr[BACKGROUND_PARAMS_COUNT:]
    
    # Data structure for the report table
    report_data = []
    
    # 5.1. Add Background (Uncertainty and t-ratio shown separately)
    for i in range(BACKGROUND_PARAMS_COUNT):
        param_name = f'Bkg_c{i}'
        t_ratio = abs(bkg_params[i] / bkg_perr[i]) if bkg_perr[i] != 0 else np.inf
        # For background, we use the old format (separate columns)
        report_data.append([
            param_name, 
            bkg_params[i], 
            bkg_perr[i], 
            t_ratio
        ])

    # 5.2. Add Peaks
    for i in range(N_peaks):
        A_fit, mu_fit, sigma_fit = peak_params_flat[i*3 : i*3 + 3]
        A_perr, mu_perr, sigma_perr = peak_perr_flat[i*3 : i*3 + 3]
        
        # --- NEW: Peak Energy (Mean) reported as Value ± Error string ---
        energy_measurement_str = format_measurement(mu_fit, mu_perr, precision=4)
        t_ratio_mu = abs(mu_fit / mu_perr) if mu_perr != 0 else np.inf
        
        # Mean (mu) - The crucial energy value
        # Note: The Value and Uncertainty columns for this row will be blank 
        # because the combined string is placed in the 'Parameter' column for visual grouping.
        report_data.append([
            f'P{i+1}_Energy ($\mu$) (keV)', 
            energy_measurement_str, 
            "", # Value column blanked
            t_ratio_mu
        ])
        
        # Amplitude (A)
        t_ratio_A = abs(A_fit / A_perr) if A_perr != 0 else np.inf
        report_data.append([f'P{i+1}_Amp (A)', A_fit, A_perr, t_ratio_A])
        
        # Sigma (sigma) - Width
        t_ratio_sigma = abs(sigma_fit / sigma_perr) if sigma_perr != 0 else np.inf
        report_data.append([f'P{i+1}_Sigma (\\sigma)', sigma_fit, sigma_perr, t_ratio_sigma])
        
    # Create DataFrame and print table
    df_report = pd.DataFrame(report_data, columns=[
        'Parameter', 
        'Value', 
        'Uncertainty (1\\sigma)', 
        't-ratio'
    ])
    
    print("\n" + "="*80)
    print("                **GAMMA SPECTRUM PEAK FITTING REPORT**")
    print(f"       Total Peaks: {N_peaks} | Reduced Chi-Squared: {chi2_red:.3f} | DOF: {dof}")
    print("="*80)
    
    # Use tabulate to print a clean, formatted table
    # floatfmt adjusted to handle the mixed-type columns (string for Energy)
    print(tabulate(df_report, 
                   headers='keys', 
                   tablefmt='fancy_grid', 
                   showindex=False,
                   floatfmt={
                       'Value': '.4e', # Scientific notation for A and sigma
                       'Uncertainty (1\$\sigma\$)': '.4e', # Scientific notation for errors
                       't-ratio': '.2f' # Two decimals for t-ratio
                   }))
    print("="*80 + "\n")


    # ... (Steps 6 and 7 for Plotting remain unchanged) ...
    # --- 6.Plotting (Two-Panel Rigorous View) ---
    bkg_params = popt[:BACKGROUND_PARAMS_COUNT]
    peak_params_flat = popt[BACKGROUND_PARAMS_COUNT:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})

    # Calculate components
    background_fit = background(x, *bkg_params)
    total_gaussian_fit = np.zeros_like(x, dtype=float)
    for i in range(N_peaks):
        A_fit, mu_fit, sigma_fit = peak_params_flat[i*3 : i*3 + 3]
        total_gaussian_fit += gaussian(x, A_fit, mu_fit, sigma_fit)
        
    # Data Plot
    axes[0].errorbar(x, y, yerr=np.sqrt(np.clip(y, 1, np.inf)), 
                     fmt='.', color='blue', label= r'Data ($\pm\sqrt{N}$)', 
                     capsize=2, markersize=3, alpha=0.5)
    
    # Fitted Model and Components
    axes[0].plot(x, y_fit, label=f'Total Fitted Model ({N_peaks} Peaks)', color='red', linewidth=1.5)
    axes[0].plot(x, total_gaussian_fit + background_fit[0], label='Total Gaussian Peaks', color='green', linestyle='--', linewidth=1)
    axes[0].plot(x, background_fit, label='Fitted Background', color='orange', linestyle=':', linewidth=1)

    title_mu = initial_params_list[0][0] if initial_params_list else 'N/A'
    axes[0].set_title(r'Gamma Spectrum Multi-Peak Fit: Total Peaks Fitted = {N_peaks}'.format(N_peaks=N_peaks))
    axes[0].set_ylabel('Counts')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Residuals
    axes[1].scatter(x, chi_residuals, s=10, color="black", alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--")
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

    # --- 7. Logarithmic View ---
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