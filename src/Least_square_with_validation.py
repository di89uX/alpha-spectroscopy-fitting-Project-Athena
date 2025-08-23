import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"data\synthetic\Cs137_spectrum.csv")
x = df['Energy_keV'].values
y = df['Counts'].values

# --- Model ---
def gaussian_with_background(x, amp, mu, sigma, bg_const):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + bg_const

# --- Initial Guess ---
def estimate_initial_guesses(x, y):
    bg_const = np.median(y[:100])
    peak_index = np.argmax(y)
    mu = x[peak_index]
    amp = y[peak_index] - bg_const

    half_max = bg_const + amp / 2
    indices_above_half = np.where(y > half_max)[0]
    if len(indices_above_half) >= 2:
        fwhm = x[indices_above_half[-1]] - x[indices_above_half[0]]
        sigma = fwhm / 2.355
    else:
        sigma = 5
    return [amp, mu, sigma, bg_const]

initial_guess = estimate_initial_guesses(x, y)
print("Initial guess:", initial_guess)

# --- Fit ---
params, covariance = curve_fit(gaussian_with_background, x, y, p0=initial_guess)
amp_fit, mu_fit, sigma_fit, bg_const_fit = params
y_fit = gaussian_with_background(x, *params)

# --- Uncertainties ---
perr = np.sqrt(np.diag(covariance))

# --- Residuals ---
residuals = y - y_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(np.mean(residuals**2))
reduced_chi2 = ss_res / (len(y) - len(params))

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# 1. Fit vs Data
axes[0].scatter(x, y, label="Data", color="blue", s=10)
axes[0].plot(x, y_fit, label="Fitted Gaussian", color="red", linewidth=1)
axes[0].set_title("Gaussian Fit")
axes[0].set_xlabel("Energy (keV)")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid(True)

# 2. Initial Guess vs Data
y_init = gaussian_with_background(x, *initial_guess)
axes[1].plot(x, y, label="Data")
axes[1].plot(x, y_init, label="Initial Guess", linestyle='--')
axes[1].set_title("Initial Guess vs Data")
axes[1].set_xlabel("Energy (keV)")
axes[1].set_ylabel("Counts")
axes[1].legend()
axes[1].grid(True)

# 3. Residuals
axes[2].scatter(x, residuals, s=10, color="black")
axes[2].axhline(0, color="red", linestyle="--")
axes[2].set_title("Residuals")
axes[2].set_xlabel("Energy (keV)")
axes[2].set_ylabel("Residuals")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# --- Print Results ---
print("\nFitted Parameters (with uncertainties):")
print(f"Amp     = {amp_fit:.2f} ± {perr[0]:.2f}")
print(f"Mu      = {mu_fit:.2f} ± {perr[1]:.2f} keV")
print(f"Sigma   = {sigma_fit:.2f} ± {perr[2]:.2f} keV")
print(f"BG Const= {bg_const_fit:.2f} ± {perr[3]:.2f}")

print("\nModel Validation Metrics:")
print(f"R²                = {r_squared:.4f}")
print(f"RMSE              = {rmse:.4f}")
print(f"Reduced Chi²      = {reduced_chi2:.4f}")
