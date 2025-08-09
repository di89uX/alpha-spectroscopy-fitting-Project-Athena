import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

#Load data from CSV
df = pd.read_csv(r"data\synthetic\spectrum_Cs-137.csv")
#print(df.head())
x = df['energy'].values
y = df['Counts'].values

def gaussian_with_background(x, amp, mu, sigma, bg_const):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + bg_const


def estimate_initial_guesses(x, y):
    # Estimate background from low-count regions
    bg_const = np.median(y[:100])  # or use y[-100:]

    # Find the peak
    peak_index = np.argmax(y)
    mu = x[peak_index]
    amp = y[peak_index] - bg_const

    # Estimate width (sigma) using FWHM approximation
    half_max = bg_const + amp / 2

    # Find indices where y crosses half_max
    indices_above_half = np.where(y > half_max)[0]
    if len(indices_above_half) >= 2:
        fwhm = x[indices_above_half[-1]] - x[indices_above_half[0]]
        sigma = fwhm / 2.355  # Convert FWHM to sigma
    else:
        sigma = 5  # Fallback

    return [amp, mu, sigma, bg_const]

# Get the initial guess parameters: [amp, mu, sigma, bg_const]
initial_guess = estimate_initial_guesses(
    x, y, 
    
)

print("Initial guess:", initial_guess)


#Perform the curve fitting
params, covariance = curve_fit(gaussian_with_background, x, y, p0=initial_guess)

#Extract fitted parameters
amp_fit, mu_fit, sigma_fit, bg_const_fit = params
#generate fitted data
y_fit = gaussian_with_background(x, *params)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

# --- Top subplot: fitted Gaussian ---
axes[0].scatter(x, y, label="Data", color="blue", s=10)
axes[0].plot(x, y_fit, label="Fitted Gaussian", color="red", linewidth=1)
axes[0].set_xlabel("Energy (keV)")
axes[0].set_ylabel("Counts")
axes[0].set_title("Non-Linear Least Squares Fit")
axes[0].legend()
axes[0].grid(True)

# --- Bottom subplot: initial guess ---
y_init = gaussian_with_background(x, *initial_guess)
axes[1].plot(x, y, label="Data")
axes[1].plot(x, y_init, label="Initial Guess", linestyle='--')
axes[1].set_xlabel("Energy (keV)")
axes[1].set_ylabel("Counts")
axes[1].set_title("Initial Guess vs Data")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# Print fitted parameters
print(f"Fitted parameters:\nAmp = {amp_fit:.2f}, Mu = {mu_fit:.2f} keV, Sigma = {sigma_fit:.2f} keV, Background = {bg_const_fit:.2f}")



