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


#Initial parameter guesses: [amp,mu,sigma,bg_const]
initial_guess = [400,650,5,7]
#Perform the curve fitting
params, covariance = curve_fit(gaussian_with_background, x, y, p0=initial_guess)

#Extract fitted parameters
amp_fit, mu_fit, sigma_fit, bg_const_fit = params
#generate fitted data
y_fit = gaussian_with_background(x, *params)

plt.scatter(x, y, label="Data", color="blue", s=10)
plt.plot(x, y_fit, label="Fitted Gaussian", color="red", linewidth=1)
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")
plt.title("Non-Linear Least Squares Fit to Gamma Spectrum")
plt.legend()
plt.grid(True)
plt.show()

# Print fitted parameters
print(f"Fitted parameters:\nAmp = {amp_fit:.2f}, Mu = {mu_fit:.2f} keV, Sigma = {sigma_fit:.2f} keV, Background = {bg_const_fit:.2f}")



