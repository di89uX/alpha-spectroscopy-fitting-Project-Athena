import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_channels = 1024
energy = np.linspace(0,2000,n_channels)

def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
#Create a synthetic Gaussian peak (Cs-137 at 662keV)
peak = gauss(energy,amp = 500, mu=662, sigma=5)

#background
background = 10+ 2*np.sin(0.02*energy)
#Create a synthetic spectrum
spectrum = peak + background
noisy_spectrum = spectrum + np.random.poisson(spectrum)

df = pd.DataFrame({
    'Channel': np.arange(n_channels),
    'energy': energy,
    'Counts': noisy_spectrum,})

df.to_csv('data/synthetic/spectrum_Cs-137.csv',index=False)
plt.plot(energy,noisy_spectrum)
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Synthetic Spectrum with Cs-137 Peak')
plt.grid()
plt.show()
