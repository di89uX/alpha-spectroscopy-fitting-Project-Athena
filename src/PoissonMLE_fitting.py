import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

FILE = "data\synthetic\Cs137_spectrum.csv"
df = pd.read_csv(FILE)
print("Columns in csv:", list(df.columns))
print(df.head())

# Try to detect channel and counts columns
# Common names: 'Channel', 'channel', 'CH', 'Energy', 'counts', 'Counts'
col_lower = [c.lower() for c in df.columns]
# heuristics
channel_col = None
counts_col = None
for c,l in zip(df.columns, col_lower):
    if any(k in l for k in ["channel","chan","energy","e"]):
        channel_col = c
    if any(k in l for k in ["count","counts","cnt","y"]):
        counts_col = c

# Fallbacks
if channel_col is None:
    # choose first numeric column as channel
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols)>=1:
        channel_col = numeric_cols[0]
if counts_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols)>=2:
        counts_col = numeric_cols[1]
    else:
        counts_col = channel_col  # in worst case

print("Using channel column:", channel_col)
print("Using counts column:", counts_col)

x = df[channel_col].values
y = df[counts_col].values

#Define model, NLL and gradient
def model(x, A, mu, sigma, bg):
    return bg + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def nll(params, x, y):
    A, mu_c, sigma, bg = params
    mu_pred = model(x, A, mu_c, sigma, bg)
    mu_pred = np.clip(mu_pred, 1e-12, None)
    return np.sum(mu_pred - y * np.log(mu_pred))

def grad_nll(params, x, y):
    A, mu_c, sigma, bg = params
    exp_term = np.exp(-0.5 * ((x - mu_c) / sigma) ** 2)
    mu_pred = bg + A * exp_term
    mu_pred = np.clip(mu_pred, 1e-12, None)
    common = (1.0 - y / mu_pred)
    dA = exp_term
    dmu = A * exp_term * ((x - mu_c) / (sigma ** 2))
    dsigma = A * exp_term * ((x - mu_c) ** 2 / (sigma ** 3))
    dbg = np.ones_like(x)
    gA = np.sum(common * dA)
    gmu = np.sum(common * dmu)
    gsigma = np.sum(common * dsigma)
    gbg = np.sum(common * dbg)
    return np.array([gA, gmu, gsigma, gbg])

def approx_hessian_from_grad(grad_func, params, args=(), eps=1e-6):
    p = np.array(params, dtype=float)
    n = p.size
    H = np.zeros((n, n))
    for i in range(n):
        step = eps * max(1.0, abs(p[i]))
        ei = np.zeros(n); ei[i] = step
        g1 = grad_func(p + ei, *args)
        g2 = grad_func(p - ei, *args)
        H[:, i] = (g1 - g2) / (2.0 * step)
    H = 0.5 * (H + H.T)
    return H

# --- Initial guesses from data
peak_idx = np.argmax(y)
A0 = float(y[peak_idx] - np.median(y))   # crude amplitude above background
mu0 = float(x[peak_idx])
# width estimate: approximate FWHM from neighbor bins above half max
half = (y[peak_idx] + np.median(y)) / 2.0
# find left and right indices crossing half
left_idx = np.where(y[:peak_idx] <= half)[0]
left = x[left_idx[-1]] if left_idx.size>0 else x[0]
right_idx = np.where(y[peak_idx:] <= half)[0]
right = x[peak_idx + right_idx[0]] if right_idx.size>0 else x[-1]
fwhm = float(abs(right - left)) if right!=left else max(x[1]-x[0], 1.0)
sigma0 = fwhm / 2.355 if fwhm>0 else 1.0
bg0 = float(np.median(y))

p0 = np.array([max(A0, 1.0), mu0, max(sigma0, 1e-3), max(bg0, 0.0)])
bounds = [(1e-6, None), (None, None), (1e-6, None), (0.0, None)]

print("\nInitial parameter guesses:")
print("A0 =", p0[0], "mu0 =", p0[1], "sigma0 =", p0[2], "bg0 =", p0[3])

# --- Fit by minimizing NLL
res = minimize(nll, p0, args=(x, y), jac=grad_nll, bounds=bounds, method='L-BFGS-B',
               options={'ftol':1e-12, 'gtol':1e-8, 'maxiter':10000})
popt = res.x
success = res.success

# --- Hessian and uncertainties
H = approx_hessian_from_grad(grad_nll, popt, args=(x, y), eps=1e-6)
cov = None
perr = np.full_like(popt, np.nan)
try:
    cov = np.linalg.inv(H)
    perr = np.sqrt(np.diag(cov))
except Exception as e:
    cov = None
    print("Hessian inversion failed:", e)

# --- Print results
names = ["A", "mu", "sigma", "bg"]
print("\nFit success:", success)
print("Optimizer message:", res.message)
print(f"{'param':>8} {'init':>12} {'fit':>12} {'±1σ (approx)':>15}")
for i, nm in enumerate(names):
    print(f"{nm:>8} {p0[i]:12.5g} {popt[i]:12.5g} {perr[i]:15.5g}")

# --- Plot data and fit
plt.figure(figsize=(10,4))
bin_width = x[1]-x[0] if len(x)>1 else 1.0
plt.bar(x, y, width=bin_width, align='center', alpha=0.6, label='Observed counts')
xx = np.linspace(x.min(), x.max(), 2000)
plt.plot(xx, model(xx, *popt), label='Poisson-MLE fit', linewidth=2)
plt.plot(xx, model(xx, *p0), label='Initial guess', linestyle='--', alpha=0.6)
plt.xlabel(channel_col)
plt.ylabel(counts_col)
plt.title("Poisson MLE fit (Gaussian peak + constant background)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save results to a small DF and display
res_df = pd.DataFrame({
    'param': names,
    'init': p0,
    'fit': popt,
    'std_approx': perr
})



