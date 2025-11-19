import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] =42

# data estimated from Fiorillo et al 2003 10.1126/science.1077349
df = pd.read_excel('fiorillo_et_al_data_approximations.xlsx')
p = df['probability'].values
da1A = df['change in dopamine, set 1 monkey A'].values
da2A = df['change in dopamine, set 2 monkey A'].values
da1B = df['change in dopamine, set 1 monkey B'].values
da2B = df['change in dopamine, set 2 monkey B'].values
da1A = np.where(pd.isna(da1A), np.nan, da1A)
da2A = np.where(pd.isna(da2A), np.nan, da2A)
da1B = np.where(pd.isna(da1B), np.nan, da1B)
da2B = np.where(pd.isna(da2B), np.nan, da2B)


def H(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return - (p*np.log(p) + (1-p)*np.log2(1-p))

# linear model: delta DA = a * H(p) + b
def model(Hp, a, b):
    return a*Hp + b

def fit_and_plot(p, da, label, color):
    mask = ~np.isnan(da)
    x = H(p[mask])       # entropy
    y = da[mask]
    if len(y) < 2:  # need at least 2 points for fitting
        return
    # initial guess
    p0 = [ (y.max()-y.min()), y.min() ]
    popt, pcov = curve_fit(model, x, y, p0=p0)
    a, b = popt
    print(f"{label}:  a = {a:.3f},  b = {b:.3f}")

    plt.scatter(p, da, label=f"{label} data", color=color, s=60,
               edgecolors='black', linewidth=0.8, zorder=3)
    pp = np.linspace(0,1,200)
    plt.plot(pp, model(H(pp),a,b),
             color=color, linestyle='--', linewidth=2,
             label=f"{label} fit", zorder=2)

plt.figure(figsize=(4, 3))  # Small figure size suitable for papers
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

fit_and_plot(p, da1A, "Set1 Monkey A", '#1f77b4')
fit_and_plot(p, da2A, "Set2 Monkey A", '#ff7f0e')
fit_and_plot(p, da1B, "Set1 Monkey B", '#2ca02c')
fit_and_plot(p, da2B, "Set2 Monkey B", '#d62728')

plt.xlabel("Reward probability p", fontsize=12, fontweight='bold')
plt.ylabel("Δ Dopamine (normalized)", fontsize=12, fontweight='bold')
plt.title("Linear fit ΔDA = a·H(p) + b", fontsize=12, fontweight='bold')

plt.grid(False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True,
          fancybox=False, shadow=False, fontsize=10, framealpha=0.9,
          edgecolor='black')

plt.tight_layout()

plt.savefig('dopamine_analysis.pdf', format='pdf', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')


plt.show()
