import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['pdf.fonttype'] =42

# table contains estimates of the data based on Kutlu et al 2021 10.1016/j.cub.2021.08.052
df = pd.read_excel(
    "Kutlu_et_al_data_approximations.xlsx",
    sheet_name="decreasing_novelty",
    dtype={"Trial": str, "da_response": float}
)

def midpoint(trial_str):
    a, b = trial_str.split("_to_")
    return (float(a) + float(b)) / 2

df["x"] = df["Trial"].map(midpoint)

alpha, beta = 0.5, 1.0

def softmax(q, β):
    e = np.exp(β * q - np.max(β * q))
    return e / np.sum(e)

def entropy(p):
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))

def ig_at_delta(d):
    q0 = np.array([1.0, 0.0])
    p0 = softmax(q0, beta)
    q1 = q0.copy()
    q1[0] += alpha * d
    p1 = softmax(q1, beta)
    return -(entropy(p1) - entropy(p0))

df["ig_value"] = df["x"].map(ig_at_delta)

# Fit DA vs x only
s_da = linregress(df["x"], df["da_response"])
x_min, x_max = df["x"].min(), df["x"].max()
x_vals = np.array([x_min, x_max])


fig, ax = plt.subplots(figsize=(4, 3))

da_color = 'k'
ax.scatter(df["x"], df["da_response"],
           color=da_color, s=100,label="Dopamine")


fit_color = '#d62728'
ax.plot(x_vals,
        s_da.intercept + s_da.slope * x_vals,
        color=fit_color, linestyle="--",
        label="Policy-IG")

# RPE = 0 because the stimuli are value-neutral
ax.axhline(0,
           color='gray', linestyle='--',
           linewidth=1.5, label="RPE")

ax.set_xlabel("Trial bin", fontweight='bold')
ax.set_ylabel("Dopamine response", fontweight='bold', color=da_color)
ax.tick_params(axis="y", labelcolor=da_color, width=1.5)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.set_xticks(df["x"])
ax.set_xticklabels(df["Trial"], rotation=45, ha="right", fontsize=11)

ymin = min(df["ig_value"].min(), df["da_response"].min()) - 0.5
ymax = max(df["ig_value"].max(), df["da_response"].max()) + 0.5
ax.set_ylim(ymin, ymax)

ax.legend(loc="upper right", frameon=True,
          fancybox=True, shadow=True,
          bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('novelty_decline_curve.pdf',
            format='pdf', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()