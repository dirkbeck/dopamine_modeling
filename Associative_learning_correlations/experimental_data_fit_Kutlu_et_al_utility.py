import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
mpl.rcParams['pdf.fonttype'] =42

def softmax(q, beta):
    q = np.array(q, float)
    e = np.exp(beta*q - np.max(beta*q))
    return e/np.sum(e)

def entropy(p):
    p = p[p>1e-12]
    return -np.sum(p * np.log(p)) if p.size>0 else 0.0

def delta_H_exact(alpha, beta, delta_range):
    q0 = np.array([1.0, 0.0])
    pi_0 = softmax(q0, beta)
    H0 = entropy(pi_0)
    out = []
    for delta in delta_range:
        q1 = q0.copy()
        q1[0] += alpha*delta
        pi_1 = softmax(q1, beta)
        out.append(entropy(pi_1) - H0)
    return np.array(out)

alpha = 0.5
beta  = 1.0
delta_range = np.arange(-10, 5, 0.1)

# constants for the first-order term
q0 = np.array([1.0, 0.0])
pi_0 = softmax(q0, beta)
H0 = entropy(pi_0)
pi_0a = pi_0[0]
logpi_0a = np.log(pi_0a) if pi_0a>1e-12 else 0.0
c1 = -pi_0a * (logpi_0a + H0)
k  = beta*alpha

deltaH_ex  = delta_H_exact(alpha, beta, delta_range)
deltaH_lin = c1 * (k * delta_range) 

# table contains estimates of the data based on Kutlu et al 2021 10.1016/j.cub.2021.08.052
df = pd.read_excel("Kutlu_et_al_data_approximations.xlsx",
                   sheet_name="utility_curve")

# build inverse‐IG mappings for point location
deltas = np.linspace(-30, 10, 2000)
IGs    = -delta_H_exact(alpha, beta, deltas)  # policy-IG = -deltaH
mask_neg = deltas<0
y_neg, d_neg = IGs[mask_neg], deltas[mask_neg]
order = np.argsort(y_neg)
f_inv_neg = interp1d(y_neg[order], d_neg[order], bounds_error=False,
                     fill_value=(d_neg.min(), d_neg.max()))

mask_pos = deltas>0
y_pos, d_pos = IGs[mask_pos], deltas[mask_pos]
order = np.argsort(y_pos)
f_inv_pos = interp1d(y_pos[order], d_pos[order], bounds_error=False,
                     fill_value=(d_pos.min(), d_pos.max()))

colors = {
  'shock_1.7':'#8B0000','shock_1.0':'#DC143C','shock_0.3':'#FF6B6B',
  'quinine_low':'#000080','quinine_high':'#4169E1',
  'sucrose_low':'#228B22','sucrose_high':'#32CD32'
}
points = []
# shock
for lvl, da in zip(df[df.type=='schock'].level.astype(float),
                  df[df.type=='schock'].da_response):
    IGv = da
    deltahat = f_inv_neg(IGv)
    points.append((f"Shock {lvl}", deltahat, da, colors[f"shock_{lvl}"]))
# quinine
for lvl in ['low','high']:
    row = df[(df.type=='quinine')&(df.level==lvl)].iloc[0]
    deltahat = f_inv_neg(row.da_response)
    points.append((f"Quinine {lvl}", deltahat, row.da_response,
                   colors[f"quinine_{lvl}"]))
# sucrose
for lvl in ['low','high']:
    row = df[(df.type=='sucrose')&(df.level==lvl)].iloc[0]
    deltahat = f_inv_pos(row.da_response)
    points.append((f"Sucrose {lvl}", deltahat, row.da_response,
                   colors[f"sucrose_{lvl}"]))


plt.figure(figsize=(3, 2))
plt.plot(delta_range, -deltaH_ex,
         color='blue', linewidth=2, label='Policy-IG')
plt.plot(delta_range, -deltaH_lin,
         color='green', linestyle=':', linewidth=2,
         label='RPE')
for name, x_hat, da_resp, col in points:
    plt.scatter(x_hat, da_resp,
                color=col, edgecolor='white', linewidth=2,
                s=120, label=name, zorder=5)


plt.axhline(0, color='gray', linewidth=0.5, alpha=0.7)
plt.axvline(0, color='gray', linewidth=0.5, alpha=0.7)
plt.xlabel("RPE", fontweight='bold')
plt.ylabel("delta dopamine", fontweight='bold')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),
           fontsize=8, loc='upper left',
           bbox_to_anchor=(0.02, 0.98),
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('policyIG_utility_relationship.pdf', dpi=300)
plt.show()