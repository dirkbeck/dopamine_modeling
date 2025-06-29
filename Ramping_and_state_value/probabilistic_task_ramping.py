import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.3
mpl.rcParams['ytick.minor.width'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

n_steps = 200
cs_time = 20
us_time = 140
T_wait = us_time - cs_time

def beta21_steps(p):
    H_cs = - (p*math.log(p) + (1-p)*math.log(1-p))
    return np.array([H_cs * ((i/T_wait)**2 - ((i-1)/T_wait)**2)
                     for i in range(1, T_wait)])

# figure 1 - sweep p values
ps = np.linspace(0.01, 0.99, 99)
scale_ramp = 20.0
med_ramps = []

for p in ps:
    steps = beta21_steps(p)
    ramp_deltas = scale_ramp * steps
    med_ramps.append(np.median(ramp_deltas))

fig, ax = plt.subplots(figsize=(3, 2.5))
ax.plot(ps, med_ramps, '-', color='#1f77b4', linewidth=1.5)
ax.set_xlabel('Reward probability', fontsize=12)
ax.set_ylabel('Median Δ policy-IG\nof ramp (nats)', fontsize=12)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('median_ramp_amplitude.pdf', dpi=300, bbox_inches='tight')

# figure 2 - time series for p=0.5
p = 0.5
H_cs = - (p*math.log(p) + (1-p)*math.log(1-p))
scale_cs = 1.0
scale_us = 1.0

delta_ig = np.zeros(n_steps)

# CS spike
delta_ig[cs_time] = scale_cs * H_cs

# ramp part
base_steps = beta21_steps(p)
for idx, base in enumerate(base_steps, start=cs_time+1):
    delta_ig[idx] = scale_ramp * base

# US omission dip
delta_ig[us_time] = scale_us * math.log(1-p)

fig, ax = plt.subplots(figsize=(3.5, 2))
ax.plot(delta_ig, '-', color='#1f77b4', linewidth=1.5)
ax.axvline(cs_time, linestyle='--', color='gray', linewidth=1, alpha=0.7, label='CS')
ax.axvline(us_time, linestyle='--', color='gray', linewidth=1, alpha=0.7, label='US omitted')
ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Δ policy-IG (nats)', fontsize=12)
ax.legend(loc='upper right', fontsize=10, frameon=False)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('cs_ramp_omission_trace.pdf', dpi=300, bbox_inches='tight')
plt.show()