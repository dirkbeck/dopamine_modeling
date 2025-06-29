import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

def run_simulation(inhibit_blue=False):
    n_episodes = 60
    alpha = 0.1
    pi_G = pi_B = pi_R = pi_CH = 0.0
    delta_G = []; delta_B = []; delta_R = []; delta_CH = []
    for t in range(n_episodes):
        dG = alpha * (1.0 - pi_G)
        dB = 0.0 if inhibit_blue else alpha * (0.9 * pi_G - pi_B)
        dR = alpha * (0.8 * pi_G - pi_R)
        dCH = 0.0 if inhibit_blue else alpha * (((pi_G + pi_B + pi_R)/3) - pi_CH)
        delta_G.append(dG); delta_B.append(dB)
        delta_R.append(dR); delta_CH.append(dCH)
        pi_G += dG; pi_B += dB; pi_R += dR; pi_CH += dCH
    eps = np.arange(n_episodes)
    return eps, np.array(delta_G), np.array(delta_B), np.array(delta_R), np.array(delta_CH)

def run_TD(inhibit_blue=False):
    n_episodes = 60
    alpha = 0.1
    V = {'R':0.0, 'B':0.0, 'G':0.0}
    delta_R = []; delta_B = []; delta_G = []
    for t in range(n_episodes):
        if not inhibit_blue:
            # R -> B
            dR = alpha * (V['B'] - V['R']); V['R'] += dR
            # B -> G
            dB = alpha * (V['G'] - V['B']); V['B'] += dB
        else:
            # R -> G (skip B)
            dR = alpha * (V['G'] - V['R']); V['R'] += dR
            dB = 0.0
        # G -> reward
        dG = alpha * (1.0 - V['G']); V['G'] += dG
        delta_R.append(dR); delta_B.append(dB); delta_G.append(dG)
    eps = np.arange(n_episodes)
    return eps, np.array(delta_R), np.array(delta_B), np.array(delta_G)

def find_switch(delta_main, delta_comp):
    idx = np.argmax(delta_main < delta_comp)
    return idx if idx < len(delta_main) and delta_main[idx] < delta_comp[idx] else len(delta_main)


eps,  dG1, dB1, dR1, dCH1 = run_simulation(inhibit_blue=False)
_,    dG2, dB2, dR2, dCH2 = run_simulation(inhibit_blue=True)
ti_G1 = find_switch(dG1, dB1); ti_B1 = find_switch(dB1, dCH1)
ti_R1 = find_switch(dR1, dCH1); ti_CH1 = find_switch(dCH1, dR1)
ti_G2 = find_switch(dG2, dR2); ti_B2 = find_switch(dB2, dCH2)
ti_R2 = find_switch(dR2, dCH2); ti_CH2 = find_switch(dCH2, dR2)

eps_td, tR1, tB1, tG1 = run_TD(inhibit_blue=False)
_,      tR2, tB2, tG2 = run_TD(inhibit_blue=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

fig, axes = plt.subplots(2, 2, figsize=(6, 4))

# original cue‐set
ax = axes[0,0]
ax.plot(eps[:ti_G1],  dG1[:ti_G1],  '-',  color='green',  label='cheese', linewidth=1.5)
ax.plot(eps[ti_G1:],  dG1[ti_G1:],  '--', color='green', linewidth=1.5)
ax.plot(eps[:ti_B1],  dB1[:ti_B1],  '-',  color='blue',   label='blue+cheese', linewidth=1.5)
ax.plot(eps[ti_B1:],  dB1[ti_B1:],  '--', color='blue', linewidth=1.5)
ax.plot(eps[:ti_R1],  dR1[:ti_R1],  '-',  color='red',    label='red+cheese', linewidth=1.5)
ax.plot(eps[ti_R1:],  dR1[ti_R1:],  '--', color='red', linewidth=1.5)
ax.plot(eps[:ti_B1],  dCH1[:ti_B1], '--', color='purple', label='red+blue+cheese', linewidth=1.5)
ax.plot(eps[ti_B1:],  dCH1[ti_B1:], '-',  color='purple', linewidth=1.5)
ax.set_title('Policy-IG learning', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Policy-IG', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.tick_params(labelsize=12)

# inhibited cue‐set
ax = axes[1,0]
ax.plot(eps[:ti_G2],  dG2[:ti_G2],  '-',  color='green',  label='Δ_G', linewidth=1.5)
ax.plot(eps[ti_G2:],  dG2[ti_G2:],  '--', color='green', linewidth=1.5)
ax.plot(eps[:ti_B2],  dB2[:ti_B2],  '-',  color='blue',   label='Δ_B', linewidth=1.5)
ax.plot(eps[ti_B2:],  dB2[ti_B2:],  '--', color='blue', linewidth=1.5)
ax.plot(eps[:ti_R2],  dR2[:ti_R2],  '-',  color='red',    label='Δ_R', linewidth=1.5)
ax.plot(eps[ti_R2:],  dR2[ti_R2:],  '--', color='red', linewidth=1.5)
ax.plot(eps[:ti_CH2], dCH2[:ti_CH2],'--', color='purple', label='Δ_CH', linewidth=1.5)
ax.plot(eps[ti_CH2:], dCH2[ti_CH2:],  '-',  color='purple', linewidth=1.5)
ax.set_title('Policy-IG learning', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Policy-IG', fontsize=14)
ax.tick_params(labelsize=12)

# original TD
ax = axes[0,1]
ax.plot(eps_td, tG1, '-', color='green',  label='RPE at cheese', linewidth=1.5)
ax.plot(eps_td, tB1, '-', color='blue',   label='RPE at blue', linewidth=1.5)
ax.plot(eps_td, tR1, '-', color='red',    label='RPE at red', linewidth=1.5)
ax.set_title('TD learning', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('RPE', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.tick_params(labelsize=12)

# inhibited TD
ax = axes[1,1]
ax.plot(eps_td, tG2, '-', color='green',  label='δ_G', linewidth=1.5)
ax.plot(eps_td, tB2, '-', color='blue',   label='δ_B', linewidth=1.5)
ax.plot(eps_td, tR2, '-', color='red',    label='δ_R', linewidth=1.5)
ax.set_title('TD learning', fontsize=14, fontweight='bold')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('RPE', fontsize=14)
ax.tick_params(labelsize=12)

plt.tight_layout()

plt.savefig('sequential_cue_sims.pdf')
plt.show()