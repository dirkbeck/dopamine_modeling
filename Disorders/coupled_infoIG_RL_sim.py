import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 9

# simulation params
num_timesteps = 300
R_true_drug = 0.7
V_drug_initial = 0.05
IPFD_initial = 0.65
encounter_interval = 25
encounter_duration = 3

# softmax params
V_alternative = 0.55
softmax_beta = 5.0

# control agent
IPFD_control_floor = 0.10
k_IPFD_decay_control = 0.07
alpha_V_control_base = 0.15
k_alpha_control_from_IPFD = 0.1

# SUD agent
IPFD_SUD_cap = 0.95
k_IPFD_increase_SUD = 0.04
alpha_V_SUD_base = 0.13
k_alpha_SUD_from_IPFD = 0.25
k_R_amp_SUD_from_IPFD = 1.7

# low policy-IG agent
IPFD_low_floor = 0.02
k_IPFD_decay_low = 0.02
alpha_V_low_base = 0.03
k_alpha_low_from_IPFD = 0.01

# normalization value
max_v_drug = R_true_drug * (1 + k_R_amp_SUD_from_IPFD * IPFD_SUD_cap) * 1.15

# initial values
initial_exploitation = np.clip(V_drug_initial / max_v_drug, 0.15, 1.0)
initial_p_choice = np.exp(softmax_beta * V_drug_initial) / (
        np.exp(softmax_beta * V_drug_initial) + np.exp(softmax_beta * V_alternative))

# trajectories
traj_control = {'IPFD': [IPFD_initial], 'V_drug': [V_drug_initial],
                'Exploitation': [initial_exploitation], 'P_choice': [initial_p_choice]}
traj_SUD = {'IPFD': [IPFD_initial], 'V_drug': [V_drug_initial],
            'Exploitation': [initial_exploitation], 'P_choice': [initial_p_choice]}
traj_low = {'IPFD': [IPFD_initial], 'V_drug': [V_drug_initial],
            'Exploitation': [initial_exploitation], 'P_choice': [initial_p_choice]}

# agent states
V_drug_control, IPFD_control = V_drug_initial, IPFD_initial
V_drug_SUD, IPFD_SUD = V_drug_initial, IPFD_initial
V_drug_low, IPFD_low = V_drug_initial, IPFD_initial

encounter_timer = 0

for t in range(num_timesteps):
    # check for drug encounters
    if t > 0 and t % encounter_interval == 0:
        encounter_timer = encounter_duration

    # control agent
    alpha_control = alpha_V_control_base + k_alpha_control_from_IPFD * IPFD_control
    if encounter_timer > 0:
        error = R_true_drug - V_drug_control
        V_drug_control += alpha_control * error
        IPFD_control = IPFD_control_floor + (IPFD_control - IPFD_control_floor) * (1 - k_IPFD_decay_control)
        IPFD_control = max(IPFD_control_floor, IPFD_control)

    exploitation_control = np.clip(V_drug_control / max_v_drug, 0.15, 1.0)
    p_choice_control = np.exp(softmax_beta * V_drug_control) / (
            np.exp(softmax_beta * V_drug_control) + np.exp(softmax_beta * V_alternative))

    traj_control['IPFD'].append(IPFD_control)
    traj_control['V_drug'].append(V_drug_control)
    traj_control['Exploitation'].append(exploitation_control)
    traj_control['P_choice'].append(p_choice_control)

    # SUD agent
    alpha_SUD = alpha_V_SUD_base + k_alpha_SUD_from_IPFD * IPFD_SUD
    R_perceived = R_true_drug * (1 + k_R_amp_SUD_from_IPFD * IPFD_SUD)
    if encounter_timer > 0:
        error = R_perceived - V_drug_SUD
        V_drug_SUD += alpha_SUD * error
        increase = k_IPFD_increase_SUD * (V_drug_SUD / R_true_drug)
        IPFD_SUD = IPFD_SUD + increase * (IPFD_SUD_cap - IPFD_SUD)
        IPFD_SUD = min(IPFD_SUD_cap, IPFD_SUD)

    exploitation_SUD = np.clip(V_drug_SUD / max_v_drug, 0.15, 1.0)
    p_choice_SUD = np.exp(softmax_beta * V_drug_SUD) / (
            np.exp(softmax_beta * V_drug_SUD) + np.exp(softmax_beta * V_alternative))

    traj_SUD['IPFD'].append(IPFD_SUD)
    traj_SUD['V_drug'].append(V_drug_SUD)
    traj_SUD['Exploitation'].append(exploitation_SUD)
    traj_SUD['P_choice'].append(p_choice_SUD)

    # low policy-ig agent
    alpha_low = alpha_V_low_base + k_alpha_low_from_IPFD * IPFD_low
    if encounter_timer > 0:
        error = R_true_drug - V_drug_low
        V_drug_low += alpha_low * error
        IPFD_low = IPFD_low_floor + (IPFD_low - IPFD_low_floor) * (1 - k_IPFD_decay_low)
        IPFD_low = max(IPFD_low_floor, IPFD_low)

    exploitation_low = np.clip(V_drug_low / max_v_drug, 0.15, 1.0)
    p_choice_low = np.exp(softmax_beta * V_drug_low) / (
            np.exp(softmax_beta * V_drug_low) + np.exp(softmax_beta * V_alternative))

    traj_low['IPFD'].append(IPFD_low)
    traj_low['V_drug'].append(V_drug_low)
    traj_low['Exploitation'].append(exploitation_low)
    traj_low['P_choice'].append(p_choice_low)

    if encounter_timer > 0:
        encounter_timer -= 1

# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))
plt.subplots_adjust(wspace=0.3)

# policy-IG vs learned value
for i in range(len(traj_control['IPFD']) - 1):
    ax1.plot(traj_control['IPFD'][i:i + 2], traj_control['V_drug'][i:i + 2],
             color='royalblue', alpha=traj_control['Exploitation'][i], linewidth=2.5)
for i in range(len(traj_SUD['IPFD']) - 1):
    ax1.plot(traj_SUD['IPFD'][i:i + 2], traj_SUD['V_drug'][i:i + 2],
             color='crimson', alpha=traj_SUD['Exploitation'][i], linewidth=3.0)
for i in range(len(traj_low['IPFD']) - 1):
    ax1.plot(traj_low['IPFD'][i:i + 2], traj_low['V_drug'][i:i + 2],
             color='forestgreen', alpha=traj_low['Exploitation'][i], linewidth=2.2)

ax1.axhline(R_true_drug, color='black', linestyle='--', linewidth=1.2)

def add_arrows(ax, x_path, y_path, color, num_arrows=4):
    indices = np.linspace(0, len(x_path) - 2, num_arrows, dtype=int)
    for idx in indices:
        if np.sqrt((x_path[idx + 1] - x_path[idx]) ** 2 + (y_path[idx + 1] - y_path[idx]) ** 2) > 1e-2:
            ax.annotate("", xy=(x_path[idx + 1], y_path[idx + 1]), xytext=(x_path[idx], y_path[idx]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.6))


add_arrows(ax1, traj_control['IPFD'], traj_control['V_drug'], 'blue')
add_arrows(ax1, traj_SUD['IPFD'], traj_SUD['V_drug'], 'darkred')
add_arrows(ax1, traj_low['IPFD'], traj_low['V_drug'], 'darkgreen')

# start point
ax1.plot(IPFD_initial, V_drug_initial, 'ko', markersize=6)

# labels
ax1.text(traj_control['IPFD'][-15] - 0.05, traj_control['V_drug'][-15] + 0.02,
         'Control', color='royalblue', fontsize=9, ha='right', weight='bold')
ax1.text(traj_SUD['IPFD'][-25] + 0.02, traj_SUD['V_drug'][-25] + 0.05,
         'SUD', color='crimson', fontsize=9, ha='left', weight='bold')
ax1.text(traj_low['IPFD'][-15] + 0.03, traj_low['V_drug'][-15] - 0.03,
         'Low P-IG', color='forestgreen', fontsize=9, ha='left', va='top', weight='bold')

ax1.set_xlabel('Policy-IG', fontsize=10)
ax1.set_ylabel('Learned Value', fontsize=10)
ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')

# choice probability over time
time_steps = np.arange(num_timesteps + 1)
ax2.plot(time_steps, traj_control['P_choice'], color='royalblue', linewidth=2, label='Control')
ax2.plot(time_steps, traj_SUD['P_choice'], color='crimson', linewidth=2.5, label='SUD')
ax2.plot(time_steps, traj_low['P_choice'], color='forestgreen', linewidth=1.8, linestyle='--', label='Low P-IG')

ax2.set_xlabel('Episode #', fontsize=10)
ax2.set_ylabel('Choice Probability', fontsize=10)
ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, frameon=False)
ax2.set_xlim(0, num_timesteps)
ax2.set_ylim(0, 1.05)

# clean up axes
for ax in [ax1, ax2]:
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('learning_trajectories.pdf')

plt.tight_layout()
plt.show()