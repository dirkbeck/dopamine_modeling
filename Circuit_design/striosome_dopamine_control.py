import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# simulation params
dt = 0.05
T = 20
n_steps = int(T / dt)
time = np.linspace(0, T, n_steps + 1)

w_SD = 2.5
kp = 4.0
noise_scale = 0.2
C_S = 1.0
C_D = 2.0

# create target dopamine signal
target_DA = np.zeros(n_steps + 1)
t_jump = int(4 / dt)
t_ramp_start = int(8 / dt)
t_ramp_end = int(14 / dt)
t_stable_end = int(17 / dt)

target_DA[:t_jump] = 1.5
target_DA[t_jump:t_ramp_start] = 2.5
target_DA[t_ramp_start:t_ramp_end] = np.linspace(2.5, 1.0, t_ramp_end - t_ramp_start)
target_DA[t_ramp_end:t_stable_end] = 1.0
target_DA[t_stable_end:] = np.linspace(1.0, 2.0, len(target_DA) - t_stable_end)

def simulate_interaction(w_DS):
    S = np.zeros(n_steps + 1)
    D = np.zeros(n_steps + 1)
    D[0] = target_DA[0]

    for i in range(n_steps):
        # error signal
        error = target_DA[i] - D[i]
        R = kp * error

        # update S
        dSdt = -S[i] + C_S + w_DS * max(0, D[i])
        S[i + 1] = S[i] + dt * dSdt + np.random.randn() * noise_scale * dt
        S[i + 1] = max(0, S[i + 1])

        # update D
        dDdt = -D[i] + C_D + R - w_SD * max(0, S[i])
        D[i + 1] = D[i] + dt * dDdt + np.random.randn() * noise_scale * dt
        D[i + 1] = max(0, D[i + 1])

    return D

# run simulations
D_strong = simulate_interaction(-1.0)
D_weak = simulate_interaction(-0.2)
D_none = simulate_interaction(0.0)
D_excite = simulate_interaction(1.0)

# plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# dopamine activity
ax1.plot(time, target_DA, 'gray', linewidth=2, linestyle=':', label='target')
ax1.plot(time, D_strong, 'blue', linewidth=2, label='strong inhibition')
ax1.plot(time, D_weak, 'cyan', linewidth=2, linestyle='--', label='weak inhibition')
ax1.plot(time, D_none, 'black', linewidth=2, linestyle=':', label='no effect')
ax1.plot(time, D_excite, 'red', linewidth=2, linestyle='-.', label='excitation')

ax1.set_xlabel('time')
ax1.set_ylabel('circuit output')
ax1.legend()

# absolute error
ax2.plot(time, np.abs(target_DA - D_strong), 'blue', linewidth=2, label='strong inhibition')
ax2.plot(time, np.abs(target_DA - D_weak), 'cyan', linewidth=2, linestyle='--', label='weak inhibition')
ax2.plot(time, np.abs(target_DA - D_none), 'black', linewidth=2, linestyle=':', label='no effect')
ax2.plot(time, np.abs(target_DA - D_excite), 'red', linewidth=2, linestyle='-.', label='excitation')

ax2.set_xlabel('time')
ax2.set_ylabel('absolute error')
ax2.legend()

plt.savefig('striosome_dopamine_control.pdf', dpi=600, bbox_inches='tight')
plt.show()