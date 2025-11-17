import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

time = np.linspace(0, 60, 500)
tonic_level = 0.5

small_burst_times = [15, 55]
small_burst_amps = [0.3, 0.25]
small_burst_widths = [3.0, 3.5]
phasic_peak_time = 30
phasic_amplitude = 2.5
phasic_width = 6.0

noise_level = 0.18
noise_window_size = 25
add_noise = True

epoch_boundaries = [0, 20, 40, 60]
epoch_labels = ['State 1', 'State 2', 'State 3']

d1_threshold = 2.0
d1_steepness = 8.0
d2_threshold = 0.6
d2_steepness = 1.5
d2_min_activity = 0.4
d2_max_activity = 0.8

def dopamine_signal_complex(t, tonic, peak_t, amp, width, small_ts, small_amps, small_widths):
    signal = tonic + amp * np.exp(-(t - peak_t)**2 / (2 * width**2))
    for i in range(len(small_ts)):
        signal += small_amps[i] * np.exp(-(t - small_ts[i])**2 / (2 * small_widths[i]**2))
    return signal

def generate_rolling_noise(signal_shape, noise_level, window_size, seed=None):
    if seed is not None: np.random.seed(seed)
    raw_noise = np.random.normal(loc=0.0, scale=1.0, size=signal_shape)
    if window_size <= 0: return raw_noise * noise_level
    window = np.ones(window_size) / window_size
    smoothed_noise_unscaled = np.convolve(raw_noise, window, mode='same')
    rolling_noise = smoothed_noise_unscaled * noise_level * np.sqrt(window_size / 2)
    return rolling_noise

def receptor_activity_d1(da_conc, threshold, steepness):
    return 1 / (1 + np.exp(-steepness * (da_conc - threshold)))

def pathway_activity_d2_moderate(da_conc, threshold, steepness, min_activity, max_activity):
    activity_range = max_activity - min_activity
    inhibition_fraction = 1 / (1 + np.exp(steepness * (da_conc - threshold)))
    activity = min_activity + activity_range * inhibition_fraction
    return activity

deterministic_da = dopamine_signal_complex(time, tonic_level, phasic_peak_time,
                                          phasic_amplitude, phasic_width,
                                          small_burst_times, small_burst_amps, small_burst_widths)
if add_noise:
    da_noise_seed = 42
    rolling_noise = generate_rolling_noise(time.shape, noise_level, noise_window_size, seed=da_noise_seed)
    da_concentration = deterministic_da + rolling_noise
else:
    da_concentration = deterministic_da
da_concentration = np.maximum(0.01, da_concentration)

d1_response = receptor_activity_d1(da_concentration, d1_threshold, d1_steepness)
d2_response = pathway_activity_d2_moderate(da_concentration, d2_threshold, d2_steepness,
                                         d2_min_activity, d2_max_activity)

spike_rate_scale = 1.0
dt = np.mean(np.diff(time))
spike_prob = da_concentration * spike_rate_scale * dt
spike_prob = np.minimum(spike_prob, 1.0)

spike_gen_seed = 123
np.random.seed(spike_gen_seed)
random_values = np.random.rand(len(time))
spike_indices = np.where(random_values < spike_prob)[0]
spike_times = time[spike_indices]

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)

ax0 = fig.add_subplot(gs[0])
ax0.eventplot([spike_times], colors='black', lineoffsets=0.5,
             linelengths=0.8, linewidths=1.0)
ax0.set_ylim(0, 1)
ax0.set_yticks([])
ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax0.set_title(f'D1/D2 Sensitivity (DA Level as Event Rate, Noise={noise_level}, Win={noise_window_size})', fontsize=12, pad=10)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.plot(time, d1_response, color='red', lw=2.0, label='D1 Activity (Highly Phasic)')
ax1.plot(time, d2_response, color='blue', lw=2.0, label='D2 Pathway Activity (Tonic/Stable)')

peak_time_index = np.argmax(da_concentration)
d1_peak_act = d1_response[peak_time_index]
ax1.annotate('D1: Activated\nONLY by large\nphasic peak',
           xy=(time[peak_time_index], d1_peak_act * 0.9),
           xytext=(time[peak_time_index] - 15, d1_peak_act + 0.2),
           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'),
           ha='right', va='center', fontsize=9, color='red')

mid_time_index = np.where(time > small_burst_times[-1]+3)[0][0]
mid_d2_level = np.mean(d2_response[mid_time_index-20:mid_time_index+20])
ax1.annotate('D2: Relatively stable;\nslowly tracks average DA',
           xy=(time[mid_time_index], mid_d2_level),
           xytext=(time[mid_time_index] + 25, mid_d2_level - 0.1),
           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='blue'),
           ha='left', va='center', fontsize=9, color='blue')

large_burst_d2_dip_idx = peak_time_index
large_burst_d2_dip = d2_response[large_burst_d2_dip_idx]
ax1.annotate(f'D2: Minimal change\nduring large burst',
           xy=(time[large_burst_d2_dip_idx], large_burst_d2_dip + 0.02),
           xytext=(time[large_burst_d2_dip_idx] + 10, large_burst_d2_dip + 0.2),
           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='blue'),
           ha='left', va='center', fontsize=9, color='blue')

ax1.set_xlabel('Time (Arbitrary Units) / State Epochs', fontsize=10)
ax1.set_ylabel('Relative Receptor/\nPathway Activity', fontsize=9)
ax1.set_ylim(-0.05, 1.1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='center left')

for ax in [ax0, ax1]:
    line_alpha = 0.5 if ax == ax0 else 0.7
    for bound in epoch_boundaries[1:-1]:
        ax.axvline(bound, color='k', linestyle='--', lw=1, alpha=line_alpha)

for i in range(len(epoch_labels)):
    label_pos_x = (epoch_boundaries[i] + epoch_boundaries[i+1]) / 2
    ax1.text(label_pos_x, 1.05, epoch_labels[i], ha='center', va='bottom', fontsize=9, color='dimgray')

plt.tight_layout(h_pad=0.5, rect=[0, 0, 1, 0.96])
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('phasic_dopamine_d1_d2_event_rate_noise_epochs.pdf', bbox_inches='tight')
plt.show()