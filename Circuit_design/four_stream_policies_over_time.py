import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

# --- USER-ADJUSTABLE PARAMETERS ----------------------------------
streams        = ['Limbic', 'Cognitive', 'Oculomotor', 'Motor']
action_counts  = [3, 5, 10, 20]
T              = 200

initial_bias   = [5.0, 0.0, 0.0, 0.0]
t1, t2         = 50, 120
pulse_width    = 8
drift_scale    = 0.1
recur_strength = 0.95

response_to_t1 = {'Limbic':    15.0,
                  'Cognitive':  2.0,
                  'Oculomotor':20.0,
                  'Motor':      1.0}
response_to_t2 = {'Limbic':     1.0,
                  'Cognitive': 25.0,
                  'Oculomotor':-10.0,
                  'Motor':     30.0}
# ------------------------------------------------------------------

np.random.seed(1)
max_entropies = [np.log(n) for n in action_counts]
cmap = plt.get_cmap('tab20')

weights = []
for idx, n in enumerate(action_counts):
    w = np.zeros((T, n))
    w[0, 0] = initial_bias[idx]
    for t in range(1, T):
        w[t] = recur_strength*w[t-1] + drift_scale*np.random.randn(n)
        if t1 <= t < t1+pulse_width:
            w[t, 0] += response_to_t1[streams[idx]]
        if t2 <= t < t2+pulse_width:
            # switch motor action index to 1 at t2
            target = 1 if streams[idx]=='Motor' else 0
            w[t, target] += response_to_t2[streams[idx]]
    weights.append(w)

policies = [softmax(w, axis=1) for w in weights]

policy_igs = []
for i, pi in enumerate(policies):
    e  = entropy(pi.T)
    ig = max_entropies[i] - e
    policy_igs.append(ig)

time = np.arange(T)

fig, axes = plt.subplots(5, 1, figsize=(6, 8), sharex=True,
                         gridspec_kw={'height_ratios':[1,1,1,1,1]})

action_labels = {
    'Limbic':    ['Assess valence', 'Ignore stimulus'],
    'Cognitive': ['Plan approach',   'Delay decision'],
    'Oculomotor':['Fixate food',    'Scan surroundings'],
    'Motor':     ['Begin reach',    'Grasp & eat']
}

for idx, ax in enumerate(axes[:-1]):
    pi     = policies[idx]
    n_act  = action_counts[idx]
    colors = [cmap(i) for i in np.linspace(0,1,n_act)]
    ax.stackplot(time, pi.T, colors=colors, alpha=0.9)
    ax.set_ylim(0,1)
    ax.set_yticks([0,1])
    ax.axvline(t1, color='k', linestyle='--')
    ax.axvline(t2, color='r', linestyle='--')
    ax.set_ylabel(streams[idx], rotation=0, labelpad=40, va='center')

    lbls = action_labels[streams[idx]]
    legend_items = [
        Patch(color=colors[0], label=lbls[0]),
        Patch(color=colors[1], label=lbls[1])
    ]
    ax.legend(handles=legend_items, loc='center right', frameon=False, fontsize=7)

ax_ig = axes[-1]
for idx, ig in enumerate(policy_igs):
    ax_ig.plot(time, ig, lw=1.5, label=streams[idx], color=cmap(idx/len(streams)))
ax_ig.set_ylim(0, max(max_entropies))
ax_ig.set_yticks([0, max(max_entropies)])
ax_ig.axvline(t1, color='k', linestyle='--')
ax_ig.axvline(t2, color='r', linestyle='--')
ax_ig.set_xlabel('Time step')
ax_ig.set_ylabel('Policy-IG', rotation=0, labelpad=50, va='center')
ax_ig.legend(loc='lower center', bbox_to_anchor=(0.5,-0.4), ncol=4, fontsize=7)
ax_ig.set_ylim([0,3])

plt.tight_layout(h_pad=0.5)
plt.savefig('four_stream_policies_over_time.pdf')
plt.show()