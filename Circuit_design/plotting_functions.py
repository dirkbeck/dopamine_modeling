import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np

# Publication-style rcParams
mpl.rcParams.update({
    'font.family':    'Arial',
    'font.size':      10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize':9,
    'ytick.labelsize':9,
    'legend.fontsize':8,
    'figure.dpi':     300,
    'pdf.fonttype':   42,
})

def plot_layer_component_relationships(all_agent_metrics):
    """
    Plots normalized impact of cortical layers on actor, critic, and policy-IG components.
    Saves to 'cortical_region_importance_bars.pdf' and 'cortical_region_importance_heatmap.pdf'.
    """
    # Build impacts data
    impacts_data = {layer: {'Actor': [], 'Critic': [], 'Governor': []}
                    for layer in ['env_stimuli', 'motor', 'conflict', 'short_term', 'long_term']}
    for metrics in all_agent_metrics:
        sim = metrics.get('ACG', {})
        for layer_key, comps in sim.get('layer_impacts', {}).items():
            if layer_key in impacts_data:
                impacts_data[layer_key]['Actor'].append(comps.get('actor', 0))
                impacts_data[layer_key]['Critic'].append(comps.get('critic', 0))
                impacts_data[layer_key]['Governor'].append(comps.get('governor', 0))

    layer_names = ['Environmental\nStimuli', 'Motor', 'Conflict', 'Short Term', 'Long Term']
    layer_keys  = ['env_stimuli', 'motor', 'conflict', 'short_term', 'long_term']
    components  = ['Actor', 'Critic', 'Governor']

    # Prepare DataFrame
    plot_data = []
    for i, key in enumerate(layer_keys):
        for comp in components:
            vals = np.array(impacts_data[key][comp])
            mean = vals.mean() if vals.size else 0
            stderr = vals.std()/np.sqrt(len(vals)) if vals.size else 0
            plot_data.append({'Layer': layer_names[i],
                              'Component': comp,
                              'MeanImpact': mean,
                              'StdErr': stderr})
    df = pd.DataFrame(plot_data)

    # Normalize each component
    for comp in components:
        comp_vals = df[df['Component']==comp]['MeanImpact']
        m = comp_vals.max()
        if m > 0:
            df.loc[df['Component']==comp, 'MeanImpact'] /= m
            df.loc[df['Component']==comp, 'StdErr']    /= m

    # Bar plot
    x = np.arange(len(layer_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    colors = ['#3366CC','#99CC33','#FFCC33']
    for i, comp in enumerate(components):
        comp_df = df[df['Component']==comp].set_index('Layer').reindex(layer_names)
        ax.bar(x + (i-1)*width,
               comp_df['MeanImpact'], width,
               yerr=comp_df['StdErr'], capsize=3,
               color=colors[i], alpha=0.8, label=comp)

    ax.set_xlabel('Intermediate Processing Layers')
    ax.set_ylabel('Normalized Impact (SEM)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5,1.15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('cortical_region_importance_bars.pdf', bbox_inches='tight')
    plt.show()

    # Heatmap
    heatmap_data = np.zeros((3, len(layer_names)))
    for i, comp in enumerate(components):
        for j, key in enumerate(layer_keys):
            vals = np.array(impacts_data[key][comp])
            heatmap_data[i,j] = vals.mean() if vals.size else 0
        if heatmap_data[i].max() > 0:
            heatmap_data[i] /= heatmap_data[i].max()

    fig, ax = plt.subplots(figsize=(3.5,2.5))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1,
                xticklabels=layer_names, yticklabels=components,
                cbar_kws={'label':'Normalized Impact'}, ax=ax,
                linewidths=0.5)
    ax.set_xlabel('Intermediate Processing Layers')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('cortical_region_importance_heatmap.pdf', bbox_inches='tight')
    plt.show()


def plot_boxplots(results):
    """
    Boxplot comparison of agent performances.
    Saves to 'performance_comparison.pdf'.
    """
    agent_order = ['ACG','AC','AG','Random']
    names = {'ACG':'Actor-Critic-Governor',
             'AC':'Actor-Critic',
             'AG':'Actor-Policy-IG',
             'Random':'Random Agent'}
    data = [results[a] for a in agent_order]
    labels = [names[a] for a in agent_order]

    fig, ax = plt.subplots(figsize=(3.5,2.5))
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = ['#3366CC','#99CC33','#FFCC33','#FF6666']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # scatter individual points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, color='k', s=6, alpha=0.6)

    ax.set_ylabel('Mean Reward (Last 100 Episodes)')
    ax.set_title('Performance Comparison', pad=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('performance_comparison.pdf', bbox_inches='tight')
    plt.show()

    # Statistical summary table
    fig, ax = plt.subplots(figsize=(3.5,2.5))
    cell_text = []
    for a in agent_order:
        arr = np.array(results[a])
        cell_text.append([
            f"{arr.mean():.2f}",
            f"{arr.std():.2f}",
            f"{arr.min():.2f}",
            f"{arr.max():.2f}"
        ])
    table = ax.table(cellText=cell_text, rowLabels=labels,
                     colLabels=['Mean','Std','Min','Max'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.axis('off')
    ax.set_title('Agent Performance Summary', pad=4)
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(all_rewards_layered, all_rewards_direct, episodes, smoothing_window=50):
    """
    Plots learning curves for agents with layered input vs direct input.
    Saves to 'comparison_with_no_layers.pdf'.
    """
    agent_names = list(all_rewards_layered.keys())
    ncols = 2
    nrows = (len(agent_names)+1)//2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7,5), squeeze=False)
    axes = axes.flatten()
    x = np.arange(episodes)

    for idx, agent in enumerate(agent_names):
        ax = axes[idx]
        # layered input
        if agent in all_rewards_layered:
            arr = np.array(all_rewards_layered[agent])
            mean = pd.Series(arr.mean(axis=0)).rolling(smoothing_window, min_periods=1).mean()
            std  = arr.std(axis=0)
            ax.plot(x, mean, color='blue', lw=1.5, label='With Layers')
            ax.fill_between(x, mean-std, mean+std, color='blue', alpha=0.2)
        # direct input
        if agent in all_rewards_direct:
            arr = np.array(all_rewards_direct[agent])
            mean = pd.Series(arr.mean(axis=0)).rolling(smoothing_window, min_periods=1).mean()
            std  = arr.std(axis=0)
            ax.plot(x, mean, color='red', lw=1.5, label='Direct Input')
            ax.fill_between(x, mean-std, mean+std, color='red', alpha=0.2)

        ax.set_title(agent)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)

    # remove empty axes
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('comparison_with_no_layers.pdf', bbox_inches='tight')
    plt.show()