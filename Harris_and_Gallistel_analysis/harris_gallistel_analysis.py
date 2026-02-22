import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'axes.linewidth': 0.75,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'none',
    'figure.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def load_and_prepare_data(excel_path, sheet_name='T2Acq'):
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
    df = df.dropna(how='all').reset_index(drop=True)

    required = ['Rat', 'C', 'T', 'Info']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def assign_to_groups(df, theoretical_CT):
    df = df.copy().reset_index(drop=True)
    df['group'] = 0
    for i in range(len(df)):
        iota = df.loc[i, 'Info']
        if pd.isna(iota) or iota <= 0:
            continue
        df.loc[i, 'group'] = np.argmin(np.abs(theoretical_CT - iota)) + 1
    return df


def fit_gallistel(log_iota, log_trials):
    slope, intercept, r, p, se = stats.linregress(log_iota, log_trials)
    return slope, np.exp(intercept), r**2, intercept


def fit_policy_ig(log_iota, trials):
    X = np.column_stack([np.ones_like(log_iota), 1.0 / log_iota])
    coeffs, _, _, _ = np.linalg.lstsq(X, trials, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((trials - pred)**2)
    ss_tot = np.sum((trials - np.mean(trials))**2)
    return coeffs[0], coeffs[1], 1 - ss_res / ss_tot


def analyze_acquisition_criterion(df, criterion_col, first_n_groups=10):
    if criterion_col not in df.columns:
        return None

    df_work = df.copy().reset_index(drop=True)

    mask = (df_work['group'].values <= first_n_groups) & \
           df_work[criterion_col].notna().values & \
           (df_work[criterion_col].values > 0) & \
           df_work['Info'].notna().values & \
           (df_work['Info'].values > 0)

    df_f = df_work.loc[mask].copy().reset_index(drop=True)
    if len(df_f) < 10:
        return None

    gs = df_f.groupby('group').agg({
        'Info': 'mean', criterion_col: 'median', 'Rat': 'count'
    }).reset_index()
    gs.columns = ['group', 'informativeness', 'median_trials', 'n_rats']
    gs = gs[gs['n_rats'] >= 3].reset_index(drop=True)
    if len(gs) < 5:
        return None

    log_im1 = np.log(gs['informativeness'].values - 1)
    log_t = np.log(gs['median_trials'].values)
    log_i = np.log(gs['informativeness'].values)
    trials = gs['median_trials'].values
    v = np.isfinite(log_im1) & np.isfinite(log_t)
    if np.sum(v) < 3:
        return None

    slope, k, r2_gal, intercept = fit_gallistel(log_im1[v], log_t[v])
    a, b, r2_pig = fit_policy_ig(log_i[v], trials[v])
    mean_t = np.mean(trials[v])

    ind_mask = (df_f['Info'].values > 1) & (df_f[criterion_col].values > 0)
    df_ind = df_f.loc[ind_mask].reset_index(drop=True)

    r2_gal_s = r2_pig_s = r2_td_s = np.nan
    if len(df_ind) >= 3:
        li = np.log(df_ind['Info'].values - 1)
        lt = np.log(df_ind[criterion_col].values)
        vi = np.isfinite(li) & np.isfinite(lt)
        if np.sum(vi) >= 3:
            _, _, r2_gal_s, _ = fit_gallistel(li[vi], lt[vi])
            _, _, r2_pig_s = fit_policy_ig(np.log(df_ind['Info'].values[vi]),
                                           df_ind[criterion_col].values[vi])
            r2_td_s = 0.0

    return {
        'criterion': criterion_col,
        'n_rats': len(df_f), 'n_groups': len(gs),
        'slope_gal': slope, 'k_gal': k, 'intercept_gal': intercept,
        'r2_gal_group': r2_gal, 'r2_gal_session': r2_gal_s,
        'a_pig': a, 'b_pig': b,
        'r2_pig_group': r2_pig, 'r2_pig_session': r2_pig_s,
        'mean_trials': mean_t,
        'r2_td_group': 0.0, 'r2_td_session': r2_td_s,
        'group_stats': gs, 'df_filtered': df_f,
    }


def clean_ax(ax):
    ax.tick_params(which='both', direction='out', length=3, width=0.75)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def create_figure(all_results):
    fig_w, fig_h = 6.5, 2.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    n_panels = 2
    ml, mr, mb, mt, gap = 0.08, 0.02, 0.38, 0.10, 0.10
    pw = (1 - ml - mr - (n_panels - 1) * gap) / n_panels
    ph = 1 - mb - mt

    def panel_pos(i):
        return [ml + i * (pw + gap), mb, pw, ph]

    colors = {'gal': '#3366CC', 'pig': '#E53E3E', 'td': '#38A169'}

    n_m = len(all_results)
    x = np.arange(n_m)
    w = 0.25

    labels = []
    for r in all_results:
        c = r['criterion']
        c = c.replace('CS rate > Context rate', 'CS > Context')
        c = c.replace('parsedCS > parsedITI', 'parsed CS > ITI')
        c = c.replace('nDkl ', '')
        c = c.replace('Odds ', '')
        c = c.replace(':', 'to')
        labels.append(c)

    r2_gg = [r['r2_gal_group'] for r in all_results]
    r2_pg = [r['r2_pig_group'] for r in all_results]
    r2_tg = [r['r2_td_group'] for r in all_results]
    r2_gs = [r['r2_gal_session'] if not np.isnan(r['r2_gal_session']) else 0 for r in all_results]
    r2_ps = [r['r2_pig_session'] if not np.isnan(r['r2_pig_session']) else 0 for r in all_results]
    r2_ts = [r['r2_td_session'] if not np.isnan(r.get('r2_td_session', np.nan)) else 0
             for r in all_results]

    ax1 = fig.add_axes(panel_pos(0))
    ax1.bar(x - w, r2_gg, w, color=colors['gal'], edgecolor='k', linewidth=0.5)
    ax1.bar(x, r2_pg, w, color=colors['pig'], edgecolor='k', linewidth=0.5)
    ax1.bar(x + w, r2_tg, w, color=colors['td'], edgecolor='k', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
    ax1.set_ylabel('$R^2$')
    ax1.set_title('Group-level (medians)', fontsize=7)
    ax1.set_ylim([0, 1])
    ax1.set_yticks([0, 1])
    clean_ax(ax1)
    ax1.legend(['Informativeness', 'Policy-IG', 'TD-RPE'],
               fontsize=5, ncol=3, frameon=False, loc='upper left',
               bbox_to_anchor=(0, -0.18), borderaxespad=0)

    ax2 = fig.add_axes(panel_pos(1))
    ax2.bar(x - w, r2_gs, w, color=colors['gal'], edgecolor='k', linewidth=0.5)
    ax2.bar(x, r2_ps, w, color=colors['pig'], edgecolor='k', linewidth=0.5)
    ax2.bar(x + w, r2_ts, w, color=colors['td'], edgecolor='k', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
    ax2.set_ylabel('$R^2$')
    ax2.set_title('Session-level (individual rats)', fontsize=7)
    ax2.set_ylim([0, 1])
    ax2.set_yticks([0, 1])
    clean_ax(ax2)
    ax2.legend(['Informativeness', 'Policy-IG', 'TD-RPE'],
               fontsize=5, ncol=3, frameon=False, loc='upper left',
               bbox_to_anchor=(0, -0.18), borderaxespad=0)

    return fig


def main(excel_path, sheet_name='T2Acq'):
    theoretical_CT = np.array([1.5, 3, 4.5, 6, 9, 15, 20, 27, 36, 54, 72, 110, 180, 300])

    df = load_and_prepare_data(excel_path, sheet_name=sheet_name)
    df = assign_to_groups(df, theoretical_CT)

    print(f"Group counts:\n{df.groupby('group')['Rat'].count()}\n")

    criteria = [
        'CS rate > Context rate',
        'min nDkl',
        'nDkl Odds 4:1',
        'nDkl p<.05',
        'nDkl p<.01',
        'nDkl p<.001',
        'parsedCS > parsedITI',
        'parsedCS > parsedITI Odds 4to1',
        'parsedCS > parsedITI Odds 10:1',
        'parsedCS > parsedITI Odds 20:1',
        'parsedCS > parsedITI Odds 100:1',
        'parsedCS > parsedITI Odds 1000:1',
        'Earliest',
    ]

    all_results = []
    for c in criteria:
        if c not in df.columns:
            continue
        r = analyze_acquisition_criterion(df, c, first_n_groups=10)
        if r is not None:
            all_results.append(r)
            print(f"{c:40s}  slope={r['slope_gal']:.3f}  Gal={r['r2_gal_group']:.3f}  PIG={r['r2_pig_group']:.3f}")

    if not all_results:
        print("No valid results")
        return None

    fig = create_figure(all_results)
    fig.savefig('model_comparison.pdf', bbox_inches='tight', format='pdf')
    fig.savefig('model_comparison.svg', bbox_inches='tight', format='svg')

    print(f"\n{'Criterion':<40s} {'Slope':>7s} {'Gal(grp)':>9s} {'PIG(grp)':>9s} {'Gal(ses)':>9s} {'PIG(ses)':>9s}")
    print("-" * 85)
    for r in all_results:
        gs = r['r2_gal_session'] if not np.isnan(r['r2_gal_session']) else 0
        ps = r['r2_pig_session'] if not np.isnan(r['r2_pig_session']) else 0
        print(f"{r['criterion']:<40s} {r['slope_gal']:>7.3f} {r['r2_gal_group']:>9.3f} "
              f"{r['r2_pig_group']:>9.3f} {gs:>9.3f} {ps:>9.3f}")

    print(f"\nMean slope: {np.mean([r['slope_gal'] for r in all_results]):.3f}")

    return all_results


if __name__ == "__main__":
    results = main('Acquisition_Table.xlsx', sheet_name='T2Acq')