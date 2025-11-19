import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress, pearsonr, spearmanr
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
FILE = 'Stauffer_et_al_data_approximations.xlsx'

def analyze_monkey(monkey_id, ax):
    # load and interpolate utility
    u_df = pd.read_excel(FILE, sheet_name=f'Monkey{monkey_id}_utility')
    d_df = pd.read_excel(FILE, sheet_name=f'Monkey{monkey_id}_dopamine')
    spline      = UnivariateSpline(u_df['reward'], u_df['utility'], k=3, s=0.01)
    d_df['util_interp'] = spline(d_df['reward'])
    x_util = d_df['util_interp'].values
    y_dopa = d_df['dopamine'].values

    # full OLS & correlations on util->dopa
    X_sm  = sm.add_constant(x_util)
    model = sm.OLS(y_dopa, X_sm).fit()
    print(f"\n\n=== Monkey {monkey_id} OLS Summary ===")
    print(model.summary())
    r, p_r     = pearsonr(x_util, y_dopa)
    rho, p_rho = spearmanr(x_util, y_dopa)
    print(f"Monkey {monkey_id} Pearson r = {r:.4f}, p = {p_r:.4e}")
    print(f"Monkey {monkey_id} Spearman ρ = {rho:.4f}, p = {p_rho:.4e}")

    # quick slope/intercept util->dopa
    slope, intercept, r_val, p_val, stderr = linregress(x_util, y_dopa)
    print(f"Monkey {monkey_id} slope (dopamine per util) = {slope:.4f}, R² = {r_val**2:.3f}")

    # Scale utility into nats of policy‐IG
    x_nats = slope * x_util

    # re-fit line on (x_nats, y_dopa)
    m_nat, b_nat, r_nat, p_nat, se_nat = linregress(x_nats, y_dopa)
    print(f"Monkey {monkey_id} fit in nats: slope = {m_nat:.4f}, intercept = {b_nat:.4f}, R2 = {r_nat**2:.3f}")


    ax.scatter(x_nats, y_dopa, color='C0', s=50, label='Data points')
    x_line = np.linspace(0, x_nats.max()*1.05, 200)
    y_line = m_nat * x_line + b_nat
    ax.plot(x_line, y_line, 'C1--', lw=2,
            label=f'$R^2$={r_nat**2:.2f}')

    ax.set_title(f'Monkey {monkey_id}')
    ax.set_xlabel('Δ policy–IG (nats)')
    ax.set_ylabel('Dopamine (a.u.)')
    ax.set_xlim(0, x_nats.max()*1.05)
    ax.set_ylim(0, max(y_dopa.max(), y_line.max())*1.05)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
analyze_monkey(1, axes[0])
analyze_monkey(2, axes[1])
axes[0].set_ylabel('Dopamine response (a.u.)')
plt.tight_layout()
plt.savefig('pIG_vs_dopamine.pdf', dpi=300, bbox_inches='tight')
plt.show()