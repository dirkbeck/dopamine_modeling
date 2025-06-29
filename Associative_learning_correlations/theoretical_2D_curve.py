import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def softmax(q_values, beta):
  q_values = np.asarray(q_values, dtype=float)
  max_q = np.max(beta * q_values)
  exp_q = np.exp(beta * q_values - max_q)
  sum_exp_q = np.sum(exp_q)
  if sum_exp_q <= 1e-12:
      return np.ones_like(q_values) / len(q_values)
  return exp_q / sum_exp_q

def calculate_entropy(policy):
    policy = np.asarray(policy)
    nonzero_policy = policy[policy > 1e-12]
    if len(nonzero_policy) == 0:
        return 0.0
    logs = np.log(nonzero_policy)
    if not np.all(np.isfinite(logs)):
         return -np.sum(nonzero_policy[np.isfinite(logs)] * logs[np.isfinite(logs)])
    return -np.sum(nonzero_policy * logs)

def calculate_entropy_change(old_policy, new_policy):
    return calculate_entropy(new_policy) - calculate_entropy(old_policy)

alpha = 0.1
beta = 2
delta_range = np.arange(-30, 10.1, 0.5)

q_values_initial = np.array([1.0, 0.0])

pi_old_theoretical = softmax(q_values_initial, beta)
H_old_theoretical = calculate_entropy(pi_old_theoretical)
pi_old_a = pi_old_theoretical[0]

if pi_old_a <= 1e-12:
    c1 = 0.0
else:
    if not np.isfinite(H_old_theoretical): H_old_theoretical = 0
    log_pi_old_a = np.log(pi_old_a)
    if not np.isfinite(log_pi_old_a): log_pi_old_a = 0
    c1 = -pi_old_a * (log_pi_old_a + H_old_theoretical)

c_2_approx = -0.5 * pi_old_a * (1.0 - pi_old_a)

delta_H_linear_approx = []
delta_H_quadratic_approx = []
for delta in delta_range:
    k = beta * alpha * delta
    delta_H_lin = c1 * k
    delta_H_quad = c1 * k + c_2_approx * k**2
    delta_H_linear_approx.append(delta_H_lin)
    delta_H_quadratic_approx.append(delta_H_quad)

delta_H_linear_approx = np.array(delta_H_linear_approx)
delta_H_quadratic_approx = np.array(delta_H_quadratic_approx)

delta_H_exact_list = []

pi_old_exact = softmax(q_values_initial, beta)

for delta in delta_range:
    q_values_new = q_values_initial.copy()
    q_values_new[0] = q_values_initial[0] + alpha * delta

    pi_new_exact = softmax(q_values_new, beta)

    entropy_change_exact = calculate_entropy_change(pi_old_exact, pi_new_exact)

    if not np.isfinite(entropy_change_exact):
         entropy_change_exact = np.nan

    delta_H_exact_list.append(entropy_change_exact)

delta_H_exact_list = np.array(delta_H_exact_list)

plt.figure(figsize=(4, 3))

plt.plot(delta_range, -delta_H_exact_list, color='blue', linewidth=2, label='Exact ΔH')

plt.plot(delta_range, -delta_H_linear_approx, color='green', linestyle=':', linewidth=1.5,
         label='Linear Approx. (1st Order)')

plt.plot(delta_range, -delta_H_quadratic_approx, color='red', linestyle='--', linewidth=1.5,
         label='Quadratic Approx. (2nd Order)')

plt.xlabel("RPE")
plt.ylabel("Δ Information gain (nats)")
plt.title("RPE")
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=8)
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('entropy_change_vs_rpe_exact_vs_approx.pdf')
plt.show()