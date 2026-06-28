# %%
from experiment_utils import estimate_true_success_rate, false_positive_risk

# %%
true_success_rate = estimate_true_success_rate(win_rate=0.12, alpha=0.10, power=0.90)
fpr = false_positive_risk(alpha=0.10, power=0.90, prior_success_rate=true_success_rate)
print(f"Estimated true success rate: {true_success_rate:.2%}")
print(f"False positive risk: {fpr:.2%}")

# %%
