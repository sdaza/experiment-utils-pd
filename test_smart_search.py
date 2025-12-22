"""
Demonstration of the improved find_sample_size method with analytical approximations.
"""

import time

from experiment_utils.power_sim import PowerSim

# Test case from user's example
p = PowerSim(metric="proportion", variants=1, nsim=500, alpha=0.05, alternative="two-tailed", correction="bonferroni")

print("Testing improved find_sample_size method")
print("=" * 60)
print("Target: 80% power, baseline=0.10, effect=0.01")
print(f"Using {p.nsim} simulations per evaluation")
print("=" * 60)

start_time = time.time()
result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.01],
    min_sample_size=1000,
    max_sample_size=50000,
)
elapsed_time = time.time() - start_time

print("\nResults:")
print(result.to_string(index=False))
print(f"\nTime elapsed: {elapsed_time:.2f} seconds")
print("\nThe method used analytical approximation to narrow the search range,")
print("significantly reducing the number of simulations needed!")
