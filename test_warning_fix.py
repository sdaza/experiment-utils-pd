"""Test the warning fix for variant-only comparisons"""
from experiment_utils.power_sim import PowerSim

print("Test 1: Variant-only comparison (1, 2) with optimize_allocation")
print("=" * 60)
p = PowerSim(metric='proportion', variants=2, nsim=1000, alpha=0.05)

result = p.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.08, 0.05],
    target_comparisons=[(1, 2)],  # Only variant 1 vs variant 2
    optimize_allocation=True,
    power_criteria="any",
    min_sample_size=500,
    max_sample_size=15000
)

print("\nResult:")
print(result[['total_sample_size', 'sample_sizes_by_group', 'achieved_power_by_comparison']])
print()

# Test 2: Multiple variant comparisons without control
print("\nTest 2: Multiple variant comparisons without control")
print("=" * 60)
p = PowerSim(metric='proportion', variants=3, nsim=1000, alpha=0.05)

result = p.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.05, 0.03, 0.07],
    target_comparisons=[(1, 2), (2, 3)],  # Only variant comparisons
    optimize_allocation=True,
    min_sample_size=500,
    max_sample_size=15000
)

print("\nResult:")
print(result[['total_sample_size', 'sample_sizes_by_group', 'achieved_power_by_comparison']])
print()

# Test 3: Mixed comparisons (control and variant)
print("\nTest 3: Mixed comparisons (control + variant comparisons)")
print("=" * 60)
result = p.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.05, 0.03, 0.07],
    target_comparisons=[(0, 1), (1, 2)],  # Control vs variant 1, and variant 1 vs 2
    optimize_allocation=True,
    min_sample_size=500,
    max_sample_size=15000
)

print("\nResult:")
print(result[['total_sample_size', 'sample_sizes_by_group', 'achieved_power_by_comparison']])
print("Note: Control should have non-zero allocation since (0,1) needs it")
print()

print("All tests completed successfully!")
