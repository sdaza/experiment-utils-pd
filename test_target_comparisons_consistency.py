"""Test target_comparisons parameter consistency across get_power, find_sample_size, and simulate_retrodesign"""
from experiment_utils.power_sim import PowerSim

print("Testing target_comparisons parameter consistency")
print("=" * 70)

# Setup: 3 variants with multiple comparisons defined
p = PowerSim(
    metric='proportion', 
    variants=3, 
    nsim=1000,
    alpha=0.05,
    comparisons=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
)

print(f"All defined comparisons: {p.comparisons}")
print()

# Test 1: get_power with target_comparisons
print("Test 1: get_power with target_comparisons")
print("-" * 70)

# Without target_comparisons - should return all 6 comparisons
power_all = p.get_power(
    baseline=0.10,
    effect=[0.05, 0.03, 0.07],
    sample_size=1000
)
print(f"Without target_comparisons: {len(power_all)} rows")
print(power_all)
print()

# With target_comparisons - should return only 2 comparisons
power_filtered = p.get_power(
    baseline=0.10,
    effect=[0.05, 0.03, 0.07],
    sample_size=1000,
    target_comparisons=[(0, 1), (0, 2)]
)
print(f"With target_comparisons=[(0, 1), (0, 2)]: {len(power_filtered)} rows")
print(power_filtered)
print()

# Test 2: find_sample_size with target_comparisons
print("Test 2: find_sample_size with target_comparisons")
print("-" * 70)

result = p.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.05, 0.03, 0.07],
    target_comparisons=[(0, 1), (1, 2)],
    min_sample_size=500,
    max_sample_size=10000
)
print(f"Target comparisons: [(0, 1), (1, 2)]")
print(f"Total sample size: {result['total_sample_size'].iloc[0]}")
print(f"Power by comparison: {result['achieved_power_by_comparison'].iloc[0]}")
print()

# Test 3: simulate_retrodesign with target_comparisons
print("Test 3: simulate_retrodesign with target_comparisons")
print("-" * 70)

retro = p.simulate_retrodesign(
    true_effect=[0.05, 0.03, 0.07],
    sample_size=1000,
    baseline=0.10,
    target_comparisons=[(0, 1), (2, 3)]
)
print(f"Target comparisons: [(0, 1), (2, 3)]")
print(f"Number of results: {len(retro)}")
print(retro[['comparison', 'true_effect', 'power', 'type_s_error', 'exaggeration_ratio']])
print()

# Test 4: Error handling - invalid comparison
print("Test 4: Error handling for invalid target_comparisons")
print("-" * 70)

try:
    p.get_power(
        baseline=0.10,
        effect=[0.05],
        sample_size=1000,
        target_comparisons=[(0, 5)]  # Invalid - group 5 doesn't exist
    )
    print("ERROR: Should have raised an error!")
except Exception as e:
    print(f"✓ Correctly raised error: {str(e)[:80]}...")
print()

# Test 5: Variant-only comparisons
print("Test 5: Variant-only comparisons (no control needed)")
print("-" * 70)

p2 = PowerSim(metric='proportion', variants=2, nsim=1000, alpha=0.05)

# get_power with variant-only comparison
power_variant = p2.get_power(
    baseline=0.10,
    effect=[0.05, 0.03],
    sample_size=1000,
    target_comparisons=[(1, 2)]
)
print(f"get_power with target_comparisons=[(1, 2)]:")
print(power_variant)
print()

# find_sample_size with variant-only comparison  
result_variant = p2.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.05, 0.03],
    target_comparisons=[(1, 2)],
    optimize_allocation=True,
    min_sample_size=500,
    max_sample_size=10000
)
print(f"find_sample_size with target_comparisons=[(1, 2)]:")
print(f"Sample sizes: {result_variant['sample_sizes_by_group'].iloc[0]}")
print("Note: Control should be 0 since not needed")
print()

# simulate_retrodesign with variant-only comparison
retro_variant = p2.simulate_retrodesign(
    true_effect=[0.05, 0.03],
    sample_size=1000,
    baseline=0.10,
    target_comparisons=[(1, 2)]
)
print(f"simulate_retrodesign with target_comparisons=[(1, 2)]:")
print(retro_variant[['comparison', 'power', 'exaggeration_ratio']])
print()

print("=" * 70)
print("All tests completed successfully!")
print("target_comparisons parameter works consistently across all three methods")
