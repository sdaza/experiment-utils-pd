"""
Example: Using find_sample_size to determine required sample size for target power
"""

from experiment_utils import PowerSim

# Example 1: Find sample size for proportion metric with 80% power
print("Example 1: Proportion metric")
print("-" * 50)
p = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=1,
    nsim=500,
    alpha=0.05,
    alternative='two-tailed'
)

result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],  # 10% baseline conversion rate
    effect=[0.02],    # 2 percentage point increase (absolute)
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print(result)
print()

# Example 2: Find sample size for average metric with 90% power
print("Example 2: Average metric")
print("-" * 50)
p2 = PowerSim(
    metric='average',
    relative_effect=False,
    variants=1,
    nsim=500,
    alpha=0.05,
    alternative='two-tailed'
)

result2 = p2.find_sample_size(
    target_power=0.90,
    baseline=[100],
    effect=[5],
    standard_deviation=[20],
    min_sample_size=100,
    max_sample_size=10000,
    tolerance=0.02,
    step_size=100
)

print(result2)
print()

# Example 3: Multiple comparisons with different effect sizes
print("Example 3: Multiple variants")
print("-" * 50)
p3 = PowerSim(
    metric='proportion',
    relative_effect=True,
    variants=2,  # 2 treatment variants + 1 control = 3 groups
    nsim=300,
    alpha=0.05,
    alternative='two-tailed'
)

result3 = p3.find_sample_size(
    target_power=0.80,
    baseline=[0.20],
    effect=[0.10, 0.15],  # 10% and 15% relative increases
    min_sample_size=500,
    max_sample_size=20000,
    tolerance=0.02,
    step_size=500
)

print(result3)
print()

# Example 4: Custom allocation ratio (30% control, 70% treatment)
print("Example 4: Custom allocation - 30/70 split")
print("-" * 50)
p4 = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=1,
    nsim=500,
    alpha=0.05,
    alternative='two-tailed'
)

result4 = p4.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02],
    allocation_ratio=[0.3, 0.7],  # 30% control, 70% treatment
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print(result4)
print()

# Example 5: Unequal allocation with 3 groups
print("Example 5: Three groups with custom allocation")
print("-" * 50)
p5 = PowerSim(
    metric='average',
    relative_effect=False,
    variants=2,  # 2 variants + 1 control
    nsim=300,
    alpha=0.05,
    alternative='two-tailed'
)

result5 = p5.find_sample_size(
    target_power=0.85,
    baseline=[100],
    effect=[5, 8],
    standard_deviation=[20],
    allocation_ratio=[0.5, 0.25, 0.25],  # 50% control, 25% each variant
    min_sample_size=500,
    max_sample_size=20000,
    tolerance=0.02,
    step_size=500
)

print(result5)
print()

# Example 6: Multiple comparison corrections - Bonferroni vs FDR
print("Example 6: Multiple comparison corrections")
print("-" * 50)

# With Bonferroni correction (more conservative)
p6a = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=2,  # 3 total groups = 3 pairwise comparisons
    nsim=300,
    alpha=0.05,
    alternative='two-tailed',
    correction='bonferroni'  # Conservative correction
)

result6a = p6a.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02, 0.025],
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print("With Bonferroni correction:")
print(result6a)
print()

# With FDR correction (less conservative)
p6b = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=2,
    nsim=300,
    alpha=0.05,
    alternative='two-tailed',
    correction='fdr'  # False Discovery Rate correction
)

result6b = p6b.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02, 0.025],
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print("With FDR correction:")
print(result6b)
print()

# No correction
p6c = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=2,
    nsim=300,
    alpha=0.05,
    alternative='two-tailed',
    correction=None  # No multiple comparison correction
)

result6c = p6c.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02, 0.025],
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print("Without multiple comparison correction:")
print(result6c)
print("\nNote: Bonferroni requires larger samples than FDR, which requires larger than no correction")
print()

# Example 7: Specify custom comparisons with 4 variants
print("Example 7: Custom comparisons with multiple variants")
print("-" * 50)

# With 4 variants, you have 5 total groups (indices 0-4)
# Group 0 = control, Groups 1-4 = variants
# By default, this would create 10 pairwise comparisons: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
# But you can specify only the ones you want

# Only compare each variant to control (4 comparisons instead of 10)
p7 = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=4,
    comparisons=[(0, 1), (0, 2), (0, 3), (0, 4)],  # Only control vs each variant
    nsim=300,
    alpha=0.05,
    alternative='two-tailed',
    correction='bonferroni'
)

result7 = p7.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02, 0.025, 0.03, 0.035],  # Different effects for each variant
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print("Only control vs variants (4 comparisons):")
print(result7)
print()

# Or compare specific variants to each other
p8 = PowerSim(
    metric='proportion',
    relative_effect=False,
    variants=4,
    comparisons=[(0, 1), (0, 4), (1, 4)],  # Control vs variant1, control vs variant4, variant1 vs variant4
    nsim=300,
    alpha=0.05,
    alternative='two-tailed',
    correction='bonferroni'
)

result8 = p8.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.02, 0.025, 0.03, 0.035],
    min_sample_size=1000,
    max_sample_size=50000,
    tolerance=0.02,
    step_size=500
)

print("Custom comparisons: (0 vs 1), (0 vs 4), (1 vs 4):")
print(result8)
print("\nNote: Fewer comparisons means less stringent correction, thus smaller required sample sizes")
