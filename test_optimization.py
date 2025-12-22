from experiment_utils.power_sim import PowerSim

p = PowerSim(
    metric='proportion',
    variants=3,
    nsim=2000,
    alpha=0.05,
    correction=None,
)

print('Testing equal allocation:')
result_equal = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    min_sample_size=300,
    max_sample_size=10000,
)
print(f'Total sample: {result_equal["total_sample_size"].iloc[0]}')
print(f'Sample sizes: {result_equal["sample_sizes_by_group"].iloc[0]}')
print(f'Powers: {result_equal["achieved_power_by_comparison"].iloc[0]}')

print('\n\nTesting optimized allocation:')
result_opt = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    optimize_allocation=True,
    min_sample_size=300,
    max_sample_size=10000,
)
print(f'Total sample: {result_opt["total_sample_size"].iloc[0]}')
print(f'Sample sizes: {result_opt["sample_sizes_by_group"].iloc[0]}')
print(f'Powers: {result_opt["achieved_power_by_comparison"].iloc[0]}')

savings = result_equal['total_sample_size'].iloc[0] - result_opt['total_sample_size'].iloc[0]
pct_savings = (savings / result_equal['total_sample_size'].iloc[0]) * 100
print(f'\nSample size reduction: {savings} ({pct_savings:.1f}%)')
