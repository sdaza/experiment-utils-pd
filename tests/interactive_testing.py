# interactive testing
# %% experiment analyzer
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# %%
def sample_data(
    n_model=5000,
    n_random=1000,
    base_model_conversion_mean=0.3,
    base_model_conversion_variance=0.01,
    base_random_conversion_mean=0.1,
    base_random_conversion_variance=0.01,
    model_treatment_effect=0.10,
    random_treatment_effect=0.05,
    random_seed=42,
):

    np.random.seed(random_seed)

    # Function to get a truncated normal distribution
    def get_truncated_normal(mean, variance, size):
        std_dev = np.sqrt(variance)
        lower, upper = 0, 1
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

    # Generate baseline conversions with a truncated normal distribution
    base_model_conversion = get_truncated_normal(
        base_model_conversion_mean, base_model_conversion_variance, n_model
    )
    base_random_conversion = get_truncated_normal(
        base_random_conversion_mean, base_random_conversion_variance, n_random
    )

    # model group data
    model_treatment = np.random.binomial(1, 0.8, n_model)
    model_conversion = (
        base_model_conversion + model_treatment_effect * model_treatment
    ) > np.random.rand(n_model)

    model_data = pd.DataFrame(
        {
            "experiment": 123,
            "expected_ratio": 0.8,
            "group": "model",
            "treatment": model_treatment,
            "conversion": model_conversion.astype(int),
            "baseline_conversion": base_model_conversion,
        }
    )

    # random group data
    random_treatment = np.random.binomial(1, 0.5, n_random)
    random_conversion = (
        base_random_conversion + random_treatment_effect * random_treatment
    ) > np.random.rand(n_random)
    random_data = pd.DataFrame(
        {
            "experiment": 123,
            "expected_ratio": 0.5,
            "group": "random",
            "treatment": random_treatment,
            "conversion": random_conversion.astype(int),
            "baseline_conversion": base_random_conversion,
        }
    )

    # Combine data
    data = pd.concat([model_data, random_data])

    return data

df = sample_data()

# %%
outcomes = "conversion"
treatment_col = "treatment"
experiment_identifier = "group"
covariates = "baseline_conversion"

analyzer = ExperimentAnalyzer(
    data=df,
    outcomes=outcomes,
    treatment_col=treatment_col,
    exp_sample_ratio_col='expected_ratio',
    experiment_identifier=experiment_identifier,
    covariates=covariates,
    regression_covariates=['baseline_conversion'],
    adjustment="balance",
    balance_method="ps-logistic")

analyzer.get_effects()
analyzer.results

# %%
analyzer.adjust_pvalues(method='sidak')
analyzer.results
# %%
analyzer = ExperimentAnalyzer(
    data=df,
    outcomes=outcomes,
    treatment_col=treatment_col,
    exp_sample_ratio_col='expected_ratio',
    experiment_identifier=experiment_identifier,
    covariates=covariates,
    regression_covariates=['baseline_conversion'],
    adjustment="balance",
    balance_method="ps-logistic", 
    bootstrap=True,
    bootstrap_iterations=1000)

analyzer.get_effects()
analyzer.results

# %%
analyzer.adjust_pvalues(method='sidak')
analyzer.results

# %%
# %% load libraries
import numpy as np
import pandas as pd
from experiment_utils.experiment_analyzer import ExperimentAnalyzer  # Import the package

# %% simulate data
np.random.seed(42)

# control 
n_control = 10000
control_install_rate = 0.20
control_installs = np.random.binomial(1, control_install_rate, n_control)

# test
n_test = 10000
test_install_rate = 0.185
test_installs = np.random.binomial(1, test_install_rate, n_test)

# dataframe
df = pd.DataFrame({
    'group': ['control'] * n_control + ['test'] * n_test,
    'installed': list(control_installs) + list(test_installs)
})
df['treatment'] = df['group'].apply(lambda x: 1 if x == 'test' else 0)

# %% analysis using the experiment analyzer
analyzer = ExperimentAnalyzer(
    df,
    treatment_col="treatment",
    outcomes=['installed'],
)

# %% explore results
analyzer.get_effects()
rr = analyzer.results
print(rr)

# %% 
analyzer.test_non_inferiority(relative_margin=0.10)

# %% 
print(analyzer.results)


# %% Test multiple comparison adjustment functionality
import numpy as np
import pandas as pd
from experiment_utils.experiment_analyzer import ExperimentAnalyzer

def test_multiple_comparison_adjustment():
    # Simulate data for two outcomes
    np.random.seed(123)
    n = 1000
    df = pd.DataFrame({
        'experiment': np.repeat([1, 2], n),
        'treatment': np.tile(np.random.binomial(1, 0.5, n), 2),
        'outcome1': np.random.normal(0, 1, 2 * n),
        'outcome2': np.random.normal(0, 1, 2 * n)
    })
    outcomes = ['outcome1', 'outcome2']
    analyzer = ExperimentAnalyzer(
        data=df,
        outcomes=outcomes,
        treatment_col='treatment',
        experiment_identifier=['experiment']
    )
    analyzer.get_effects()
    # Before adjustment
    print('Raw p-values:')
    print(analyzer.results[['experiment', 'outcome', 'pvalue']])
    # Apply Bonferroni adjustment
    analyzer.adjust_pvalues(method='bonferroni')
    print('Bonferroni adjusted p-values:')
    print(analyzer.results)
    # Apply FDR adjustment
    analyzer.adjust_pvalues(method='sidak')
    print('FDR adjusted p-values:')
    print(analyzer.results)

test_multiple_comparison_adjustment()

# %%
from experiment_utils.power_sim import PowerSim  # noqa: E402

p = PowerSim(metric='proportion', variants=2, nsim=1000, 
             alpha=0.05)

# %% get exaggertion  ratio
p.simulate_retrodesign(
    true_effect=[0.05, 0.08],
    baseline=[0.10],
    sample_size=300,
)

# %%
p = PowerSim(metric='proportion', variants=1, nsim=5000, 
             alpha=0.05, correction=None)

p.get_power(
    baseline=[0.18],
    effect=[0.03],
    sample_size=[2750])

# %%
p.find_sample_size(
    target_power=0.80,
    baseline=[0.18],
    effect=[0.03],
    min_sample_size=300,
    max_sample_size=8000,
    step_size=100
)

# %%
p.simulate_retrodesign(
    true_effect=[0.05],
    baseline=[0.18],
    sample_size=300
)

# rt
# %%
p.get_power(
    baseline=[0.18],
    effect=[0.03],
    sample_size=[2675],
    )


# %% Multiple variants - same power for all comparisons
p = PowerSim(
    metric='proportion',
    variants=3,
    # comparisons=[(0, 1), (0, 2), (0, 3)],  # Only vs control
    correction=None,
    # correction='holm',
    nsim=1000
)

# %%
result = p.find_sample_size(
    target_power=0.80,  # Same target for all
    baseline=0.18,
    optimize_allocation=True,
    effect=0.05, 
    min_sample_size=300,
    max_sample_size=20000,
)
result

# %% Multiple variants - different power targets per comparison
result = p.find_sample_size(
    target_power={(0, 1): 0.90, (0, 2): 0.80, (0, 3): 0.70},  # Different targets!
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07], 
    min_sample_size=300,
    max_sample_size=10000,
)
result

# %% Only power specific comparisons of interest
result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    target_comparisons=[(1, 2), (3, 2), (1, 3)],  # Only care about first two variants
    min_sample_size=300,
    optimize_allocation=True,
    max_sample_size=10000,
)
result

# %% Power criteria: "any" - need at least one comparison powered
result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    power_criteria="any",  # Less conservative - just need one powered
    min_sample_size=300,
    max_sample_size=10000,
)
result

# %% Power criteria: "all" - need all comparisons powered (default)
result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    power_criteria="all",  # More conservative - all must be powered
    min_sample_size=300,
    max_sample_size=10000,
    a
)
result

# %% Optimize allocation: automatically find best allocation ratios
# With equal allocation, comparisons with larger effects are overpowered
result_equal = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],  # Different effects per variant
    min_sample_size=300,
    max_sample_size=10000,
)
print("Equal allocation:")
print(f"Total sample: {result_equal['total_sample_size'].iloc[0]}")
print(f"Powers: {result_equal['achieved_power_by_comparison'].iloc[0]}")

# %% With optimize_allocation=True, minimizes total sample size
result_optimized = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],  # Different effects per variant
    optimize_allocation=True,  # Find optimal allocation
    min_sample_size=300,
    max_sample_size=10000,
)
print("\nOptimized allocation:")
print(f"Total sample: {result_optimized['total_sample_size'].iloc[0]}")
print(f"Sample sizes: {result_optimized['sample_sizes_by_group'].iloc[0]}")
print(f"Powers: {result_optimized['achieved_power_by_comparison'].iloc[0]}")

# %% Optimize with different power targets per comparison
result_optimized_diff = p.find_sample_size(
    target_power={(0, 1): 0.90, (0, 2): 0.80, (0, 3): 0.85},
    baseline=[0.10],
    effect=[0.05, 0.03, 0.07],
    optimize_allocation=True,
    min_sample_size=300,
    max_sample_size=10000,
)
print("\nOptimized with different power targets:")
print(f"Total sample: {result_optimized_diff['total_sample_size'].iloc[0]}")
print(f"Sample sizes: {result_optimized_diff['sample_sizes_by_group'].iloc[0]}")
print(f"Powers: {result_optimized_diff['achieved_power_by_comparison'].iloc[0]}")

# %%
result = p.find_sample_size(
    target_power=0.80,
    baseline=[0.10],
    effect=[0.01],
    min_sample_size=1000,
    max_sample_size=50000,
)

# %%
print(result)
# %%
p.get_power(
    baseline=[0.10],
    effect=[0.05],
    sample_size=[500000]
)

# %% testing
from experiment_utils.power_sim import PowerSim  # noqa: E402

p = PowerSim(metric="proportion",
             relative_effect=True, nsim=1000,
             alpha=0.05,
             correction='holm')

# %%
p.find_sample_size(
    target_power=0.80,
    baseline=0.0719,
    effect=0.02,
    min_sample_size=10000,
    max_sample_size=1200000,
    step_size=100000
)

# %%
p.get_power(
    baseline=0.0719,
    effect=0.02,
    sample_size=530000
)
# %% create categorical treatment data and analyze
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer


def sample_data_numeric_categorical():
    """
    Generate sample data with numeric categorical treatment (1, 2, 3)
    """
    np.random.seed(42)

    n_per_group = 300

    # Three treatment groups: 1 (control), 2 (treatment_a), 3 (treatment_b)
    groups = [1] * n_per_group + [2] * n_per_group + [3] * n_per_group

    # Baseline characteristics
    baseline_conversion = np.random.beta(2, 5, n_per_group * 3)

    # Treatment effects
    effects = {1: 0.001, 2: 0.03, 3: 0.05}

    # Generate outcomes
    conversion = []
    for i, group in enumerate(groups):
        prob = baseline_conversion[i] + effects[group]
        conversion.append(1 if np.random.rand() < prob else 0)

    data = pd.DataFrame({
        'experiment_id': 1,
        'treatment': groups,
        'conversion': conversion,
        'baseline_conversion': baseline_conversion
    })

    return data

data = sample_data_numeric_categorical()

# %% 
data['new_treatment'] = data['treatment'].map({1: 'control', 2: 'treatment_a', 3: 'treatment_b'})

# %%
analyzer = ExperimentAnalyzer(
        data=data,
        outcomes='conversion',
        treatment_col='new_treatment',
        experiment_identifier='experiment_id', 
        bootstrap=False,
        pvalue_adjustment=None,
        # covariates=['baseline_conversion'],
        # regression_covariates=['baseline_conversion'],
    )

analyzer.get_effects()
results = analyzer.results
results

# %%
retro = analyzer.calculate_retrodesign(
    true_effect={
        ('treatment_a', 'control'): 0.02,
        ('treatment_b', 'control'): 0.20,
        ('treatment_b', 'treatment_a'): 0.20  
    }
)
retro

# %%
analyzer.adjust_pvalues(method='holm')
analyzer.results

# %% categorical features
from experiment_utils.experiment_analyzer import ExperimentAnalyzer
import numpy as np
import pandas as pd

data = pd.DataFrame(
    {
        "outcome": np.random.binomial(1, 0.1,   800),
        "treatment": np.random.choice(['control', 'treatment'], 800),
        "city": np.random.choice(["New York", "Los Angeles", "San-Francisco", "Boston"], 800),
        "tier": np.random.choice(["Premium+", "Standard", "Basic", "Free Trial"], 800),
    }
)

# %%
exp = ExperimentAnalyzer(
    data=data,
    outcomes=["outcome"],
    treatment_col="treatment",
    covariates=["city", "tier"],
    adjustment="balance",
    pvalue_adjustment=None,
    regression_covariates=["city", "tier"],
)
# %%
exp.get_effects()

# %%
exp.balance
# %% 
exp.adjusted_balance

# %%
exp.results

# %% retrodesign
retro = exp.calculate_retrodesign()
# %%
retro
# %%
