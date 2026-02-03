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

# %% Check balance diagnostics
print("\n" + "="*80)
print("BALANCE DIAGNOSTICS FOR FIRST TEST CASE")
print("="*80)

# Unadjusted balance (before PS weighting)
if analyzer.balance is not None:
    print("\nUnadjusted Balance (before PS weighting):")
    print(analyzer.balance.to_string())
    print(f"\nBalance summary:")
    print(f"  - Balanced covariates: {analyzer.balance['balance_flag'].sum()}/{len(analyzer.balance)}")
    print(f"  - Mean |SMD|: {analyzer.balance['smd'].abs().mean():.4f}")
    print(f"  - Max |SMD|: {analyzer.balance['smd'].abs().max():.4f}")
else:
    print("No balance information available")

# %%
# Adjusted balance (after PS weighting)
if analyzer.adjusted_balance is not None:
    print("\n" + "="*80)
    print("Adjusted Balance (after PS weighting):")
    print(analyzer.adjusted_balance.to_string())
    print(f"\nAdjusted balance summary:")
    print(f"  - Balanced covariates: {analyzer.adjusted_balance['balance_flag'].sum()}/{len(analyzer.adjusted_balance)}")
    print(f"  - Mean |SMD|: {analyzer.adjusted_balance['smd'].abs().mean():.4f}")
    print(f"  - Max |SMD|: {analyzer.adjusted_balance['smd'].abs().max():.4f}")
    
    # Compare improvement
    if analyzer.balance is not None:
        print(f"\nImprovement:")
        before_mean_smd = analyzer.balance['smd'].abs().mean()
        after_mean_smd = analyzer.adjusted_balance['smd'].abs().mean()
        improvement = before_mean_smd - after_mean_smd
        pct_improvement = 100 * improvement / before_mean_smd
        print(f"  - Mean |SMD| reduction: {improvement:.4f} ({pct_improvement:.1f}% improvement)")
else:
    print("No adjusted balance information available")

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

# %% =============================================================================
# P-value adjustments and relative CIs comparison
# Interactive test for multiple comparison procedures (Bonferroni, Holm, Sidak, FDR)
# and relative confidence intervals with bootstrap vs non-bootstrap methods
# =============================================================================

# %% Data generation function
def sample_multi_treatment_data(n_per_group=400, random_seed=42):
    """
    Generate sample data with multiple treatment groups (categorical).
    
    Creates control + 3 treatment variants with different effect sizes
    to demonstrate p-value adjustments and relative CIs.
    
    Parameters
    ----------
    n_per_group : int
        Number of samples per treatment group
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Data with treatment groups and outcomes
    """
    np.random.seed(random_seed)
    
    n_total = n_per_group * 4
    
    # Four treatment groups: control, treatment_a, treatment_b, treatment_c
    groups = (['control'] * n_per_group + 
              ['treatment_a'] * n_per_group + 
              ['treatment_b'] * n_per_group + 
              ['treatment_c'] * n_per_group)
    
    # Baseline characteristics (beta distribution for realistic conversion rates)
    baseline_conversion = np.random.beta(2, 5, n_total)
    
    # Additional covariates for balance checking
    # Age: simulate some imbalance to make balance diagnostics interesting
    age_means = {'control': 35, 'treatment_a': 36, 'treatment_b': 35.5, 'treatment_c': 34}
    age = []
    for group in groups:
        age.append(np.random.normal(age_means[group], 10))
    age = np.array(age)
    
    # Prior purchases: binary covariate
    prior_purchase = np.random.binomial(1, 0.4, n_total)
    
    # Platform: categorical covariate (will create slight imbalance)
    platform_probs = {
        'control': [0.5, 0.3, 0.2],
        'treatment_a': [0.45, 0.35, 0.2],
        'treatment_b': [0.5, 0.3, 0.2],
        'treatment_c': [0.48, 0.32, 0.2]
    }
    platform = []
    for group in groups:
        platform.append(np.random.choice(['mobile', 'web', 'app'], p=platform_probs[group]))
    
    # Treatment effects - different magnitudes to show adjustment impact
    # Outcomes influenced by baseline + covariates + treatment
    effects = {
        'control': 0.0,
        'treatment_a': 0.04,      # Small effect
        'treatment_b': 0.08,      # Moderate effect
        'treatment_c': 0.035      # Small effect
    }
    
    # Generate binary outcomes based on baseline + treatment effect + covariate effects
    conversion = []
    for i, group in enumerate(groups):
        # Base probability from baseline
        prob = baseline_conversion[i]
        # Add treatment effect
        prob += effects[group]
        # Add small age effect (older users convert slightly better)
        prob += (age[i] - 35) * 0.002
        # Prior purchase effect
        prob += 0.05 * prior_purchase[i]
        # Clip to valid probability
        prob = np.clip(prob, 0, 1)
        conversion.append(1 if np.random.rand() < prob else 0)
    
    data = pd.DataFrame({
        'experiment_id': 1,
        'treatment': groups,
        'conversion': conversion,
        'baseline_conversion': baseline_conversion,
        'age': age,
        'prior_purchase': prior_purchase,
        'platform': platform
    })
    
    return data

# %% Generate data
df_mcp = sample_multi_treatment_data(n_per_group=400, random_seed=123)
print("Sample sizes by treatment:")
print(df_mcp['treatment'].value_counts())
print("\nConversion rates by treatment:")
print(df_mcp.groupby('treatment')['conversion'].mean().sort_index())
print("\nCovariate summary statistics:")
print(df_mcp.groupby('treatment')[['age', 'prior_purchase']].mean())
print("\nPlatform distribution:")
print(pd.crosstab(df_mcp['treatment'], df_mcp['platform'], normalize='index').round(3))

# %% =============================================================================
# CHECK BALANCE: Random Assignment Quality
# Assess whether randomization achieved balance across treatment groups
# =============================================================================
print("\n" + "="*80)
print("BALANCE DIAGNOSTICS: Random Assignment Quality Check")
print("="*80)

# Create analyzer with covariates to check balance
analyzer_balance_check = ExperimentAnalyzer(
    data=df_mcp,
    outcomes='conversion',
    treatment_col='treatment',
    experiment_identifier='experiment_id',
    covariates=['age', 'prior_purchase', 'platform'],  # Include covariates for balance check
    bootstrap=False,
    pvalue_adjustment=None
)

# %%
analyzer_balance_check.get_effects()

# Check balance
print("\n" + "="*80)
print("UNADJUSTED BALANCE (after random assignment)")
print("="*80)
print("\nStandardized Mean Differences (SMD) for each covariate:")
print("  - SMD < 0.1: Well balanced (✓)")
print("  - SMD >= 0.1: Imbalanced (✗)")
print("\nBalance by comparison:")

if analyzer_balance_check.balance is not None:
    balance_df = analyzer_balance_check.balance.copy()
    print(f"\n{balance_df.to_string()}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Balance Summary:")
    print("="*80)
    n_balanced = balance_df['balance_flag'].sum()
    n_total = len(balance_df)
    pct_balanced = 100 * n_balanced / n_total
    print(f"Balanced covariates: {n_balanced}/{n_total} ({pct_balanced:.1f}%)")
    print(f"Mean absolute SMD: {balance_df['smd'].abs().mean():.4f}")
    print(f"Max absolute SMD: {balance_df['smd'].abs().max():.4f}")
    
    # Flag problematic comparisons
    imbalanced = balance_df[balance_df['balance_flag'] == 0]
    if not imbalanced.empty:
        print(f"\n⚠️  WARNING: {len(imbalanced)} imbalanced covariate(s) detected:")
        print(imbalanced[['covariate', 'smd']].to_string(index=False))
    else:
        print("\n✓ All covariates are well balanced!")
else:
    print("No balance information available.")

# %% =============================================================================
# BALANCE ADJUSTMENT: Improve covariate balance using propensity scores
# =============================================================================
print("\n" + "="*80)
print("BALANCE ADJUSTMENT: Using Propensity Score Weighting")
print("="*80)

# Run analysis with balance adjustment
analyzer_with_balance_adj = ExperimentAnalyzer(
    data=df_mcp,
    outcomes='conversion',
    treatment_col='treatment',
    experiment_identifier='experiment_id',
    covariates=['age', 'prior_purchase', 'platform'],
    regression_covariates=['age', 'prior_purchase', 'platform'],
    adjustment='balance',
    balance_method='ps-logistic',  # Propensity score weighting
    bootstrap=False,
    pvalue_adjustment=None
)

# %%
analyzer_with_balance_adj.get_effects()

# Check adjusted balance
print("\n" + "="*80)
print("ADJUSTED BALANCE (after propensity score weighting)")
print("="*80)

if analyzer_with_balance_adj.adjusted_balance is not None:
    adj_balance_df = analyzer_with_balance_adj.adjusted_balance.copy()
    print(f"\n{adj_balance_df.to_string()}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("Adjusted Balance Summary:")
    print("="*80)
    n_balanced_adj = adj_balance_df['balance_flag'].sum()
    n_total_adj = len(adj_balance_df)
    pct_balanced_adj = 100 * n_balanced_adj / n_total_adj
    print(f"Balanced covariates: {n_balanced_adj}/{n_total_adj} ({pct_balanced_adj:.1f}%)")
    print(f"Mean absolute SMD: {adj_balance_df['smd'].abs().mean():.4f}")
    print(f"Max absolute SMD: {adj_balance_df['smd'].abs().max():.4f}")
    
    # Compare before/after
    if analyzer_balance_check.balance is not None:
        print("\n" + "="*80)
        print("IMPROVEMENT FROM BALANCE ADJUSTMENT:")
        print("="*80)
        
        # Merge balance and adjusted balance for comparison
        balance_before = analyzer_balance_check.balance.copy()
        balance_after = adj_balance_df.copy()
        
        # Get common covariates for fair comparison (focus on first comparison as example)
        if 'experiment_id' in balance_before.columns:
            # Take first comparison group as example
            exp_ids = balance_before['experiment_id'].unique()
            if len(exp_ids) > 0:
                example_exp = exp_ids[0]
                before_example = balance_before[balance_before['experiment_id'] == example_exp]
                after_example = balance_after[balance_after['experiment_id'] == example_exp]
                
                comparison = pd.DataFrame({
                    'covariate': before_example['covariate'].values,
                    'smd_before': before_example['smd'].values,
                    'smd_after': after_example['smd'].values,
                })
                comparison['improvement'] = comparison['smd_before'].abs() - comparison['smd_after'].abs()
                comparison['pct_improvement'] = 100 * comparison['improvement'] / comparison['smd_before'].abs()
                
                print(f"\nExample comparison (experiment_id={example_exp}):")
                print(comparison.round(4).to_string(index=False))
                
                avg_improvement = comparison['improvement'].mean()
                print(f"\nAverage absolute SMD reduction: {avg_improvement:.4f}")
                print(f"Average % improvement: {comparison['pct_improvement'].mean():.1f}%")
    
    # Check effective sample size reduction
    results_balanced = analyzer_with_balance_adj.results
    if 'ess_treatment_reduction' in results_balanced.columns:
        print("\n" + "="*80)
        print("EFFECTIVE SAMPLE SIZE (ESS) After Weighting:")
        print("="*80)
        ess_cols = ['treatment_group', 'control_group', 'ess_treatment', 'ess_control', 
                    'ess_treatment_reduction', 'ess_control_reduction']
        print(results_balanced[ess_cols].to_string(index=False))
        print("\nNote: ESS reduction shows the 'cost' of achieving balance through weighting")
else:
    print("No adjusted balance information available.")

# %% =============================================================================
# Non-Bootstrap Analysis - Using Fieller's method for relative CIs
# =============================================================================
print("\n" + "="*80)
print("NON-BOOTSTRAP ANALYSIS (Fieller's method for relative CIs)")
print("="*80)

analyzer_no_boot = ExperimentAnalyzer(
    data=df_mcp,
    outcomes='conversion',
    treatment_col='treatment',
    experiment_identifier='experiment_id',
    bootstrap=False,
    pvalue_adjustment=None  # We'll apply adjustments manually to compare methods
)

# %%
analyzer_no_boot.get_effects()
results_no_boot = analyzer_no_boot.results.copy()

# Display raw results
print("\nRaw results (no adjustment):")
cols_display = ['treatment_group', 'control_group', 'absolute_effect', 
                'relative_effect', 'pvalue', 'stat_significance',
                'rel_effect_lower', 'rel_effect_upper']
print(results_no_boot[cols_display].round(4))

# %% =============================================================================
# Bootstrap Analysis - Using percentile method for relative CIs
# =============================================================================
print("\n" + "="*80)
print("BOOTSTRAP ANALYSIS (Percentile method for relative CIs)")
print("="*80)

analyzer_boot = ExperimentAnalyzer(
    data=df_mcp,
    outcomes='conversion',
    treatment_col='treatment',
    experiment_identifier='experiment_id',
    bootstrap=True,
    bootstrap_iterations=1000,
    pvalue_adjustment=None
)

# %%
analyzer_boot.get_effects()
results_boot = analyzer_boot.results.copy()

# Display bootstrap results
print("\nBootstrap results (no adjustment):")
print(results_boot[cols_display].round(4))

# %% =============================================================================
# Compare Bootstrap vs Non-Bootstrap Relative CIs
# =============================================================================
print("\n" + "="*80)
print("COMPARISON: Bootstrap vs Non-Bootstrap Relative CIs")
print("="*80)

comparison_ci = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'relative_effect': results_no_boot['relative_effect'],
    'fieller_lower': results_no_boot['rel_effect_lower'],
    'fieller_upper': results_no_boot['rel_effect_upper'],
    'bootstrap_lower': results_boot['rel_effect_lower'],
    'bootstrap_upper': results_boot['rel_effect_upper'],
})
comparison_ci['fieller_width'] = comparison_ci['fieller_upper'] - comparison_ci['fieller_lower']
comparison_ci['bootstrap_width'] = comparison_ci['bootstrap_upper'] - comparison_ci['bootstrap_lower']

print("\nRelative CI comparison:")
print(comparison_ci.round(4))

# %% =============================================================================
# EXPLANATION: MCP P-values and Relative CIs
# =============================================================================
print("\n" + "="*80)
print("EXPLANATION: What are MCP p-values and why no relative CI adjustment?")
print("="*80)

print("""
MCP = Multiple Comparison Procedure

1. WHAT IS pvalue_mcp?
   - The adjusted p-value after applying multiple testing correction
   - Corrects for inflated Type I error when doing multiple comparisons
   - Example: With 3 comparisons at α=0.05, without correction you have 
     ~14.3% chance of at least one false positive (not 5%)

2. HOW ARE MCP P-VALUES COMPUTED?

   Bonferroni:
   - Most conservative
   - p_adj = min(p × k, 1.0) where k = number of comparisons
   - Example: p=0.03, k=3 → p_adj = 0.09
   
   Holm (step-down):
   - Less conservative than Bonferroni
   - Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
   - Compare p(i) to α/(m - i + 1)
   - Adjusted: p_adj(i) = max(min((m-i+1) × p(i), 1), p_adj(i-1))
   - Example: p=[0.01, 0.03, 0.04], k=3
     • p(1): 0.01 × 3 = 0.03
     • p(2): 0.03 × 2 = 0.06
     • p(3): 0.04 × 1 = 0.06 (monotonicity: max with previous)
   
   Sidak:
   - Assumes independence between tests
   - p_adj = 1 - (1 - p)^k
   - Less conservative than Bonferroni when tests are independent
   
   FDR-BH (Benjamini-Hochberg):
   - Controls False Discovery Rate (not FWER)
   - Allows more discoveries, less conservative
   - Uses rank-based threshold: p(i) ≤ (i/m) × α

3. HOW ARE rel_effect_lower_mcp AND rel_effect_upper_mcp COMPUTED?
   
   Problem: Relative effect = coefficient / intercept (ratio of two estimates)
   
   - Absolute effect CI: Easy to adjust
     CI_adj = estimate ± z_adj × SE
     where z_adj uses adjusted alpha (e.g., α/k for Bonferroni)
   
   - Relative effect CI: Requires full covariance information
     • Need: SE(coefficient), SE(intercept), AND Cov(coefficient, intercept)
     • Fieller's method uses quadratic formula with covariance
     • This information IS NOW STORED in results (se_intercept, cov_coef_intercept)
     • Cannot simply use z_adj × SE for a ratio
   
   Solution implemented:
   - Estimators now store: se_intercept, cov_coef_intercept
   - adjust_pvalues() uses these to compute Fieller CIs with adjusted alpha
   - Result: Proper adjusted relative CIs that account for ratio uncertainty!
   
   Note: For bootstrap results, covariance info may not be available, so 
   adjusted relative CIs will be NaN (use workaround below if needed)

4. ADJUSTED ABSOLUTE CIs (abs_effect_lower_mcp, abs_effect_upper_mcp):
   
   These ARE computed correctly:
   - Use adjusted critical values from adjusted alpha
   - For Bonferroni: z_crit = norm.ppf(1 - α/(2k))
   - For Holm: Use conservative Bonferroni α/k (no single per-comparison alpha)
   - Result: Wider CIs that maintain family-wise coverage
""")

# %% =============================================================================
# Apply Multiple Comparison Adjustments
# Test Bonferroni, Holm, Sidak, and FDR-BH methods
# =============================================================================
print("\n" + "="*80)
print("MULTIPLE COMPARISON ADJUSTMENTS")
print("="*80)

adjustment_methods = ['bonferroni', 'holm', 'sidak', 'fdr_bh']

# %% Show manual calculation of MCP p-values for understanding
print("\nMANUAL CALCULATION EXAMPLE (for educational purposes):")
print("="*80)

raw_pvals = results_no_boot['pvalue'].values
k = len(raw_pvals)
print(f"\nRaw p-values: {raw_pvals}")
print(f"Number of comparisons (k): {k}")

# Bonferroni
bonf_pvals = np.minimum(raw_pvals * k, 1.0)
print(f"\nBonferroni adjusted p-values:")
for i, (raw, adj) in enumerate(zip(raw_pvals, bonf_pvals)):
    print(f"  Comparison {i+1}: {raw:.6f} × {k} = {adj:.6f}")

# Holm
sorted_idx = np.argsort(raw_pvals)
sorted_pvals = raw_pvals[sorted_idx]
holm_pvals_sorted = np.zeros(k)
for i in range(k):
    holm_pvals_sorted[i] = min((k - i) * sorted_pvals[i], 1.0)
    if i > 0:
        holm_pvals_sorted[i] = max(holm_pvals_sorted[i], holm_pvals_sorted[i-1])
# Reorder back
holm_pvals = np.zeros(k)
holm_pvals[sorted_idx] = holm_pvals_sorted

print(f"\nHolm adjusted p-values (step-down procedure):")
print(f"  Sorted raw p-values: {sorted_pvals}")
for i, (raw, adj) in enumerate(zip(sorted_pvals, holm_pvals_sorted)):
    multiplier = k - i
    print(f"  p({i+1}): {raw:.6f} × {multiplier} = {raw*multiplier:.6f} → {adj:.6f} (after monotonicity)")

# Sidak
sidak_pvals = 1 - (1 - raw_pvals) ** k
print(f"\nSidak adjusted p-values:")
for i, (raw, adj) in enumerate(zip(raw_pvals, sidak_pvals)):
    print(f"  Comparison {i+1}: 1 - (1 - {raw:.6f})^{k} = {adj:.6f}")

print("\nNote: FDR-BH uses a more complex rank-based procedure")
print("="*80)

# Store results for each method
results_by_method = {}

for method in adjustment_methods:
    print(f"\n{'='*80}")
    print(f"Applying {method.upper()} adjustment")
    print('='*80)
    
    # Apply to non-bootstrap results
    analyzer_no_boot_copy = ExperimentAnalyzer(
        data=df_mcp,
        outcomes='conversion',
        treatment_col='treatment',
        experiment_identifier='experiment_id',
        bootstrap=False,
        pvalue_adjustment=None
    )
    analyzer_no_boot_copy.get_effects()
    analyzer_no_boot_copy.adjust_pvalues(method=method)
    
    results_by_method[method] = analyzer_no_boot_copy.results.copy()
    
    # Display key columns
    cols_adjusted = ['treatment_group', 'control_group', 'pvalue', 'pvalue_mcp', 
                     'stat_significance', 'stat_significance_mcp', 'mcp_method']
    print(f"\n{method.upper()} adjusted results:")
    print(results_by_method[method][cols_adjusted].round(6))

# %% =============================================================================
# Side-by-Side Comparison of All Adjustment Methods
# =============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON: All Adjustment Methods")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'relative_effect': results_no_boot['relative_effect'],
    'raw_pvalue': results_no_boot['pvalue'],
    'raw_sig': results_no_boot['stat_significance']
})

# Add adjusted p-values from each method
for method in adjustment_methods:
    comparison_df[f'{method}_pvalue'] = results_by_method[method]['pvalue_mcp']
    comparison_df[f'{method}_sig'] = results_by_method[method]['stat_significance_mcp']

print("\nP-value comparison across methods:")
print(comparison_df.round(6))

# %% Summary statistics
print("\n" + "="*80)
print("SUMMARY: Significance Changes After Adjustment")
print("="*80)

summary = pd.DataFrame({
    'method': ['raw'] + adjustment_methods,
    'n_significant': [
        results_no_boot['stat_significance'].sum()
    ] + [
        results_by_method[m]['stat_significance_mcp'].sum() 
        for m in adjustment_methods
    ]
})
summary['n_comparisons'] = len(results_no_boot)
summary['prop_significant'] = summary['n_significant'] / summary['n_comparisons']

print("\nNumber of significant comparisons by method:")
print(summary)

# %% =============================================================================
# Adjusted Confidence Intervals Comparison (Absolute and Relative)
# =============================================================================
print("\n" + "="*80)
print("ADJUSTED CONFIDENCE INTERVALS")
print("="*80)

# Compare adjusted CIs across methods - ABSOLUTE EFFECTS
ci_comparison_abs = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'absolute_effect': results_no_boot['absolute_effect'],
    'raw_lower': results_no_boot['abs_effect_lower'],
    'raw_upper': results_no_boot['abs_effect_upper'],
})

for method in adjustment_methods:
    ci_comparison_abs[f'{method}_lower'] = results_by_method[method]['abs_effect_lower_mcp']
    ci_comparison_abs[f'{method}_upper'] = results_by_method[method]['abs_effect_upper_mcp']
    ci_comparison_abs[f'{method}_width'] = (
        results_by_method[method]['abs_effect_upper_mcp'] - 
        results_by_method[method]['abs_effect_lower_mcp']
    )

print("\nAbsolute effect CI comparison:")
display_cols = ['comparison', 'absolute_effect', 'raw_lower', 'raw_upper'] + \
               [f'{m}_lower' for m in adjustment_methods] + \
               [f'{m}_upper' for m in adjustment_methods]
print(ci_comparison_abs[display_cols].round(6))

print("\nAbsolute CI widths by method:")
ci_comparison_abs['raw_width'] = ci_comparison_abs['raw_upper'] - ci_comparison_abs['raw_lower']
width_summary = ci_comparison_abs[['comparison', 'raw_width'] + [f'{m}_width' for m in adjustment_methods]]
print(width_summary.round(6))

# Compare adjusted CIs across methods - RELATIVE EFFECTS
print("\n" + "="*80)
print("ADJUSTED RELATIVE EFFECT CIs (Now properly computed!)")
print("="*80)

ci_comparison_rel = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'relative_effect': results_no_boot['relative_effect'],
    'raw_lower': results_no_boot['rel_effect_lower'],
    'raw_upper': results_no_boot['rel_effect_upper'],
})

for method in adjustment_methods:
    ci_comparison_rel[f'{method}_lower'] = results_by_method[method]['rel_effect_lower_mcp']
    ci_comparison_rel[f'{method}_upper'] = results_by_method[method]['rel_effect_upper_mcp']
    # Calculate width only for non-NaN values
    rel_lower = results_by_method[method]['rel_effect_lower_mcp']
    rel_upper = results_by_method[method]['rel_effect_upper_mcp']
    ci_comparison_rel[f'{method}_width'] = rel_upper - rel_lower

print("\nRelative effect CI comparison (using Fieller's method with adjusted alpha):")
display_cols_rel = ['comparison', 'relative_effect', 'raw_lower', 'raw_upper'] + \
                   [f'{m}_lower' for m in adjustment_methods] + \
                   [f'{m}_upper' for m in adjustment_methods]
print(ci_comparison_rel[display_cols_rel].round(6))

print("\nRelative CI widths by method:")
ci_comparison_rel['raw_width'] = ci_comparison_rel['raw_upper'] - ci_comparison_rel['raw_lower']
width_summary_rel = ci_comparison_rel[['comparison', 'raw_width'] + [f'{m}_width' for m in adjustment_methods]]
print(width_summary_rel.round(6))

print("\nNote: Adjusted relative CIs are computed using Fieller's method with adjusted alpha.")
print("This properly accounts for uncertainty in both numerator (treatment effect)")
print("and denominator (control mean), including their covariance.")

# %% =============================================================================
# Bootstrap + Adjustment Methods
# =============================================================================
print("\n" + "="*80)
print("BOOTSTRAP WITH ADJUSTMENT METHODS")
print("="*80)

# Apply adjustments to bootstrap results
bootstrap_results_by_method = {}

for method in adjustment_methods:
    analyzer_boot_copy = ExperimentAnalyzer(
        data=df_mcp,
        outcomes='conversion',
        treatment_col='treatment',
        experiment_identifier='experiment_id',
        bootstrap=True,
        bootstrap_iterations=1000,
        pvalue_adjustment=None
    )
    analyzer_boot_copy.get_effects()
    analyzer_boot_copy.adjust_pvalues(method=method)
    
    bootstrap_results_by_method[method] = analyzer_boot_copy.results.copy()
    
    print(f"\n{method.upper()} with bootstrap:")
    cols_boot_adj = ['treatment_group', 'control_group', 'pvalue', 'pvalue_mcp', 
                     'stat_significance_mcp']
    print(bootstrap_results_by_method[method][cols_boot_adj].round(6))

# %% Final comparison: Bootstrap vs Non-Bootstrap with Holm adjustment
print("\n" + "="*80)
print("FINAL COMPARISON: Bootstrap vs Non-Bootstrap with HOLM adjustment")
print("="*80)

final_comparison = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'relative_effect': results_no_boot['relative_effect'],
    'no_boot_pvalue': results_by_method['holm']['pvalue_mcp'],
    'no_boot_sig': results_by_method['holm']['stat_significance_mcp'],
    'bootstrap_pvalue': bootstrap_results_by_method['holm']['pvalue_mcp'],
    'bootstrap_sig': bootstrap_results_by_method['holm']['stat_significance_mcp'],
    'rel_ci_lower_no_boot': results_by_method['holm']['rel_effect_lower'],
    'rel_ci_upper_no_boot': results_by_method['holm']['rel_effect_upper'],
    'rel_ci_lower_boot': bootstrap_results_by_method['holm']['rel_effect_lower'],
    'rel_ci_upper_boot': bootstrap_results_by_method['holm']['rel_effect_upper'],
})

print("\nHolm-adjusted results comparison:")
print(final_comparison.round(6))

# %% =============================================================================
# ALTERNATIVE APPROACH: Re-running with Adjusted Alpha
# (Now optional since adjust_pvalues computes relative CIs properly)
# Still useful for bootstrap results where covariance info unavailable
# =============================================================================
print("\n" + "="*80)
print("ALTERNATIVE APPROACH: Re-run with Adjusted Alpha")
print("="*80)

print("\nThe adjust_pvalues() method now computes adjusted relative CIs properly!")
print("However, you can also re-run get_effects() with adjusted alpha.")
print("This approach is useful for:")
print("  - Bootstrap results (where covariance info may not be available)")
print("  - When you want CIs directly at the adjusted level")
print("\nMethod: Re-run get_effects() with adjusted alpha parameter")

# Example: Bonferroni adjustment with k=3 comparisons
k = len(results_no_boot)
alpha_adjusted = 0.05 / k  # Bonferroni adjustment

print(f"\nOriginal alpha: 0.05")
print(f"Number of comparisons: {k}")
print(f"Bonferroni adjusted alpha: {alpha_adjusted:.6f}")

# Re-run analyzer with adjusted alpha
analyzer_adjusted_alpha = ExperimentAnalyzer(
    data=df_mcp,
    outcomes='conversion',
    treatment_col='treatment',
    experiment_identifier='experiment_id',
    bootstrap=False,
    alpha=alpha_adjusted,  # Use adjusted alpha
    pvalue_adjustment=None
)

# %%
analyzer_adjusted_alpha.get_effects()
results_adjusted_alpha = analyzer_adjusted_alpha.results.copy()

# Compare standard vs adjusted alpha CIs
print("\nComparison: Standard alpha (0.05) vs Bonferroni adjusted alpha")
print("="*80)

comparison_adjusted = pd.DataFrame({
    'comparison': results_no_boot['treatment_group'] + ' vs ' + results_no_boot['control_group'],
    'relative_effect': results_no_boot['relative_effect'],
    # Standard alpha CIs
    'rel_ci_lower_α005': results_no_boot['rel_effect_lower'],
    'rel_ci_upper_α005': results_no_boot['rel_effect_upper'],
    # Adjusted alpha CIs (proper Bonferroni adjustment)
    'rel_ci_lower_adj': results_adjusted_alpha['rel_effect_lower'],
    'rel_ci_upper_adj': results_adjusted_alpha['rel_effect_upper'],
})

comparison_adjusted['ci_width_α005'] = (comparison_adjusted['rel_ci_upper_α005'] - 
                                         comparison_adjusted['rel_ci_lower_α005'])
comparison_adjusted['ci_width_adj'] = (comparison_adjusted['rel_ci_upper_adj'] - 
                                        comparison_adjusted['rel_ci_lower_adj'])
comparison_adjusted['width_ratio'] = (comparison_adjusted['ci_width_adj'] / 
                                       comparison_adjusted['ci_width_α005'])

print("\nRelative effect CIs with proper adjustment:")
print(comparison_adjusted.round(4))

print(f"\nCI width increases by factor of ~{comparison_adjusted['width_ratio'].mean():.2f} on average")
print("This maintains family-wise coverage at the desired level.")

# %% Visualize the difference
print("\n" + "="*80)
print("SUMMARY: How adjust_pvalues() NOW computes relative CIs")
print("="*80)
print("""
The adjust_pvalues() method NOW:
✓ DOES adjust: p-values using statsmodels.multipletests
✓ DOES adjust: Absolute effect CIs using adjusted z-critical values
✓ DOES adjust: Relative effect CIs using Fieller's method with adjusted alpha!

How it works:
- Estimators store: se_intercept, cov_coef_intercept
- adjust_pvalues() uses Fieller's theorem with adjusted alpha
- Properly accounts for uncertainty in coefficient / intercept ratio
- Includes covariance between numerator and denominator

When relative CIs may still be NaN:
1. Bootstrap results (covariance info not available from percentile method)
2. Very small denominators (numerical instability)
3. Degenerate cases (discriminant < 0 in Fieller's quadratic)

For bootstrap + adjustment: Use alternative approach (re-run with adjusted alpha)
""")

print("\n" + "="*80)
print("INTERACTIVE TEST COMPLETE")
print("="*80)
print("\nKey findings:")
print("1. Balance diagnostics: SMD checks randomization quality across covariates")
print("2. Balance adjustment (PS weighting): Improves covariate balance, reduces bias")
print("3. Multiple comparison adjustments (Bonferroni, Holm, Sidak, FDR) inflate p-values")
print("4. Holm is less conservative than Bonferroni but controls FWER")
print("5. Bootstrap provides alternative CI estimation via percentile method")
print("6. Relative CIs shown for both Fieller (non-bootstrap) and percentile (bootstrap)")
print("7. Adjusted absolute CIs are wider to maintain family-wise coverage")
print("8. Adjusted relative CIs NOW properly computed using Fieller's method!")
print("9. Covariance information (se_intercept, cov_coef_intercept) stored in results")
print("10. For bootstrap + adjustment, use alternative approach (re-run with adjusted alpha)")

print("\n" + "="*80)
print("BALANCE DIAGNOSTICS SUMMARY")
print("="*80)
print("""
Balance checking is crucial for assessing randomization quality:

1. Standardized Mean Difference (SMD):
   - Measures difference in covariate means between treatment groups
   - SMD < 0.1: Generally considered well balanced
   - SMD >= 0.1: May indicate imbalance requiring adjustment

2. Balance Adjustment Methods:
   - Propensity Score (PS) weighting: Reweights units to achieve balance
   - Entropy balancing: Optimizes weights to match moment constraints
   - Trade-off: Improves balance but reduces effective sample size (ESS)

3. When to use balance adjustment:
   - Observational studies (always)
   - Randomized experiments with imbalance (optional, for precision)
   - When pre-treatment covariates predict outcomes strongly

4. Accessing balance information:
   - analyzer.balance: Shows unadjusted SMDs
   - analyzer.adjusted_balance: Shows SMDs after weighting
   - Compare before/after to assess adjustment effectiveness
""")