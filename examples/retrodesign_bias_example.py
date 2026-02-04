"""
Example demonstrating the difference between type_m_error and relative_bias in retrodesign.

This example shows why relative_bias is typically lower than type_m_error in underpowered studies.
"""

# %% Imports
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer
from experiment_utils.power_sim import PowerSim

# %% Example 1: ExperimentAnalyzer retrodesign
print("1. ExperimentAnalyzer Retrodesign")

# Create underpowered experiment
np.random.seed(42)
df = pd.DataFrame(
    {
        "treatment": np.random.choice([0, 1], 200),
        "outcome": np.random.randn(200) + 0.1,  # Small effect
    }
)

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["outcome"],
)

analyzer.get_effects()

# Calculate retrodesign assuming true effect is 0.05
retro = analyzer.calculate_retrodesign(true_effect=0.05)

print("\nRetrodesign Results:")
print(retro[["outcome", "power", "type_s_error", "type_m_error", "relative_bias"]].to_string())

print("\nInterpretation:")
power = retro["power"].iloc[0]
type_m = retro["type_m_error"].iloc[0]
rel_bias = retro["relative_bias"].iloc[0]

print(f"  Power: {power:.1%} - Low power means high uncertainty")
print(f"  Type M Error: {type_m:.2f}x - Significant results overestimate by {type_m:.2f}x on average")
print(f"  Relative Bias: {rel_bias:.2f}x - Accounting for sign, bias is {rel_bias:.2f}x")
print("  \nWhy relative_bias < type_m_error:")
print("    - Type M uses absolute values: mean(|observed|/|true|)")
print("    - Relative bias preserves signs: mean(observed/true)")
print("    - When underpowered, some significant results have wrong sign (Type S errors)")
print("    - These negative values partially offset positive overestimates")

# %% Example 2: PowerSim retrodesign simulation
print("\n2. PowerSim Retrodesign Simulation")

power_sim = PowerSim(metric="proportion", variants=1, nsim=10000)

# Very underpowered study
retro_underpowered = power_sim.simulate_retrodesign(true_effect=0.02, sample_size=300, baseline=0.10)

# Well-powered study
retro_powered = power_sim.simulate_retrodesign(true_effect=0.02, sample_size=3000, baseline=0.10)

print("\nUnderpowered Study (n=300):")
print(f"  Power: {retro_underpowered['power'].iloc[0]:.1%}")
print(f"  Type S Error: {retro_underpowered['type_s_error'].iloc[0]:.2%}")
print(f"  Exaggeration Ratio: {retro_underpowered['exaggeration_ratio'].iloc[0]:.2f}x")
print(f"  Relative Bias: {retro_underpowered['relative_bias'].iloc[0]:.2f}x")
diff = retro_underpowered['exaggeration_ratio'].iloc[0] - retro_underpowered['relative_bias'].iloc[0]
print(f"  Difference: {diff:.2f}x")

print("\nWell-Powered Study (n=3000):")
print(f"  Power: {retro_powered['power'].iloc[0]:.1%}")
print(f"  Type S Error: {retro_powered['type_s_error'].iloc[0]:.2%}")
print(f"  Exaggeration Ratio: {retro_powered['exaggeration_ratio'].iloc[0]:.2f}x")
print(f"  Relative Bias: {retro_powered['relative_bias'].iloc[0]:.2f}x")
print(f"  Difference: {retro_powered['exaggeration_ratio'].iloc[0] - retro_powered['relative_bias'].iloc[0]:.2f}x")

# %% Example 3: Demonstrating with different power levels
print("\n3. How Metrics Change with Power")

power_levels = []
sample_sizes = [200, 400, 800, 1600, 3200]

power_sim = PowerSim(metric="proportion", variants=1, nsim=5000)

for n in sample_sizes:
    retro = power_sim.simulate_retrodesign(true_effect=0.02, sample_size=n, baseline=0.10)
    power_levels.append(
        {
            "sample_size": n,
            "power": retro["power"].iloc[0],
            "type_s_error": retro["type_s_error"].iloc[0],
            "exaggeration_ratio": retro["exaggeration_ratio"].iloc[0],
            "relative_bias": retro["relative_bias"].iloc[0],
            "difference": retro["exaggeration_ratio"].iloc[0] - retro["relative_bias"].iloc[0],
        }
    )

df_power = pd.DataFrame(power_levels)

print("\nMetrics by Sample Size:")
print(df_power.to_string(index=False))

print("\n\nKey Insights:")
print("  1. As power increases, both exaggeration_ratio and relative_bias decrease")
print("  2. The difference between them shrinks as power increases")
print("  3. With high power, Type S errors are rare, so both metrics converge")
print("  4. Use relative_bias when you care about directional bias")
print("  5. Use exaggeration_ratio when you care about magnitude regardless of sign")

# %% Example 4: Practical decision making
print("\n4. Practical Guidance")

print("\nWhen to use each metric:")
print("  Type M Error (Exaggeration Ratio):")
print("    - Conservative assessment of overestimation")
print("    - Worst-case scenario for magnitude")
print("    - Standard metric in retrodesign literature")
print("    - Good for understanding maximum overestimation")
print("\n  Relative Bias:")
print("    - More realistic expected bias")
print("    - Accounts for bidirectional uncertainty")
print("    - Better for meta-analysis corrections")
print("    - Reflects what happens on average with signed values")

print("\nRule of thumb:")
print("  - If Type M Error > 2: Study is severely underpowered")
print("  - If Relative Bias > 1.5: Expected bias is substantial")
print("  - If difference is large: High Type S error risk")
print("  - If both are close to 1: Study is well-powered")
