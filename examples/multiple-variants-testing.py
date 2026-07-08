# %% setup
from experiment_utils import PowerSim

# %% 
ps = PowerSim(
    metric="proportion",
    relative_effect=True,
    variants=3,
    nsim=5000,
    alpha=0.05,
    correction="dunnett",  # Benjamini-Hochberg (linear step-up)
    alternative="two-tailed",
    early_stopping=False,
    comparisons=[(0, 1), (0, 2), (0, 3)]
)

# %%
power_df = ps.get_power(
    baseline=0.10,
    effect=0.05,
    sample_size=75433

)
print(power_df)


# %% 
ps.find_sample_size(
    baseline=0.10,
    effect=0.05,
    power=0.8,
    min_sample_size=10000,
    max_sample_size=500000, 
    # step_size=100,
    tolerance=0.001,
    correction="dunnett",
)


# %%
ps.find_sample_size(
    baseline=0.10,
    effect=0.05,
    power=0.8,
    min_sample_size=10000,
    max_sample_size=500000, 
    # step_size=100,
    tolerance=0.001,
    correction="fdr",
)



# %%
