#!/usr/bin/env python3
"""
Test script to verify balanced random assignment with multiple variants
"""

# %%
import numpy as np
import pandas as pd

from experiment_utils.utils import balanced_random_assignment

# %% Create sample data
np.random.seed(123)
n = 1000

df = pd.DataFrame({"user_id": range(n), "segment": np.random.choice(["A", "B", "C"], n)})

# %%
print("=" * 60)
print("TEST 1: Binary assignment (test/control with 50/50)")
print("=" * 60)
df["assignment_1"] = balanced_random_assignment(df, seed=42, allocation_ratio=0.5)
print(df["assignment_1"].value_counts())
print(f"Proportions:\n{df['assignment_1'].value_counts(normalize=True)}\n")

# %%
print("=" * 60)
print("TEST 2: Binary assignment (test/control with 70/30)")
print("=" * 60)
df["assignment_2"] = balanced_random_assignment(df, seed=42, allocation_ratio=0.7)
print(df["assignment_2"].value_counts())
print(f"Proportions:\n{df['assignment_2'].value_counts(normalize=True)}\n")

# %%
print("=" * 60)
print("TEST 3: Three variants with equal allocation")
print("=" * 60)
df["assignment_3"] = balanced_random_assignment(df, seed=42, variants=["control", "variant_a", "variant_b"])
print(df["assignment_3"].value_counts())
print(f"Proportions:\n{df['assignment_3'].value_counts(normalize=True)}\n")

# %%
print("=" * 60)
print("TEST 4: Three variants with custom allocation (50/30/20)")
print("=" * 60)
df["assignment_4"] = balanced_random_assignment(
    df,
    seed=42,
    variants=["control", "variant_a", "variant_b"],
    allocation_ratio={"control": 0.5, "variant_a": 0.3, "variant_b": 0.2},
)
print(df["assignment_4"].value_counts())
print(f"Proportions:\n{df['assignment_4'].value_counts(normalize=True)}\n")

# %%
print("=" * 60)
print("TEST 5: Four variants with equal allocation")
print("=" * 60)
df["assignment_5"] = balanced_random_assignment(
    df, seed=42, variants=["control", "variant_a", "variant_b", "variant_c"]
)
print(df["assignment_5"].value_counts())
print(f"Proportions:\n{df['assignment_5'].value_counts(normalize=True)}\n")

# %%
print("=" * 60)
print("TEST 6: Block randomization (by segment)")
print("=" * 60)
df["assignment_6"] = balanced_random_assignment(
    df,
    seed=42,
    variants=["control", "variant_a", "variant_b"],
    allocation_ratio={"control": 0.4, "variant_a": 0.4, "variant_b": 0.2},
    balance_covariates=["segment"],
)
print("Overall distribution:")
print(df["assignment_6"].value_counts())
print(f"\nProportions:\n{df['assignment_6'].value_counts(normalize=True)}")
print("\nDistribution by segment:")
print(pd.crosstab(df["segment"], df["assignment_6"], normalize="index"))

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)

# %%
print("\n" + "=" * 60)
print("TEST 7: Balance checking with categorical covariate")
print("=" * 60)

# Add categorical covariates for stratification
np.random.seed(42)
df["age_group"] = np.random.choice(["18-30", "31-45", "46-60", "60+"], n)
df["region"] = np.random.choice(["North", "South", "East", "West"], n)

# Assignment with balance checking using categorical stratification
df["assignment_7"] = balanced_random_assignment(
    df, seed=42, allocation_ratio=0.5, balance_covariates=["age_group"], check_balance=True
)

# %%
print("\n" + "=" * 60)
print("TEST 8: Balance checking with multiple categorical covariates")
print("=" * 60)

# Assignment with multiple categorical covariates
df["assignment_8"] = balanced_random_assignment(
    df, seed=42, allocation_ratio=0.5, balance_covariates=["age_group", "region"], check_balance=True
)

# %%
print("\n" + "=" * 60)
print("TEST 9: Balance checking with custom comparisons (3 variants)")
print("=" * 60)

# Multiple variants with custom comparisons
df["assignment_9"] = balanced_random_assignment(
    df,
    seed=42,
    variants=["control", "variant_a", "variant_b"],
    balance_covariates=["age_group"],
    comparison=[("variant_a", "control"), ("variant_b", "control")],
    check_balance=True,
)

# %%
print("\n" + "=" * 60)
print("TEST 10: Disable balance checking")
print("=" * 60)

# Assignment with balance checking disabled
df["assignment_10"] = balanced_random_assignment(
    df, seed=42, allocation_ratio=0.5, balance_covariates=["region"], check_balance=False
)
print("No balance output (check_balance=False)")

# %%
print("\n" + "=" * 60)
print("TEST 11: Balance checking with custom SMD threshold")
print("=" * 60)

# Assignment with custom SMD threshold
df["assignment_11"] = balanced_random_assignment(
    df,
    seed=42,
    allocation_ratio=0.5,
    balance_covariates=["age_group", "region"],
    check_balance=True,
    smd_threshold=0.05,  # Stricter threshold
)

print("\n" + "=" * 60)
print("All balance checking tests completed successfully!")
print("=" * 60)
