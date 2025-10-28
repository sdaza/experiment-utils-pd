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
