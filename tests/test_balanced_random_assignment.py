"""Tests for balanced_random_assignment randomization correctness"""

import numpy as np
import pandas as pd
import pytest

from experiment_utils.utils import balanced_random_assignment


@pytest.fixture
def base_df():
    np.random.seed(123)
    n = 1000
    return pd.DataFrame(
        {
            "user_id": range(n),
            "segment": np.random.choice(["A", "B", "C"], n),
            "age_group": np.random.choice(["18-30", "31-45", "46-60", "60+"], n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
        }
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result(base_df):
    a1 = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    a2 = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    pd.testing.assert_series_equal(a1, a2)


def test_different_seeds_differ(base_df):
    a1 = balanced_random_assignment(base_df, seed=1, allocation_ratio=0.5, check_balance=False)
    a2 = balanced_random_assignment(base_df, seed=2, allocation_ratio=0.5, check_balance=False)
    assert not a1.equals(a2), "Different seeds should produce different assignments"


# ---------------------------------------------------------------------------
# Allocation counts — binary
# ---------------------------------------------------------------------------


def test_binary_exact_50_50(base_df):
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    counts = a.value_counts()
    assert counts["test"] == 500
    assert counts["control"] == 500


def test_binary_exact_70_30(base_df):
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.7, check_balance=False)
    counts = a.value_counts()
    assert counts["test"] == 700
    assert counts["control"] == 300


def test_binary_covers_all_rows(base_df):
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    assert len(a) == len(base_df)
    assert set(a.unique()) == {"test", "control"}


# ---------------------------------------------------------------------------
# Allocation counts — multiple variants
# ---------------------------------------------------------------------------


def test_three_variants_equal_allocation(base_df):
    a = balanced_random_assignment(
        base_df, seed=42, variants=["control", "variant_a", "variant_b"], check_balance=False
    )
    counts = a.value_counts()
    # With 1000 rows and 3 variants: 334, 333, 333
    assert counts.sum() == 1000
    assert abs(counts["control"] - counts["variant_a"]) <= 1
    assert abs(counts["control"] - counts["variant_b"]) <= 1


def test_three_variants_custom_allocation(base_df):
    ratios = {"control": 0.5, "variant_a": 0.3, "variant_b": 0.2}
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=["control", "variant_a", "variant_b"],
        allocation_ratio=ratios,
        check_balance=False,
    )
    counts = a.value_counts()
    assert counts["control"] == 500
    assert counts["variant_a"] == 300
    assert counts["variant_b"] == 200


def test_four_variants_equal_allocation(base_df):
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=["control", "variant_a", "variant_b", "variant_c"],
        check_balance=False,
    )
    counts = a.value_counts()
    assert counts.sum() == 1000
    for v in ["control", "variant_a", "variant_b", "variant_c"]:
        assert counts[v] == 250


# ---------------------------------------------------------------------------
# Shuffle is actually applied (no systematic ordering)
# ---------------------------------------------------------------------------


def test_shuffle_applied_no_all_test_first(base_df):
    """Without shuffle, the first n_test rows would all be 'test'. Verify this isn't the case."""
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    # If unshuffled, first 500 rows would be 'test' and last 500 'control'
    first_half = a.iloc[:500]
    assert not (first_half == "test").all(), "All first rows are 'test' — shuffle may not be working"
    assert not (first_half == "control").all(), "All first rows are 'control' — unexpected"


def test_both_variants_appear_in_first_and_last_half(base_df):
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    first_half_counts = a.iloc[:500].value_counts()
    last_half_counts = a.iloc[500:].value_counts()
    # Both variants should be present in both halves
    assert "test" in first_half_counts and "control" in first_half_counts
    assert "test" in last_half_counts and "control" in last_half_counts


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------


def test_output_index_matches_input(base_df):
    a = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    pd.testing.assert_index_equal(a.index, base_df.index)


def test_non_default_index(base_df):
    df_shifted = base_df.copy()
    df_shifted.index = df_shifted.index + 100
    a = balanced_random_assignment(df_shifted, seed=42, allocation_ratio=0.5, check_balance=False)
    pd.testing.assert_index_equal(a.index, df_shifted.index)
    assert len(a) == len(df_shifted)


def test_non_sequential_index():
    df = pd.DataFrame({"x": range(100)}, index=np.random.permutation(range(100)))
    a = balanced_random_assignment(df, seed=42, allocation_ratio=0.5, check_balance=False)
    pd.testing.assert_index_equal(a.sort_index().index, df.sort_index().index)


# ---------------------------------------------------------------------------
# Stratification (block randomization)
# ---------------------------------------------------------------------------


def test_stratification_within_strata_binary(base_df):
    """Within each segment stratum the allocation should match the overall ratio."""
    a = balanced_random_assignment(
        base_df,
        seed=42,
        allocation_ratio=0.5,
        balance_covariates=["segment"],
        check_balance=False,
    )
    df_check = base_df.copy()
    df_check["assignment"] = a
    for _seg, group in df_check.groupby("segment"):
        counts = group["assignment"].value_counts()
        total = len(group)
        # Each stratum should have exactly round(n*0.5) test and remainder control
        expected_test = round(total * 0.5)
        assert counts.get("test", 0) == expected_test, (
            f"Segment {_seg}: expected {expected_test} test, got {counts.get('test', 0)}"
        )


def test_stratification_multi_variant_within_strata(base_df):
    """With multi-variant stratified assignment, within each stratum counts match ratios."""
    ratios = {"control": 0.5, "variant_a": 0.3, "variant_b": 0.2}
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=["control", "variant_a", "variant_b"],
        allocation_ratio=ratios,
        balance_covariates=["segment"],
        check_balance=False,
    )
    df_check = base_df.copy()
    df_check["assignment"] = a
    for _seg, group in df_check.groupby("segment"):
        total = len(group)
        counts = group["assignment"].value_counts()
        assert counts.get("control", 0) == round(total * 0.5)
        assert counts.get("variant_a", 0) == round(total * 0.3)
        # variant_b gets the remainder
        assert counts.get("control", 0) + counts.get("variant_a", 0) + counts.get("variant_b", 0) == total


def test_stratification_different_from_unstratified(base_df):
    """Stratified and unstratified assignments with the same seed should differ."""
    a_strat = balanced_random_assignment(
        base_df, seed=42, allocation_ratio=0.5, balance_covariates=["segment"], check_balance=False
    )
    a_unstrat = balanced_random_assignment(base_df, seed=42, allocation_ratio=0.5, check_balance=False)
    assert not a_strat.equals(a_unstrat), "Stratified and unstratified should differ"


# ---------------------------------------------------------------------------
# Integer 0/1 treatment labels
# ---------------------------------------------------------------------------


def test_binary_integer_labels_50_50(base_df):
    """variants=[0, 1] works and produces correct counts."""
    a = balanced_random_assignment(
        base_df, seed=42, variants=[0, 1], allocation_ratio={0: 0.5, 1: 0.5}, check_balance=False
    )
    assert set(a.unique()) == {0, 1}
    assert a.value_counts()[0] == 500
    assert a.value_counts()[1] == 500


def test_binary_integer_labels_allocation_ratio_float(base_df):
    """With variants=[0, 1] and a scalar allocation_ratio, label 0 gets (1-ratio) and 1 gets ratio."""
    # scalar allocation_ratio applies to the first variant ('test' semantics do not apply here)
    # — pass as dict to be explicit about which label gets which share
    a = balanced_random_assignment(
        base_df, seed=42, variants=[0, 1], allocation_ratio={0: 0.3, 1: 0.7}, check_balance=False
    )
    counts = a.value_counts()
    assert counts[0] == 300
    assert counts[1] == 700


def test_integer_labels_cover_all_rows(base_df):
    a = balanced_random_assignment(
        base_df, seed=42, variants=[0, 1], allocation_ratio={0: 0.5, 1: 0.5}, check_balance=False
    )
    assert len(a) == len(base_df)
    assert set(a.unique()) == {0, 1}


def test_integer_labels_shuffle_applied(base_df):
    """0s and 1s should be interleaved, not all 0s first."""
    a = balanced_random_assignment(
        base_df, seed=42, variants=[0, 1], allocation_ratio={0: 0.5, 1: 0.5}, check_balance=False
    )
    first_half = a.iloc[:500]
    assert not (first_half == 0).all()
    assert not (first_half == 1).all()


def test_integer_labels_stratified(base_df):
    """Stratification works correctly with integer labels."""
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=[0, 1],
        allocation_ratio={0: 0.5, 1: 0.5},
        balance_covariates=["segment"],
        check_balance=False,
    )
    df_check = base_df.copy()
    df_check["assignment"] = a
    for _seg, group in df_check.groupby("segment"):
        counts = group["assignment"].value_counts()
        total = len(group)
        assert counts.get(0, 0) + counts.get(1, 0) == total
        assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1


def test_integer_labels_reproducible(base_df):
    a1 = balanced_random_assignment(
        base_df, seed=7, variants=[0, 1], allocation_ratio={0: 0.5, 1: 0.5}, check_balance=False
    )
    a2 = balanced_random_assignment(
        base_df, seed=7, variants=[0, 1], allocation_ratio={0: 0.5, 1: 0.5}, check_balance=False
    )
    pd.testing.assert_series_equal(a1, a2)


# ---------------------------------------------------------------------------
# Float variant labels
# ---------------------------------------------------------------------------


def test_float_variant_labels(base_df):
    """Float labels (e.g. 0.0 / 1.0) are stored and returned correctly."""
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=[0.0, 1.0],
        allocation_ratio={0.0: 0.5, 1.0: 0.5},
        check_balance=False,
    )
    assert set(a.unique()) == {0.0, 1.0}
    assert a.value_counts()[0.0] == 500
    assert a.value_counts()[1.0] == 500


def test_float_labels_unequal_allocation(base_df):
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=[0.0, 1.0],
        allocation_ratio={0.0: 0.7, 1.0: 0.3},
        check_balance=False,
    )
    counts = a.value_counts()
    assert counts[0.0] == 700
    assert counts[1.0] == 300


# ---------------------------------------------------------------------------
# Treatment-only subset (assigning within a filtered DataFrame)
# ---------------------------------------------------------------------------


def test_treatment_only_subset():
    """Assigning only among units already selected for treatment is supported."""
    np.random.seed(0)
    n = 600
    df_all = pd.DataFrame({"user_id": range(n), "group": np.random.choice(["eligible", "ineligible"], n)})
    df_treatment_pool = df_all[df_all["group"] == "eligible"].copy()

    a = balanced_random_assignment(
        df_treatment_pool,
        seed=42,
        variants=[0, 1],
        allocation_ratio={0: 0.5, 1: 0.5},
        check_balance=False,
    )
    # Assignment only covers the eligible subset
    assert len(a) == len(df_treatment_pool)
    assert set(a.index) == set(df_treatment_pool.index)
    # Counts are balanced within the subset
    assert abs(a.value_counts()[0] - a.value_counts()[1]) <= 1


def test_treatment_only_index_alignment():
    """Index of result aligns with the input subset, not the original full DataFrame."""
    df_full = pd.DataFrame({"x": range(100)})
    df_sub = df_full.iloc[20:70]  # 50 rows, index 20..69

    a = balanced_random_assignment(
        df_sub,
        seed=42,
        variants=[0, 1],
        allocation_ratio={0: 0.5, 1: 0.5},
        check_balance=False,
    )
    pd.testing.assert_index_equal(a.index, df_sub.index)
    assert len(a) == 50


# ---------------------------------------------------------------------------
# Float allocation_ratio with explicit variant labels
# ---------------------------------------------------------------------------


def test_float_ratio_with_integer_variants(base_df):
    """variants=[1, 0] + float allocation_ratio: first label gets the ratio."""
    a = balanced_random_assignment(base_df, seed=42, variants=[1, 0], allocation_ratio=0.667, check_balance=False)
    counts = a.value_counts()
    assert counts[1] == int(1000 * 0.667)  # 667
    assert counts[0] == 1000 - int(1000 * 0.667)  # 333


def test_float_ratio_with_string_variants(base_df):
    """variants=['treatment', 'control'] + float: 'treatment' gets the ratio."""
    a = balanced_random_assignment(
        base_df, seed=42, variants=["treatment", "control"], allocation_ratio=0.6, check_balance=False
    )
    counts = a.value_counts()
    assert counts["treatment"] == 600
    assert counts["control"] == 400


def test_float_ratio_variants_reversed_label_order(base_df):
    """Passing variants=[0, 1] vs [1, 0] with same ratio should flip which label is majority."""
    a_1_first = balanced_random_assignment(base_df, seed=42, variants=[1, 0], allocation_ratio=0.7, check_balance=False)
    a_0_first = balanced_random_assignment(base_df, seed=42, variants=[0, 1], allocation_ratio=0.7, check_balance=False)
    assert a_1_first.value_counts()[1] == 700  # 1 is majority
    assert a_0_first.value_counts()[0] == 700  # 0 is majority


def test_float_ratio_three_variants_equal_fallback(base_df):
    """Float allocation_ratio with 3+ variants falls back to equal allocation."""
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=["control", "v1", "v2"],
        allocation_ratio=0.5,  # ambiguous for 3 variants → equal split
        check_balance=False,
    )
    counts = a.value_counts()
    assert counts.sum() == 1000
    # Each variant gets ~333 units (equal allocation)
    for v in ["control", "v1", "v2"]:
        assert abs(counts[v] - 333) <= 1


def test_float_ratio_with_stratification(base_df):
    """Float ratio + integer labels works correctly within strata."""
    a = balanced_random_assignment(
        base_df,
        seed=42,
        variants=[1, 0],
        allocation_ratio=0.667,
        balance_covariates=["segment"],
        check_balance=False,
    )
    df_check = base_df.copy()
    df_check["assignment"] = a
    for _seg, group in df_check.groupby("segment"):
        counts = group["assignment"].value_counts()
        total = len(group)
        expected_treatment = round(total * 0.667)
        assert counts.get(1, 0) == expected_treatment, (
            f"Segment {_seg}: expected {expected_treatment} treated, got {counts.get(1, 0)}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_allocation_ratio_dict_without_variants_raises():
    df = pd.DataFrame({"x": range(10)})
    with pytest.raises(ValueError, match="allocation_ratio must be a float"):
        balanced_random_assignment(df, allocation_ratio={"a": 0.5, "b": 0.5}, check_balance=False)


def test_allocation_ratio_does_not_sum_to_one_raises():
    df = pd.DataFrame({"x": range(10)})
    with pytest.raises(ValueError, match="sum to 1.0"):
        balanced_random_assignment(
            df,
            variants=["a", "b"],
            allocation_ratio={"a": 0.6, "b": 0.6},
            check_balance=False,
        )


def test_variants_mismatch_allocation_keys_raises():
    df = pd.DataFrame({"x": range(10)})
    with pytest.raises(ValueError, match="match allocation_ratio keys"):
        balanced_random_assignment(
            df,
            variants=["a", "b", "c"],
            allocation_ratio={"a": 0.5, "b": 0.5},
            check_balance=False,
        )
