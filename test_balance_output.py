"""Test balance output consistency"""
import numpy as np
import pandas as pd
from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# Create sample data
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'treatment': np.random.choice([0, 1], n),
    'outcome': np.random.randn(n),
    'age': np.random.normal(35, 10, n),
    'income': np.random.normal(50000, 15000, n),
})

print("Test 1: Balance log during get_effects()")
print("-" * 60)
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col='treatment',
    outcomes=['outcome'],
    covariates=['age', 'income'],
)
analyzer.get_effects()
print()

print("Test 2: Balance log during check_balance()")
print("-" * 60)
analyzer2 = ExperimentAnalyzer(
    data=df,
    treatment_col='treatment',
    outcomes=['outcome'],
    covariates=['age', 'income'],
)
balance = analyzer2.check_balance()
