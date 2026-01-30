# test IV assumptions

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

# %%
np.random.seed(42)
n = 50000  # total users
user_ids = np.arange(1, n + 1)
Z = np.random.binomial(1, 0.5, size=n)
U = np.random.normal(loc=0, scale=1.0, size=n)
D = np.random.binomial(1, 1 / (1 + np.exp(-(-1 + 0.5 * U + 2.5 * Z))))

# outcome
baseline = 0.0
effect_U = 1.0  # effect of being mobile on Y
effect_D = 2.0  # effect of actually receiving the message
noise = np.random.normal(loc=0, scale=1.0, size=n)

Y = baseline + effect_U * U + effect_D * D + noise

# Put it all in a DataFrame for convenience
df = pd.DataFrame({"user_id": user_ids, "Z": Z, "U": U, "D": D, "Y": Y})

print(df.head(15))
# %%
df[["Z", "D"]].corr()

# %%
# # Using IV2SLS with proper syntax
iv_formula = "Y ~ 1 + [D ~ Z]"
iv_model = IV2SLS.from_formula(iv_formula, data=df)
iv_results = iv_model.fit()

print("\\nCorrect IV Estimate using linearmodels:")
print(iv_results.summary)

# %%
ols = smf.ols("Y ~ D + U", data=df).fit()
print(ols.summary())
