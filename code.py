# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load and view data
shopping_data = pd.read_csv("online_shopping_session_data.csv")
shopping_data.head()

# ### Objective: calculate the percentage of customers by customer type, for November–December only

# Filter data for November and December
NovDec_data = shopping_data[shopping_data["Month"].isin(["Nov", "Dec"])]

# Group by CustomerType and calculate purchase rate
purchase_rates = NovDec_data.groupby("CustomerType")["Purchase"].mean().to_dict()

print(f'purchase_rates = {{"Returning_Customer": {purchase_rates["Returning_Customer"]:.3f}, '
      f'"New_Customer": {purchase_rates["New_Customer"]:.3f}}}')

# Convert purchase rates to %
purchase_percent = {k: v * 100 for k, v in purchase_rates.items()}

# Visualise the result
plt.bar(purchase_percent.keys(), purchase_percent.values(), color=["#7E57C2", "#BA68C8"])
plt.title("Purchase rate (%) by Customer type (Nov–Dec)")
plt.ylabel("Purchase rate (%)")
plt.ylim(0, 100)
plt.xticks(
    ticks=range(len(purchase_percent)),
    labels=["New Customer", "Returning Customer"]
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.show()

# ________

# ### Objective: for returning customers, find the 2 duration variables (*_Duration) that have the highest correlation

# Filter returning customers in Nov & Dec
returning = NovDec_data[NovDec_data["CustomerType"] == "Returning_Customer"]

# Select only duration columns
duration_cols = [col for col in returning.columns if "Duration" in col]
duration_data = returning[duration_cols]

# Compute correlation matrix
corr_matrix = duration_data.corr().abs()

# Get top correlation pair
corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
top_pair = [(x, y) for (x, y) in corr_unstacked.index if x != y][0]
top_correlation = {
    "pair": top_pair,
    "correlation": corr_matrix.loc[top_pair[0], top_pair[1]]
}

print(f'top_correlation = {{"pair": {top_correlation["pair"]}, "correlation": {top_correlation["correlation"]:.3f}}}')

# Visualise the result
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="Purples", fmt=".2f", linewidths=0.5)
plt.title("Correlation between time spent on different page types (Returning customers)", fontsize=11, pad=10)
plt.show()

# ______

# **Let p be the purchase rate of previous returning customers. New campaign increases by 15% → p_new = p * 1.15.**
# ### Objective: use binomial distribution to calculate the probability of having at least 100 orders in 500 sessions

# Original purchase rate for returning customers
p_original = purchase_rates['Returning_Customer']
p_new = min(p_original * 1.15, 1)  #cap at 1

# Parameters
n = 500  #sessions
k = 100  #target number of sales

# Probability of at least 100 sales
prob_at_least_100_sales = 1 - stats.binom.cdf(k-1, n, p_new)
prob_at_least_100_sales

# Visualise the result
x = np.arange(0, 150)
plt.bar(x, stats.binom.pmf(x, n, p_new))
plt.title("Binomial distribution: Returning customers campaign")
plt.xlabel("Number of Purchases")
plt.ylabel("Probability")

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.show()
