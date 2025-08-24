import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Define A/B test variations
subject_lines = ['Subject A', 'Subject B', 'Subject C']
preheaders = ['Preheader X', 'Preheader Y']
# Generate synthetic data
num_emails = 1000
data = {
    'Subject': np.random.choice(subject_lines, size=num_emails),
    'Preheader': np.random.choice(preheaders, size=num_emails),
    'Clicks': np.random.binomial(1, 0.1, size=num_emails) # 10% click-through rate on average
}
df = pd.DataFrame(data)
# --- 2. Data Analysis ---
# Create a contingency table to analyze the relationship between subject lines, preheaders, and clicks
contingency_table = pd.crosstab([df['Subject'], df['Preheader']], df['Clicks'])
# Perform Chi-squared test for independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("Contingency Table:")
print(contingency_table)
print(f"\nChi-squared test results:\nChi2: {chi2:.2f}\nP-value: {p:.3f}\nDegrees of freedom: {dof}")
# Calculate click-through rates for each combination
click_rates = contingency_table.apply(lambda x: x[1] / x.sum(), axis=1)
print("\nClick-through rates for each combination:")
print(click_rates)
# Find the combination with the highest click-through rate
best_combination = click_rates.idxmax()
print(f"\nThe best performing combination is: {best_combination} with a click-through rate of {click_rates.max():.2%}")
# --- 3. Visualization ---
plt.figure(figsize=(12, 6))
sns.barplot(x=click_rates.index.get_level_values('Subject'), y=click_rates.values, hue=click_rates.index.get_level_values('Preheader'))
plt.title('Click-Through Rates by Subject Line and Preheader')
plt.xlabel('Subject Line')
plt.ylabel('Click-Through Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save the plot to a file
output_filename = 'click_through_rates.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Clicks by Subject and Preheader")
plt.xlabel("Clicks (0=No, 1=Yes)")
plt.ylabel("Subject/Preheader Combination")
plt.tight_layout()
output_filename2 = "heatmap.png"
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")