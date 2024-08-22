import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse

import matplotlib.pyplot as plt

# Function to parse values with uncertainties
def parse_value_with_uncertainty(value):
  mean, uncertainty = value.split(' ± ')
  return float(mean), float(uncertainty)

# Create argument parser
parser = argparse.ArgumentParser(description='Analyze results')
parser.add_argument('--input', type=str, help='Path to the input CSV file')
parser.add_argument('--output', type=str, help='Path to save the output plot')
parser.add_argument('--rows', type=int, default=None, help='Number of rows to include from the CSV')

# Parse command line arguments
args = parser.parse_args()

# Read the CSV file
data = pd.read_csv(args.input)
if len(data) >= args.rows:
  data = pd.read_csv(args.input, nrows=args.rows)
else:
  raise ValueError('Number of rows to include is greater than the number of rows in the CSV file')
# Parse the values and uncertainties
data[['experimental_val_mean', 'experimental_val_uncertainty']] = data['experimental_val'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))
data[['fed_Z_mean', 'fed_Z_uncertainty']] = data['fed_Z'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))
data[['fed_Z_lb_mean', 'fed_Z_lb_uncertainty']] = data['fed_Z_lb'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))
data[['fed_Z_learned_mean', 'fed_Z_learned_uncertainty']] = data['fed_Z_learned'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))

# Exclude rows with extremely large fedZ or fedZlb values
# data = data[(data['fed_Z_mean'].abs() < 100) & (data['fed_Z_lb_mean'].abs() < 100)]

# excludes rows with extremeleylarge fed_Z_uncertainty
data = data[data['fed_Z_uncertainty'].abs() < 5]

# Sort the data based on the absolute difference between experimental and calculated values
data['abs_diff'] = np.abs(data['experimental_val_mean'] - data['fed_Z_mean'])
data = data.sort_values('abs_diff')

# Extract the necessary columns
experimental_val = data['experimental_val_mean']
experimental_uncertainty = data['experimental_val_uncertainty']
fed_Z = data['fed_Z_mean']
fed_Z_uncertainty = data['fed_Z_uncertainty']
fed_Z_lb = data['fed_Z_lb_mean']
fed_Z_lb_uncertainty = data['fed_Z_lb_uncertainty']
fed_Z_learned = data['fed_Z_learned_mean']
fed_Z_learned_uncertainty = data['fed_Z_learned_uncertainty']


# Calculate the number of points to include based on the top 95%
num_points = int(len(data) * 1)

# Select the top 95% closest points
data = data[:num_points]

# Update the variables with the selected data
experimental_val = data['experimental_val_mean']
experimental_uncertainty = data['experimental_val_uncertainty']
fed_Z = data['fed_Z_mean']
fed_Z_uncertainty = data['fed_Z_uncertainty']
fed_Z_lb = data['fed_Z_lb_mean']
fed_Z_lb_uncertainty = data['fed_Z_lb_uncertainty']
fed_Z_learned = data['fed_Z_learned_mean']
fed_Z_learned_uncertainty = data['fed_Z_learned_uncertainty']

# Function to plot scatter plot with linear regression and statistics
def plot_scatter(ax, x, y, xerr, yerr, title):
  sns.scatterplot(x=x, y=y, ax=ax, hue=np.abs(x-y), palette='coolwarm', legend=False)
  ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', ecolor='gray', alpha=0.5)
  
  # Linear regression
  model = LinearRegression().fit(x.values.reshape(-1, 1), y)
  y_pred = model.predict(x.values.reshape(-1, 1))
  ax.plot(x, y_pred, color='gray', lw=2)
  
  # Statistics
  aue = mean_absolute_error(x, y)
  cor = r2_score(y, y_pred)
  within_1_kcal = np.sum(np.abs(x - y) < 1) / len(x) * 100
  
  # Pearson and Spearman correlations
  pearson_corr, _ = pearsonr(x, y)
  spearman_corr, _ = spearmanr(x, y)
  r2_str = r'$R^2$'
  ax.set_title(f'{title}\nMAE = {aue:.2f} kcal/mol, {r2_str} = {cor:.2f}, Pearson = {pearson_corr:.2f}, Spearman = {spearman_corr:.2f}\n1 kcal/mol = {within_1_kcal:.0f}%, N = {len(x)}')
  ax.set_xlabel('ΔG_exp, kcal/mol ± Uncertainty')
  ax.set_ylabel('ΔG_calc, kcal/mol ± Uncertainty')

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each method
plot_scatter(axes[0], experimental_val, fed_Z, experimental_uncertainty, fed_Z_uncertainty, 'fed_Z')
plot_scatter(axes[1], experimental_val, fed_Z_lb, experimental_uncertainty, fed_Z_lb_uncertainty, 'fed_Z_lb')
plot_scatter(axes[2], experimental_val, fed_Z_learned, experimental_uncertainty, fed_Z_learned_uncertainty, 'fed_Z_learned')

# Adjust layout and show plot
plt.tight_layout()
# plt.show()
plt.savefig(args.output)

# Sort the data based on mean absolute error (MAE) in descending order
data = data.sort_values('abs_diff', ascending=False)

# Select the top 5 rows with the highest MAE
top_5_molecules = data.head(5)

# Print out the SMILES information for the top 5 molecules
df_new = data[['SMILES', 'abs_diff']]
df_new.to_csv('ranking.csv', index=False)