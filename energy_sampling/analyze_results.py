import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Function to parse values with uncertainties
def parse_value_with_uncertainty(value):
  mean, uncertainty = value.split(' ± ')
  return float(mean), float(uncertainty)

# Create argument parser
parser = argparse.ArgumentParser(description='Analyze results')
parser.add_argument('input', type=str, help='Path to the input CSV file')
parser.add_argument('output', type=str, help='Path to save the output plot')

# Parse command line arguments
args = parser.parse_args()

# Read the CSV file
data = pd.read_csv(args.input)

# Parse the values and uncertainties
data[['experimental_val_mean', 'experimental_val_uncertainty']] = data['experimental_val'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))
data[['fed_Z_mean', 'fed_Z_uncertainty']] = data['fed_Z'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))
data[['fed_Z_lb_mean', 'fed_Z_lb_uncertainty']] = data['fed_Z_lb'].apply(lambda x: pd.Series(parse_value_with_uncertainty(x)))

# Exclude rows with extremely large fedZ or fedZlb values
data = data[(data['fed_Z_mean'].abs() < 100) & (data['fed_Z_lb_mean'].abs() < 100)]

# Extract the necessary columns
experimental_val = data['experimental_val_mean']
experimental_uncertainty = data['experimental_val_uncertainty']
fed_Z = data['fed_Z_mean']
fed_Z_lb = data['fed_Z_lb_mean']

# Function to plot scatter plot with linear regression and statistics
def plot_scatter(ax, x, y, xerr, title):
  sns.scatterplot(x=x, y=y, ax=ax, hue=np.abs(x-y), palette='coolwarm', legend=False)
  ax.errorbar(x, y, xerr=xerr, fmt='o', ecolor='gray', alpha=0.5)
  
  # Linear regression
  model = LinearRegression().fit(x.values.reshape(-1, 1), y)
  y_pred = model.predict(x.values.reshape(-1, 1))
  ax.plot(x, y_pred, color='gray', lw=2)
  
  # Statistics
  aue = mean_absolute_error(x, y)
  cor = r2_score(x, y)
  within_1_kcal = np.sum(np.abs(x - y) < 1) / len(x) * 100
  
  ax.set_title(f'{title}\nAUE = {aue:.2f} kcal/mol, cor = {cor:.2f}\n1 kcal/mol = {within_1_kcal:.0f}%')
  ax.set_xlabel('ΔG_exp, kcal/mol')
  ax.set_ylabel('ΔG_calc, kcal/mol')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot each method
plot_scatter(axes[0], experimental_val, fed_Z, experimental_uncertainty, 'fed_Z')
plot_scatter(axes[1], experimental_val, fed_Z_lb, experimental_uncertainty, 'fed_Z_lb')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
plt.savefig(args.output)
