import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Create argument parser
parser = argparse.ArgumentParser(description='Analyze results')
parser.add_argument('--input', type=str, help='Path to the input CSV file')
parser.add_argument('--output', type=str, help='Path to save the output plot')
parser.add_argument('--rows', type=int, default=None, help='Number of rows to include from the CSV')
parser.add_argument('--ess', type=bool, default=False, help='Whether to include ESS_ratio in the plot')
# Parse command line arguments
args = parser.parse_args()

# Read the CSV file
data = pd.read_csv(args.input)
if args.rows is not None and len(data) >= args.rows:
    data = data.head(args.rows)

data.fillna(0, inplace=True)
# Extract necessary columns
experimental_val = data['experimental_value']
experimental_uncertainty = data['experimental_uncertainty']

methods = ['deltaF_EXP_accepted', 'deltaF_BAR_accepted', 'deltaF_EXP_all' ,'deltaF_BAR_all']
method_errors = ['deltaF_EXP_std_accepted', 'deltaF_BAR_std_accepted', 'deltaF_EXP_std_all', 'deltaF_BAR_std_all']  # delta_f_GFN may not have std dev

# Function to plot scatter plot with linear regression and statistics
def plot_scatter(ax, x, y, xerr, yerr, title):
    # Exclude outliers
    mask = (x - y).abs() < 25 & yerr.notnull()
    x = x[mask]
    y = y[mask]
    #xerr = xerr[mask].fillna(0)# if xerr is not None else None
    yerr = yerr[mask].fillna(0)# if yerr is not None else None

    sns.scatterplot(x=x, y=y, ax=ax, hue=np.abs(x - y), palette='coolwarm', legend=False)

    # Linear regression
    model = LinearRegression().fit(x.values.reshape(-1, 1), y)
    y_pred = model.predict(x.values.reshape(-1, 1))
    ax.errorbar(x, y, xerr=np.zeros_like(yerr), yerr=yerr, fmt='o', ecolor='gray', alpha=0.5)
    ax.plot(x, y_pred, color='gray', lw=2)

    # Statistics
    aue = mean_absolute_error(x, y)
    cor = r2_score(y, y_pred)
    within_1_kcal = np.sum(np.abs(x - y) < 1) / len(x) * 100

    # Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    r2_str = r'$R^2$'
    ax.set_title(f'{title}\nMAE = {aue:.2f} kcal/mol, {r2_str} = {cor:.2f}, Pearson = {pearson_corr:.2f}, Spearman = {spearman_corr:.2f}\nWithin 1 kcal/mol: {within_1_kcal:.0f}%, N = {len(x)}')
    ax.set_xlabel('ΔG_exp (kcal/mol) ± Uncertainty')
    ax.set_ylabel('ΔG_calc (kcal/mol) ± Uncertainty')

# Function to plot error vs ESS_ratio
def plot_error_vs_ess_single(ax, ess, error, title):
    sns.scatterplot(x=ess, y=error, ax=ax)
    ax.set_xlabel('ESS_ratio')
    ax.set_ylabel('Error (ΔG_calc - ΔG_exp)')
    ax.set_title(title)
    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(ess, error)
    spearman_corr, _ = spearmanr(ess, error)
    ax.annotate(f'Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top')

# Create subplots
rows = 3 if args.ess else 1
fig, axes = plt.subplots(rows, len(methods), figsize=(6 * len(methods), 6 * rows))

for i, method in enumerate(methods):
    y = data[method].astype(float)
    yerr_column = method_errors[i]
    yerr = data[yerr_column].astype(float) if yerr_column else None

    if args.ess:
        plot_scatter(axes[0, i], experimental_val, y, experimental_uncertainty, yerr, title=method)

        # Compute error
        data[f'error_{method}'] = y - experimental_val

        # Second row: error vs ESS_ratio_v
        ess_v = data['ESS_ratio_v']
        error = data[f'error_{method}']
        plot_error_vs_ess_single(axes[1, i], ess_v, error, title=f'{method} - ESS_ratio_v')

        # Third row: error vs ESS_ratio_s
        ess_s = data['ESS_ratio_s']
        plot_error_vs_ess_single(axes[2, i], ess_s, error, title=f'{method} - ESS_ratio_s')
    else:
        plot_scatter(axes[i], experimental_val, y, experimental_uncertainty, yerr, title=method)
# Adjust layout and save plot
plt.tight_layout()
plt.savefig(args.output)

# # Calculate absolute differences and sort
# for method in methods:
#     data[f'abs_diff_{method}'] = np.abs(data['experimental_value'] - data[method])

# # Sort by absolute difference for delta_f_GFN
# data_sorted = data.sort_values(f'abs_diff_delta_f_GFN', ascending=False)

# # Select top 5 molecules with highest error
# top_5_molecules = data_sorted.head(5)
# print("Top 5 molecules with highest error (delta_f_GFN):")
# print(top_5_molecules[['SMILES', 'abs_diff_delta_f_GFN']])

# # Save ranking to CSV
# top_5_molecules[['SMILES', 'abs_diff_delta_f_GFN']].to_csv('ranking.csv', index=False)