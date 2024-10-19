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
# read database.txt as csv for ground truth exp values


# skip heade
exp_df = pd.read_csv('database.txt', sep=';', skiprows=2)

mcmc_df = pd.read_csv('fed_results/mcmc_results_10000_cobaya_annealed_high_temp1k.csv')
# strip column names
exp_df.columns = exp_df.columns.str.strip()
data.columns = data.columns.str.strip()
mcmc_df.columns = mcmc_df.columns.str.strip()
# make sure the SMILES are in the same order
# apply strip to smiles
exp_df['SMILES'] = exp_df['SMILES'].str.strip()
data['SMILES'] = data['SMILES'].str.strip()
mcmc_df['SMILES'] = mcmc_df['SMILES'].str.strip()

data = data.merge(exp_df, on='SMILES')
data = data.merge(mcmc_df, on='SMILES')
experimental_val = data['experimental value (kcal/mol)']
experimental_uncertainty = data['experimental uncertainty (kcal/mol)']

reference_val = data['Mobley group calculated value (GAFF) (kcal/mol)']
reference_uncertainty = data['calculated uncertainty (kcal/mol)']

methods = ['deltaF_EXP', 'deltaF_BAR'] 

# methods = [method + '_accepted' for method in methods]#  + [method + '_all' for method in methods]

method_errors = [method+'_std' for method in methods]  # delta_f_GFN may not have std dev

methods += ['delta_f_GFN', 'delta_Flb_GFN']
method_errors += ['delta_f_GFN_std', 'delta_Flb_GFN_std']
# methods += ['Mobley group calculated value (GAFF) (kcal/mol)']
# method_errors += ['calculated uncertainty (kcal/mol)']
# Function to plot scatter plot with linear regression and statistics

data['delta_f_GFN_std'] = 0
data['delta_Flb_GFN_std'] = 0

col_to_human_readable = {
    'deltaF_EXP': 'Exponential Averaging',
    'deltaF_BAR': 'BAR',
    'delta_f_GFN': 'Via GFN Z estimate',
    'delta_Flb_GFN': 'Via GFN Z lower bound estimate',
}

def plot_scatter(ax, x, y, xerr, yerr, title):
    x_old = x
    # Exclude outliers
    if type(yerr) == pd.Series:
        mask = yerr.notnull() & (yerr <= 0.6) & (y.abs() < 12)

        x_mask = xerr.notnull() & (xerr <= 0.6) & (x.abs() < 12)

        mask = mask & x_mask
        xerr = xerr[mask].fillna(0)# if xerr is not None else None
        yerr = yerr[mask].fillna(0)# if yerr is not None else None
    else:
        print(type(yerr))
    y = y[mask]
    x = x[mask]
    #xerr = xerr[mask].fillna(0)# if xerr is not None else None

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
    ax.set_title(f'{col_to_human_readable[title]}\nMAE = {aue:.2f} kcal/mol, {r2_str} = {cor:.2f}, Pearson = {pearson_corr:.2f}, Spearman = {spearman_corr:.2f}\nWithin 1 kcal/mol: {within_1_kcal:.0f}%, N = {len(x)}, Failed = {len(x_old) - np.sum(mask)}')
    ax.set_xlabel('ΔG_MCMC BAR (kcal/mol) ± Uncertainty')
    ax.set_ylabel('ΔG_calc (kcal/mol) ± Uncertainty')

# Function to plot error vs ESS_ratio
def plot_error_vs_ess_single(ax, ess, error, title):
    sns.scatterplot(x=ess, y=error, ax=ax)
    ax.set_xlabel('ESS_ratio')
    ax.set_ylabel('Error (ΔG_calc - ΔG_exp)')
    ax.set_title(col_to_human_readable[title])
    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(ess, error)
    spearman_corr, _ = spearmanr(ess, error)
    ax.annotate(f'Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top')

# Create subplots
plt.subplots_adjust(top=0.9)
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

rows = 2
fig, axes = plt.subplots(rows, len(methods)//2, figsize=(8 * len(methods)//2, 8 * rows))
# add spacing between the 4 plots
plt.subplots_adjust(wspace=0.7)
plt.subplots_adjust(hspace=0.7)

# fig.suptitle('Free energy differences (kcal/mol) calculated vs. experimental')
for i, method in enumerate(methods):
    print(f'Plotting method {method}...')
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
        # make the axes indices work with two rows
        plot_scatter(axes[i%2][i//2], experimental_val, y, experimental_uncertainty, yerr, title=method)
# Adjust layout and save plot
# increase font size 


plt.tight_layout()
plt.savefig(args.output, bbox_inches='tight')

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