# Bertinizer

Bertinizer is a utility designed for fast Exploratory Data Analysis (EDA)

## Installation

You can install Bertinizer directly from GitHub using pip:

```bash
pip install git+https://github.com/monolok/bertinizer.git

## Importing
from bertinizer import plot_data, analyze_correlation, find_outliers, dataset_overview

## Using
import pandas as pd
df = pd.read_csv('your_dataset.csv')

plot_data(df, y=None, columns='all')
analyze_correlation(df, y=None, columns='all', corr=0.5)
find_outliers(df, columns='all', std=3)
dataset_overview(df, remove_nan=False)
apply_pca(df, columns='all', pca_min=0.5)

## Explain
print(plot_data.__doc__)
print(analyze_correlation.__doc__)
print(find_outliers.__doc__)
print(dataset_overview.__doc__)
print(apply_pca.__doc__)

## Upgrade
pip install git+https://github.com/monolok/bertinizer.git --upgrade

## TODO
- feature_selection
- cluster for preprocessing