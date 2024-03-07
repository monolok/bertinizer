# Bertinizer

Bertinizer is a utility designed for fast Exploratory Data Analysis (EDA).
You can install Bertinizer directly from GitHub using pip.

## Play with Bertinizer

```bash
## Installing
pip install git+https://github.com/monolok/bertinizer.git

## Upgrading
pip install git+https://github.com/monolok/bertinizer.git --upgrade

## Importing
from bertinizer import plot_data, analyze_correlation, find_outliers, dataset_overview_simplified, find_pca_num_only, explode_datetime_col, plot_optimal_k_kmeans

## Using
import pandas as pd
df = pd.read_csv('your_dataset.csv')

plot_data(df, y=None, columns='all')
analyze_correlation(df, y=None, columns='all', corr=0.5)
find_outliers(df, columns='all', std=3)
dataset_overview_simplified(df, remove_nan=False)
find_pca_num_only(df, columns='all', pca_min=0.5) #Sparse and Dense matrix
explode_datetime_col(df, col_name)
plot_optimal_k_kmeans(X, max_k=10)

## Explaining
print(plot_data.__doc__)