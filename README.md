# Bertinizer

Bertinizer is a utility designed for fast Exploratory Data Analysis (EDA) and creating plots with ease. It leverages popular Python libraries such as pandas, plotly, and scikit-learn to automate the generation of plots for both numerical and categorical data. Additionally, if a target variable is provided, Bertinizer can generate a correlation matrix plot to explore potential relationships within the data.

## Installation

You can install Bertinizer directly from GitHub using pip:

```bash
pip install git+https://github.com/yourusername/bertinizer.git

## Importing
from bertinizer.files import plot_dataset

## Using
import pandas as pd
df = pd.read_csv('your_dataset.csv')
plot_dataset(df, 'your_target_column')

## Explain
print(plot_dataset.__doc__)