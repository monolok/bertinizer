# Bertinizer

Bertinizer is a utility designed for fast Exploratory Data Analysis (EDA) and creating plots with ease. It leverages popular Python libraries such as pandas, plotly, and scikit-learn to automate the generation of plots for both numerical and categorical data. Additionally, Bertinizer introduces advanced features such as automatic One-Hot Encoding for categorical variables, Label Encoding for categorical target variables, and outlier detection. These functionalities make Bertinizer an even more powerful tool for data scientists looking to explore and understand their data.

## Installation

You can install Bertinizer directly from GitHub using pip:

```bash
pip install git+https://github.com/monolok/bertinizer.git

## Importing
from bertinizer import plot_dataset, find_outliers

## Using
import pandas as pd
df = pd.read_csv('your_dataset.csv')
plot_dataset(df, y=None, columns='all', corr=0.5)
find_outliers(df, columns='all', std=3)

## Explain
print(plot_dataset.__doc__)
print(find_outliers.__doc__)

## Upgrade
pip install git+https://github.com/monolok/bertinizer.git --upgrade
