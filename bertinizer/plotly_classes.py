import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

def plot_dataset(df, y=None, columns='all', corr=0.5):
    """
    Automatically generates plots for specified numerical and categorical columns in a DataFrame,
    applies One-Hot Encoding to categorical columns, and Label Encoding to a categorical target variable 'y'.
    Generates a correlation matrix plot for numerical features and the encoded target variable,
    and checks for strongly correlated columns.

    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - y: str, optional, the name of the target column.
    - columns: list or 'all', specifies the columns to include. Defaults to 'all'.
    - corr: float, optional, the correlation threshold for identifying strong correlations. Defaults to 0.5.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Filter columns if 'columns' parameter is not 'all'
    if columns != 'all':
        df = df[[*columns, y]] if y and y in df else df[columns]

    # Separate target variable if provided and categorical
    if y is not None and (df[y].dtype == 'object' or df[y].dtype.name == 'category'):
        y_series = df[y]
        df.drop(columns=[y], inplace=True)  # Temporarily remove y for encoding
    else:
        y_series = None
    
    # Apply One-Hot Encoding to categorical columns
    df = pd.get_dummies(df, drop_first=True)  # One-Hot Encoding
    
    # Re-add encoded target variable if it was removed
    if y_series is not None:
        le = LabelEncoder()
        df[y] = le.fit_transform(y_series)
    
    # Now proceed with plotting for numerical columns (all are numerical after encoding)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, marginal="box", color=y if y else None and y in numerical_cols, title=f"Distribution of {col}")
        fig.show()

    # Calculating correlation matrix for the DataFrame including the target variable
    corr_matrix = df.corr()

    # Generating heatmap for the correlation matrix
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'))

    fig.update_layout(title='Correlation Matrix Heatmap', xaxis_title='Features', yaxis_title='Features')
    fig.show()

    # Identify strongly correlated pairs
    strong_corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    strong_corr_pairs = strong_corr_pairs.stack().reset_index()
    strong_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    strong_corr_pairs = strong_corr_pairs.loc[strong_corr_pairs['Correlation'].abs() > corr]

    if not strong_corr_pairs.empty:
        print("Strongly correlated pairs (above threshold):")
        print(strong_corr_pairs)
    else:
        print("No strongly correlated pairs found above the threshold.")


def find_outliers(df, columns='all', std=3):
    """
    Identifies outliers in the specified columns of a DataFrame based on the standard deviation.
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame.
    - columns: list or 'all', optional, specifies the columns to check for outliers. Defaults to 'all'.
    - std: int or float, the number of standard deviations to use for defining outliers. Defaults to 3.
    
    Returns:
    A dictionary where keys are column names and values are lists of indices of the outliers.
    """
    # Filter columns if 'columns' parameter is not 'all'
    if columns == 'all':
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    else:
        numerical_cols = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    outliers_dict = {}
    
    for col in numerical_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        # Defining outliers as those more than 'std' standard deviations from the mean
        outliers = df[(df[col] < col_mean - std * col_std) | (df[col] > col_mean + std * col_std)].index.tolist()
        
        if outliers:
            outliers_dict[col] = outliers
    
    return outliers_dict