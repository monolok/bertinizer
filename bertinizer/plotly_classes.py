import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

def plot_dataset(df, y=None, columns='all', corr=0.5):
    """
    Automatically generates plots for specified numerical and categorical columns in a DataFrame.
    If a target variable 'y' is provided, it also generates a correlation matrix plot and checks for strongly
    correlated columns based on the specified threshold.
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - y: str, optional, the name of the target column.
    - columns: list or 'all', optional, specifies the columns to include. Defaults to 'all'.
    - corr: float, optional, the correlation threshold for identifying strong correlations. Defaults to 0.5.
    """
    # Filter columns if 'columns' parameter is not 'all'
    if columns != 'all':
        df = df[[*columns, y]] if y and y in df else df[columns]
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Plot for numerical columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, marginal="box", color=y if y else None, title=f"Distribution of {col}")
        fig.show()
        
    # Plot for categorical columns
    for col in categorical_cols:
        fig = px.bar(df, x=col, color=y if y else None, title=f"Distribution of {col}")
        fig.show()

    # If y is provided, generate correlation matrix plot
    if y:
        # Encoding all categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            
    # Calculating correlation matrix
    corr_matrix = df.corr()

    # Generating heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'))
    
    fig.update_layout(title='Correlation Matrix Heatmap', xaxis_title='Features', yaxis_title='Features')
    fig.show()

    # Identify strongly correlated pairs
    strong_corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
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