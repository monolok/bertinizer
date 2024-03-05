import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings

def plot_data(df, y=None, columns='all'):
    """
    Generates plots for specified numerical and categorical columns in a DataFrame,
    showing percentages for categorical data distributions.

    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - y: str, optional, the name of the target column.
    - columns: list or 'all', specifies the columns to include. Defaults to 'all'.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Filter columns if 'columns' parameter is not 'all'
    if columns != 'all':
        df = df[[*columns, y]] if y and y in df else df[columns]

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Plot for numerical columns
    for col in numerical_cols:
        fig = px.histogram(df, x=col, marginal="box", color=y if y else None and col in df.columns, title=f"Distribution of {col}")
        fig.show()
        
    # Plot for categorical columns with percentages
    for col in categorical_cols:
        count_series = df[col].value_counts(normalize=True).reset_index()  # Calculate percentage
        count_series.columns = [col, 'percentage']  # Correct column naming
        count_series['percentage'] *= 100  # Convert to percentage
        fig = px.bar(count_series, x=col, y='percentage', color=y if y else None and y in df.columns, title=f"Percentage Distribution of {col}")
        fig.update_yaxes(title='Percentage (%)')
        fig.show()

def analyze_correlation(df, y=None, columns='all', corr=0.5):
    """
    Analyzes correlations within a DataFrame, applying One-Hot Encoding to categorical columns,
    and generates a correlation matrix heatmap. Identifies strongly correlated pairs.

    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - y: str, optional, the name of the target column.
    - columns: list or 'all', specifies the columns to include. Defaults to 'all'.
    - corr: float, optional, the correlation threshold for identifying strong correlations. Defaults to 0.5.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Initialize y_series
    y_series = None
    
    # Filter columns if 'columns' parameter is not 'all'
    if columns != 'all':
        if y and y in df.columns:
            df = df[[*columns, y]]
        elif columns:
            df = df[columns]
    
    # Check and encode y if it's specified and is a column in DataFrame
    if y and y in df.columns and (df[y].dtype == 'object' or df[y].dtype.name == 'category'):
        y_series = df.pop(y)
    
    # Apply One-Hot Encoding to categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Re-add y after encoding if it was removed
    if y_series is not None:
        le = LabelEncoder()
        df[y] = le.fit_transform(y_series)
    
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
    Identifies and counts outliers in the specified columns of a DataFrame based on the standard deviation,
    and calculates the percentage of outliers relative to the total number of observations in each column.
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame.
    - columns: list or 'all', optional, specifies the columns to check for outliers. Defaults to 'all'.
    - std: int or float, the number of standard deviations to use for defining outliers. Defaults to 3.
    
    Returns:
    A dictionary where keys are column names and values are tuples containing the counts of outliers and
    their percentage of the total observations.
    """
    # Filter columns if 'columns' parameter is not 'all'
    if columns == 'all':
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    else:
        numerical_cols = [col for col in columns if col in df.select_dtypes(include=['int64', 'float64']).columns]
    
    outliers_info_dict = {}
    
    for col in numerical_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        # Defining outliers as those more than 'std' standard deviations from the mean
        outliers = df[(df[col] < col_mean - std * col_std) | (df[col] > col_mean + std * col_std)]
        outliers_count = outliers.shape[0]
        total_count = df[col].count()
        
        # Calculate the percentage of outliers
        outliers_percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0
        
        if outliers_count > 0:
            outliers_info_dict[col] = (outliers_count, outliers_percentage)
    
    return outliers_info_dict

def dataset_overview_simplified(df, remove_nan=False):
    """
    Analyzes a given DataFrame and provides a simplified overview of its structure and contents.
    Optionally removes rows with NaN values directly in the passed DataFrame if remove_nan is True.

    Parameters:
    - df: pd.DataFrame - The DataFrame to be analyzed.
    - remove_nan: bool - If True, removes rows with any NaN values in the original DataFrame.

    Returns:
    - overview_df: pd.DataFrame - A DataFrame containing the summary of the dataset's structure and content.
    """
    
    # Removing NaN values if requested
    if remove_nan:
        df.dropna(inplace=True)
    
    # Creating summary DataFrame after any modifications
    overview_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique(),
        "NaN Values": df.isnull().sum()
    })

    return overview_df

def apply_pca_numerical_only(df, columns='all', pca_min=0.5):
    """
    Applies PCA to the given DataFrame, focusing only on numerical columns.
    Warns if categorical columns are detected and excludes them from processing.
    Returns variance ratios in a DataFrame and prints details about PCA application.

    Parameters:
    - df: pd.DataFrame - The DataFrame to process.
    - columns: list or 'all' - Specific columns to consider or 'all' for the entire DataFrame.
    - pca_min: float - Minimum variance that PCA must explain.

    Returns:
    - pca_variance_df: pd.DataFrame - A DataFrame containing the variance ratio for each PCA component.
    """
    if columns != 'all':
        df = df[columns]
    
    # Identify numerical and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    # Warn if categorical columns are present
    if not categorical_features.empty:
        warnings.warn("Categorical columns detected and will not be used in PCA.")
    
    original_dimensions = len(numeric_features)
    
    # Apply StandardScaler to numerical columns
    scaler = StandardScaler()
    df_numerical_scaled = scaler.fit_transform(df[numeric_features])
    
    # Apply PCA
    pca = PCA(n_components=pca_min, svd_solver='full')
    pca.fit(df_numerical_scaled)
    pca_components = pca.n_components_
    
    # Print details
    print(f"Minimum variance for PCA: {pca_min}")
    print(f"Original number of dimensions: {original_dimensions}")
    print(f"Number of PCA components: {pca_components}")
    
    # Prepare the variance ratio DataFrame
    pca_variance_df = pd.DataFrame(pca.explained_variance_ratio_, columns=['Variance Ratio'],
                                   index=[f'PCA {i+1}' for i in range(pca_components)])
    
    return pca_variance_df