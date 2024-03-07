import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
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

def find_pca_num_only(df, columns='all', pca_min=0.10):
    """
    Applies PCA or TruncatedSVD to the given DataFrame, focusing only on numerical columns,
    depending on whether the input is dense or sparse. Checks if removing NaN values results
    in losing more than 15% of the data. If so, issues a warning and halts further processing.
    Otherwise, returns a DataFrame that includes variance ratios for PCA components or
    TruncatedSVD singular values, identifies non-processed categorical columns,
    and integrates information on missing value removal.

    Parameters:
    - df: pd.DataFrame - The DataFrame to process.
    - columns: list or 'all' - Specific columns to consider or 'all' for the entire DataFrame.
    - pca_min: float - Minimum variance that PCA must explain.

    Returns:
    - result_df: pd.DataFrame - A DataFrame containing the PCA analysis summary and missing value information, or an empty DataFrame with a warning if significant data loss is detected.
    """
    if columns != 'all':
        df = df[columns]
    
    original_size = len(df)
    df_dropped_na = df.dropna()
    removed_na_count = original_size - len(df_dropped_na)
    removed_na_percentage = (removed_na_count / original_size) * 100 if original_size > 0 else 0
    
    if removed_na_percentage > 15:
        warnings.warn("Removing NaN values results in losing more than 15% of the dataset. Consider alternative missing value handling.")
        return pd.DataFrame()
    
    numeric_features = df_dropped_na.select_dtypes(include=np.number).columns
    categorical_features = df_dropped_na.select_dtypes(exclude=np.number).columns

    is_sparse = df_dropped_na[numeric_features].sparse.density < 1 if hasattr(df_dropped_na[numeric_features], 'sparse') else False

    dim_reduction_method = TruncatedSVD(n_components=pca_min, algorithm='randomized') if is_sparse else PCA(n_components=pca_min, svd_solver='full')
    scaler = StandardScaler(with_mean=not is_sparse)  # Disable mean centering for sparse data to avoid converting to dense

    # Process only if there are numeric features
    if len(numeric_features) > 0:
        pipeline = make_pipeline(scaler, dim_reduction_method)
        pipeline.fit(df_dropped_na[numeric_features])
        components_count = dim_reduction_method.n_components_ if is_sparse else dim_reduction_method.n_components_
        variance_ratios = dim_reduction_method.explained_variance_ratio_ if not is_sparse else 'N/A for SVD'
        
    summary_info = {
        'Component': [],
        'Variance Ratio': [],
        'Status': [],
        'Info': []
    }
    
    for i in range(components_count):
        summary_info['Component'].append(f'PCA/SVD {i+1}')
        summary_info['Variance Ratio'].append(variance_ratios[i] if variance_ratios != 'N/A for SVD' else 'N/A')
        summary_info['Status'].append('Processed')
        summary_info['Info'].append('')
        
    for cat_col in categorical_features:
        summary_info['Component'].append(cat_col)
        summary_info['Variance Ratio'].append('N/A')
        summary_info['Status'].append('Non-processed (Categorical)')
        summary_info['Info'].append('')
    
    summary_info['Component'].append('Missing Value Info')
    summary_info['Variance Ratio'].append('N/A')
    summary_info['Status'].append('Missing Value Removal')
    summary_info['Info'].append(f'Missing values removed: {removed_na_count}, Percentage of total: {removed_na_percentage:.2f}%')

    if components_count > 0:
        summary_info['Info'][0] = f'Minimum variance for PCA/SVD: {pca_min}, Original dimensions: {len(numeric_features)}, Components: {components_count}'
    
    result_df = pd.DataFrame(summary_info)
    return result_df.reset_index(drop=True)

def explode_datetime_col(df, col_name):
    """
    Expands a datetime column into separate columns for hour, day of month, day of week, and day name.

    This function takes a DataFrame and a column name as input. It converts the specified column to datetime format 
    (if not already in that format) and then extracts and adds new columns to the DataFrame for the hour, day of the 
    month, day of the week, and a descriptive day name (e.g., "Monday", "Tuesday").

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the datetime column to be expanded.
    - col_name (str): The name of the column in the DataFrame that contains datetime information.

    Returns:
    - pd.DataFrame: The original DataFrame with the added columns for hour ('Hour'), day of the month ('Day'), 
      day of the week ('DayOfWeek'), and the descriptive name of the day ('DayName'). Note that the function modifies 
      the DataFrame in place and also returns it.

    Example usage:
    >>> df = pd.DataFrame({'Date/Time': ['2021-01-01 12:00:00', '2021-01-02 13:00:00']})
    >>> df = explode_datetime_col(df, 'Date/Time')
    >>> df.columns
    Index(['Date/Time', 'Hour', 'Day', 'DayOfWeek', 'DayName'], dtype='object')
    """
    # Convert 'Date/Time' to datetime
    df[col_name] = pd.to_datetime(df[col_name])
    # Extract hour and create a new column
    df['Hour'] = df[col_name].dt.hour
    # Extract day and create a new column (day of month)
    df['Day'] = df[col_name].dt.day
    # Extract day of week and create a new column
    df['DayOfWeek'] = df[col_name].dt.dayofweek  # Monday=0, Sunday=6
    # Extract a descriptive day name and create a new column
    df['DayName'] = df[col_name].dt.day_name()

    return df

def plot_optimal_k_kmeans(X, max_k=10):
    """
    Plots the WCSS values against the number of clusters for KMeans clustering using the Elbow Method.
    Pinpoints the second and third potential optimal k values directly on the plot.
    
    Parameters:
    - X: feature array or DataFrame to cluster.
    - max_k: maximum number of clusters to try (default is 10).
    
    The function does not return any value but displays a plot with the WCSS curve and marks the second and third suggested optimal k values.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=0, init='k-means++', n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Calculate gradients and second derivatives to find elbow points
    gradients = np.diff(wcss) * -1
    second_derivative = np.diff(gradients)

    # Identifying elbow points; focusing on the second and a new third potential elbow
    elbow_point1 = np.argmin(np.diff(gradients)) + 1
    elbow_point2 = np.argmin(second_derivative) + 2  # Second potential elbow
    # Introducing a third potential elbow point based on further analysis (for simplicity, we take the next significant change)
    elbow_point3 = elbow_point2 + np.argmin(second_derivative[elbow_point2:]) + 1  # This is a simplified assumption

    # Create DataFrame for plotting
    wcss_frame = pd.DataFrame({'Number of Clusters': range(1, max_k + 1), 'WCSS': wcss})

    # Plotting
    fig = px.line(wcss_frame, x='Number of Clusters', y='WCSS', markers=True, title='WCSS per Number of Clusters')
    fig.update_layout(yaxis_title="WCSS", xaxis_title="Number of Clusters")
    
    # Add markers for suggested optimal k values (focusing on elbow_point2 and elbow_point3)
    fig.add_scatter(x=[elbow_point2 + 1, elbow_point3 + 1], y=[wcss[elbow_point2], wcss[elbow_point3]],
                    mode='markers', marker=dict(size=10, color='Red'),
                    name='Suggested k')

    # Show the plot
    fig.show()