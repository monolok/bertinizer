import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

def plot_dataset(df, y=None):
    """
    Automatically generates plots for all numerical and categorical columns in a DataFrame.
    If a target variable 'y' is provided, it also generates a correlation matrix plot.
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - y: str, optional, the name of the target column.
    """
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
        # Encoding categorical variables if y is categorical
        if df[y].dtype == 'object' or df[y].dtype.name == 'category':
            le = LabelEncoder()
            df[y] = le.fit_transform(df[y])
            
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

# Example usage:
# plot_dataset(your_dataframe, 'your_target_column')