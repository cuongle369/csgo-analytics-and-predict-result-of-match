import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def clean_match_prediction_data(file_path):
    """
    Load and clean the match prediction dataset, starting with dropping specified columns

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pandas.DataFrame: Cleaned dataset ready for modeling
    """
    # 1. Load the data
    df = pd.read_csv(file_path)

    # 2. Initial data exploration before dropping columns
    print(f"Original dataset shape: {df.shape}")
    print("\nOriginal columns:")
    print(df.columns.tolist())

    # 3. Drop specified columns first as requested
    df = df.drop(['day', 'month', 'year', 'date', 'wait_time_s', 'match_time_s', 'team_a_rounds', 'team_b_rounds'],
                 axis=1)

    print(f"\nDataset shape after dropping columns: {df.shape}")
    print("\nRemaining columns:")
    print(df.columns.tolist())

    # 4. Check for missing values in remaining columns
    missing_values = df.isnull().sum()
    print("\nMissing values in remaining columns:")
    print(missing_values)

    # 5. Handle missing values in remaining columns
    # For numeric columns
    numeric_cols = ['ping', 'kills', 'assists', 'deaths', 'mvps', 'hs_percent', 'points']

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")

    # For categorical columns
    categorical_cols = ['map', 'result']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled {col} missing values with mode: {mode_val}")

    # 6. Encode categorical variables
    # Handle 'map' column
    if df['map'].dtype == 'object':
        # Create mapping dictionary for interpretability
        map_mapping = {map_name: idx for idx, map_name in enumerate(df['map'].unique())}
        print(f"\nMap encoding mapping:")
        for map_name, code in map_mapping.items():
            print(f"  {map_name} -> {code}")

        # Convert to numeric
        df['map'] = df['map'].map(map_mapping)

    # Store original result values before encoding
    if df['result'].dtype == 'object':
        result_categories = df['result'].unique()
        result_mapping = {result: idx for idx, result in enumerate(result_categories)}
        print(f"\nResult encoding mapping:")
        for result, code in result_mapping.items():
            print(f"  {result} -> {code}")

        # Convert to numeric
        df['result'] = df['result'].map(result_mapping)

    # 7. Feature engineering with remaining columns
    # Calculate K/D ratio
    df['kd_ratio'] = df['kills'] / df['deaths'].replace(0, 1)  # Avoid division by zero

    # Calculate headshot efficiency (hs_percent * kills)
    df['hs_efficiency'] = df['hs_percent'] * df['kills'] / 100

    # Calculate impact score (custom metric)
    df['impact_score'] = (df['kills'] + df['assists'] * 0.5 + df['mvps'] * 2) / (df['deaths'] + 1)

    # Calculate KAST approximation (simplified version of Kills, Assists, Survived, Traded)
    df['kast_approx'] = (df['kills'] + df['assists']) / (df['deaths'] + df['kills'] + df['assists'])

    # 8. Handle outliers in numeric columns
    print("\nHandling outliers in numeric columns...")

    for col in numeric_cols + ['kd_ratio', 'impact_score', 'hs_efficiency', 'kast_approx']:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outliers > 0:
            print(f"  Capping {outliers} outliers in {col}")

            # Cap the outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 9. Feature scaling for remaining numeric features
    print("\nScaling numeric features...")

    # Identify columns to scale
    cols_to_scale = numeric_cols + ['kd_ratio', 'impact_score', 'hs_efficiency', 'kast_approx']

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # 10. Final check and statistics
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns in final dataset: {df.columns.tolist()}")

    # Correlation with target
    correlation_with_result = df.corr()['result'].sort_values(ascending=False)
    print("\nTop correlations with match result:")
    print(correlation_with_result)

    # 11. Data quality summary
    print("\nSummary statistics for cleaned data:")
    print(df.describe())

    return df


# Function to create visualizations
def visualize_features(df):
    """
    Create informative visualizations for the cleaned data

    Parameters:
    df (pandas.DataFrame): Cleaned dataset
    """
    # Create a figure for multiple plots
    plt.figure(figsize=(16, 12))

    # Plot 1: Correlation heatmap
    plt.subplot(2, 2, 1)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap')

    # Plot 2: Distribution of performance metrics by result
    plt.subplot(2, 2, 2)
    sns.boxplot(x='result', y='impact_score', data=df)
    plt.title('Impact Score by Match Result')

    # Plot 3: Distribution of K/D ratio by result
    plt.subplot(2, 2, 3)
    sns.boxplot(x='result', y='kd_ratio', data=df)
    plt.title('K/D Ratio by Match Result')

    # Plot 4: Feature importance (based on correlation with result)
    plt.subplot(2, 2, 4)
    feature_importance = abs(df.corr()['result']).sort_values(ascending=False).drop('result')
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance (Correlation with Result)')
    plt.tight_layout()

    plt.savefig('match_prediction_features.png')
    plt.show()

    return


# Example usage
df = clean_match_prediction_data('csgo.csv')
visualize_features(df)
df.to_csv('cleaned_match_data.csv', index=False)