import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def perform_eda(file_path):
    """
    Comprehensive EDA for match prediction dataset

    Parameters:
    file_path (str): Path to the CSV file
    """
    # Load the preprocessed data
    # Note: We're assuming the data has already gone through cleaning steps
    df = pd.read_csv(file_path)

    # 1. Basic Dataset Overview
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe().T)  # Transposed for better readability

    # 2. Target Variable Analysis
    print("\n"+"="*80)
    print("TARGET VARIABLE ANALYSIS")
    print("="*80)

    # Let's examine the distribution of the result (target) variable
    print("Result value counts:")
    result_counts = df['result'].value_counts(normalize=True) * 100
    print(result_counts)

    # Visualize the distribution of results
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='result', data=df)
    plt.title('Distribution of Match Results')
    plt.xlabel('Result')
    plt.ylabel('Count')

    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('result_distribution.png')
    plt.show()

    # 3. Feature Distribution Analysis
    print("\n"+"="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'result']  # Exclude target variable

    # Plot histograms for numeric features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols[:9], 1):  # Limit to 9 plots
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()

    plt.savefig('feature_distributions.png')
    plt.show()

    # 4. Correlation Analysis
    print("\n"+"="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Print correlation with target variable
    print("Correlation with target variable (result):")
    print(corr_matrix['result'].sort_values(ascending=False))

    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .8})

    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()

    # 5. Feature Relationships with Target
    print("\n"+"="*80)
    print("FEATURE RELATIONSHIPS WITH TARGET")
    print("="*80)

    # Identify top correlated features with result
    top_correlated = corr_matrix['result'].sort_values(ascending=False).index[:6]  # Top 5 + result itself
    print("Top correlated features with result:")
    print(top_correlated)

    # Plot boxplots for top features by result
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_correlated[1:6], 1):  # Skip 'result' itself
        plt.subplot(2, 3, i)
        sns.boxplot(x='result', y=feature, data=df)
        plt.title(f'{feature} by result')
        plt.tight_layout()

    plt.savefig('features_by_result.png')
    plt.show()

    # 6. Statistical Tests
    print("\n"+"="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    # If result is binary (0/1), we can do t-tests to see if means differ significantly
    if len(df['result'].unique()) == 2:
        result_values = df['result'].unique()
        print("T-tests for difference in means between result groups:")

        for feature in numeric_cols:
            group1 = df[df['result'] == result_values[0]][feature]
            group2 = df[df['result'] == result_values[1]][feature]

            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            significance = "Significant" if p_val < 0.05 else "Not significant"

            print(f"{feature}: t-stat={t_stat:.3f}, p-value={p_val:.3g} - {significance}")

    # 7. Map Analysis (if map is categorical)
    if 'map' in df.columns:
        print("\n"+"="*80)
        print("MAP ANALYSIS")
        print("="*80)

        # Map distribution
        map_counts = df['map'].value_counts(normalize=True) * 100
        print("Distribution of maps:")
        print(map_counts)

        # Win rate by map
        if len(df['result'].unique()) == 2:
            print("\nWin rate by map:")
            win_rate_by_map = df.groupby('map')['result'].mean() * 100
            print(win_rate_by_map)

            # Visualize win rate by map
            plt.figure(figsize=(12, 6))
            win_rate_by_map.sort_values().plot(kind='barh')
            plt.title('Win Rate by Map (%)')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Map')
            plt.tight_layout()
            plt.savefig('win_rate_by_map.png')
            plt.show()

    # 8. Performance Metrics Analysis
    print("\n"+"="*80)
    print("PERFORMANCE METRICS ANALYSIS")
    print("="*80)

    # If we have performance metrics like kills, deaths, etc.
    performance_metrics = [col for col in df.columns if col in ['kills', 'deaths', 'assists', 'mvps', 'hs_percent', 'points']]

    if performance_metrics:
        # Pairplot of performance metrics colored by result
        if len(df['result'].unique()) <= 5:  # Only if reasonable number of result categories
            plt.figure(figsize=(15, 12))
            sns.pairplot(df[performance_metrics + ['result']],
                         hue='result', diag_kind='kde')
            plt.suptitle('Relationships Between Performance Metrics', y=1.02)
            plt.tight_layout()
            plt.savefig('performance_metrics_pairplot.png')
            plt.show()

        # Create scatter plot matrix for key performance metrics
        plt.figure(figsize=(14, 10))
        for i, x_feature in enumerate(performance_metrics[:3], 1):
            for j, y_feature in enumerate(performance_metrics[1:4], 1):
                if x_feature != y_feature:
                  plt.subplot(3, 3, (i-1)*3 + j - (1 if j > i else 0))
                  sns.scatterplot(x=x_feature, y=y_feature, hue='result', data=df, alpha=0.6)
                  plt.title(f'{x_feature} vs {y_feature}')
                  plt.tight_layout()

        plt.savefig('performance_metrics_scatter.png')
        plt.show()

    # 9. Key Insights Summary
    print("\n"+"="*80)
    print("KEY INSIGHTS SUMMARY")
    print("="*80)

    # Calculate win rate (if binary result)
    if len(df['result'].unique()) == 2:
        win_rate = df['result'].mean() * 100
        print(f"Overall win rate: {win_rate:.2f}%")

    # Most significant features (based on correlation)
    print("\nMost significant features (correlation with result):")
    top_features = corr_matrix['result'].abs().sort_values(ascending=False).iloc[1:6]
    for feature, corr in top_features.items():
        print(f"  {feature}: {corr:.3f}")

    # Performance metrics averages
    if performance_metrics:
        print("\nAverage performance metrics:")
        for metric in performance_metrics:
            print(f"  Average {metric}: {df[metric].mean():.2f}")

    return df

# Example usage
df = perform_eda('cleaned_match_data.csv')