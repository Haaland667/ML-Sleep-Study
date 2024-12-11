from google.colab import files  # Upload dataset
uploaded_files = files.upload()

import pandas as pd  # Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load dataset and display basic statistics
data = pd.read_csv('sleep.csv')
print(data.describe())

# Selecting numerical columns and calculating median
numeric_data = data.select_dtypes(include=np.number)
print(numeric_data.median())

# Heatmap visualization
heatmap_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
plt.figure(figsize=(12, 8))
sns.heatmap(data[heatmap_columns].corr(), cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Correlation analysis between variables
# Age vs Sleep Duration/Quality of Sleep
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Age', y='Sleep Duration', hue='Quality of Sleep', palette='viridis')
plt.legend(loc='upper right', framealpha=0.7)
plt.title('Age vs. Sleep Duration by Quality of Sleep')
plt.show()

# Heart Rate vs Sleep Duration/Sleep Disorder
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Heart Rate', y='Sleep Duration', hue='Sleep Disorder', palette='plasma')
plt.legend(loc='upper right', framealpha=0.7)
plt.title('Heart Rate vs. Sleep Duration by Sleep Disorder')
plt.show()

# Physical Activity Level vs Stress Level/Sleep Duration
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Physical Activity Level', y='Stress Level', hue='Sleep Duration', palette='cool')
plt.legend(loc='upper right', framealpha=0.7)
plt.title('Physical Activity Level vs. Stress Level by Sleep Duration')
plt.show()

# Occupation vs Sleep Duration/Quality of Sleep
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Occupation', y='Sleep Duration', hue='Quality of Sleep', palette='cubehelix')
plt.legend(loc='upper right', framealpha=0.7)
plt.xticks(rotation=45)
plt.title('Occupation vs. Sleep Duration by Quality of Sleep')
plt.show()

# Histogram visualizations
# Average Sleep Duration by Occupation
avg_sleep_by_occupation = data.groupby('Occupation')['Sleep Duration'].mean().sort_values()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x=avg_sleep_by_occupation.index, y=avg_sleep_by_occupation.values, palette='summer')
bar_plot.bar_label(bar_plot.containers[0])
plt.xticks(rotation=45)
plt.title('Average Sleep Duration by Occupation')
plt.show()

# Average Quality of Sleep by Occupation
avg_quality_by_occupation = data.groupby('Occupation')['Quality of Sleep'].mean().sort_values()
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x=avg_quality_by_occupation.index, y=avg_quality_by_occupation.values, palette='winter')
bar_plot.bar_label(bar_plot.containers[0])
plt.xticks(rotation=45)
plt.title('Average Quality of Sleep by Occupation')
plt.show()

# Function to find outliers using IQR
def detect_outliers(dataframe, col_name):
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = dataframe[(dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit)][col_name]
    return outliers.tolist()

# Identifying outliers for specific columns
columns_to_check = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
for col in columns_to_check:
    detected_outliers = detect_outliers(data, col)
    print(f"Outliers in {col}: {detected_outliers}")
