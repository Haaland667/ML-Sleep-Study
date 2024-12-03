from google.colab import files // Data retrieve
uploaded = files.upload()

import pandas as pd  // Main library import 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('sleep.csv') // Data sheet statistical vizualisation
df.describe()
numerical_df = df.select_dtypes(include=np.number)
numerical_df.median()

//Heat map vizualisation
nu_col=['Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','Heart Rate','Daily Steps']
plt.figure(figsize=(10,10))
sns.heatmap(data=df[nu_col].corr(),cmap='rocket_r',annot=True,fmt='0.1g',vmin=-1,vmax=1)
plt.show()


// Correlation study between variables :

//Age vs Duration/Quality
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='Age', y='Sleep Duration', hue='Quality of Sleep')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0)
plt.title('Age vs. Sleep Duration by Quality of Sleep')
plt.show()

// Heart rate vs Duration/Disorder
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Heart Rate', y='Sleep Duration', hue='Sleep Disorder')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)
plt.title('Heart Rate vs. Sleep Duration by Sleep Disorder')
plt.show()

// Physical activity vs Stress/Duration
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Physical Activity Level', y='Stress Level', hue='Sleep Duration')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0)
plt.title('Physical Activity Level vs. Stress Level by Sleep Duration')
plt.show()

// Occupation vs Duration/Quality
sns.scatterplot(data=df, x='Occupation', y='Sleep Duration', hue='Quality of Sleep')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)
plt.xticks(rotation=90)
plt.title('Occupation vs. Sleep Duration by Quality of Sleep')
plt.show()

// Histogram vizualisations :

//Duration by occupation
occupation_sleep=df.groupby('Occupation')['Sleep Duration'].mean().sort_values()
plt.figure(figsize=(10,6))
ax=sns.barplot(x=occupation_sleep.index,y=occupation_sleep.values,palette='spring')
ax.bar_label(ax.containers[0])
plt.xticks(rotation=90)
plt.show()

//Quality by occupation
occupation_sleep=df.groupby('Occupation')['Quality of Sleep'].mean().sort_values()
plt.figure(figsize=(10,6))
ax=sns.barplot(x=occupation_sleep.index,y=occupation_sleep.values,palette='autumn')
ax.bar_label(ax.containers[0])
plt.xticks(rotation=90)
plt.show()

// Finding outlier in the data
def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers.tolist()

# Testing on our dataset
for column in ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']:
    outliers = find_outliers_iqr(df, column)
    print(f"Outliers in {column}: {outliers}")
