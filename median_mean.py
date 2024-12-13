import pandas as pd
data = pd.read_csv('sleep.csv')

mean_values = data.select_dtypes(include=['number']).mean()

print("Mean of numerical columns")
print(mean_values)



#Median
column = data.select_dtypes(include=np.number)
print(column.median())