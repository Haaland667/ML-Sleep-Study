#OUTLIERS

col=['Daily Steps','Stress Level','Physical Activity Level','Heart Rate','Quality of Sleep','Age','Sleep Duration']
for i in col:
    detected_outliers = detect_outliers(data, i)
    print(f"Outliers in {i}: {detected_outliers}")


#MISSING VALUES

missing_values = data.isnull().sum()
print(missing_values)



#REPLACING NONE VALUES BY NO

data['Sleep Disorder'] = data['Sleep Disorder'].fillna('No')


#CHECK AGAIN THE MISSING VALS

missing_values = data.isnull().sum()
print(missing_values)