from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


encoded_labels = {}

for i in data.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    data[i] = encoder.fit_transform(data[i])
    encoded_labels[i] = encoder

X = data.drop(columns=['Sleep Disorder', 'Person ID'])
y = data['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest.fit(X_train, y_train)

y_predict_rf = random_forest.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_predict_rf)
report_rf = classification_report(y_test, y_predict_rf)

print(f"Accuracy : {accuracy_rf * 100:.2f}%")
print("Classification Report:")
print(report_rf)
