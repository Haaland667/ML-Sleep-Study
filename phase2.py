import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns




# Pre-process the dataset
# Dropping irrelevant columns for classification
columns_to_drop = ['Person ID', 'Occupation', 'Blood Pressure']
data = dataset.drop(columns=columns_to_drop)

# Encode categorical features
categorical_columns = ['Gender', 'BMI Category', 'Sleep Disorder']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target variable (classification target: Sleep Disorder)
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=7),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
}

# Evaluate models
for model_name, model in models.items():
    print(f"\n{'='*20} {model_name} {'='*20}\n")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Classification metrics
    print(f"{model_name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

    if y_pred_prob is not None:
        # Handle binary or multi-class classification
        if len(y_pred_prob.shape) == 1:  # Binary case, ensure it's 2D
            y_pred_prob = np.column_stack([1 - y_pred_prob, y_pred_prob])

        # Binarize the labels for multi-class ROC AUC calculation
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))

        # Compute AUC score for multi-class using 'ovr' (One-vs-Rest)
        roc_auc = roc_auc_score(y_test_binarized, y_pred_prob, multi_class='ovr')
        print(f"ROC AUC Score: {roc_auc:.4f}")

        # Plot ROC curves for each class
        plt.figure()
        for i in range(y_test_binarized.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f'Class {i}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve (Multi-class)')
        plt.legend(loc='lower right')
        plt.show()

print('\nVizualisation of the models: \n')

# Visualize Decision Tree
if 'Decision Tree' in models:
    decision_tree_model = models['Decision Tree']
    plt.figure(figsize=(40, 30))
    plot_tree(
        decision_tree_model,
        feature_names=X.columns,
        class_names=[str(cls) for cls in label_encoders['Sleep Disorder'].classes_],
        filled=True
    )
    plt.title('Decision Tree Visualization')
    plt.show()


# Visualize Random Forest Feature Importance
if 'Random Forest' in models:
    random_forest_model = models['Random Forest']
    feature_importances = random_forest_model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=X.columns)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

# Visualize Logistic Regression Coefficients
if 'Logistic Regression' in models:
    logistic_model = models['Logistic Regression']
    coefficients = logistic_model.coef_[0]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients, y=X.columns)
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()