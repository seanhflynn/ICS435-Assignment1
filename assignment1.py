import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=5)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
knn.fit(X_train_scaled, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = decision_tree.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=1),
        "Recall": recall_score(y_true, y_pred, zero_division=1),
        "F1 Score": f1_score(y_true, y_pred, zero_division=1),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

# Evaluate models
results = {
    "KNN": evaluate_model(y_test, y_pred_knn),
    "Decision Tree": evaluate_model(y_test, y_pred_dt),
    "Random Forest": evaluate_model(y_test, y_pred_rf)
}

# Print results
for model, metrics in results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()

# Hyperparameter tuning
knn_tuned = KNeighborsClassifier(n_neighbors=10)
dt_tuned = DecisionTreeClassifier(max_depth=5, random_state=42)
rf_tuned = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

knn_tuned.fit(X_train_scaled, y_train)
dt_tuned.fit(X_train, y_train)
rf_tuned.fit(X_train, y_train)

y_pred_knn_tuned = knn_tuned.predict(X_test_scaled)
y_pred_dt_tuned = dt_tuned.predict(X_test)
y_pred_rf_tuned = rf_tuned.predict(X_test)

tuned_results = {
    "KNN (n=10)": evaluate_model(y_test, y_pred_knn_tuned),
    "Decision Tree (max_depth=5)": evaluate_model(y_test, y_pred_dt_tuned),
    "Random Forest (max_depth=5)": evaluate_model(y_test, y_pred_rf_tuned)
}

# Print tuned results
for model, metrics in tuned_results.items():
    print(f"{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()

# Plot comparisons
def plot_comparison(results, title):
    df = pd.DataFrame(results).T.drop(columns=["Confusion Matrix"])
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.show()

plot_comparison(results, "Model Performance Comparison")
plot_comparison(tuned_results, "Tuned Model Performance Comparison")
