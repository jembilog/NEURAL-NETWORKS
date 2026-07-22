import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from neural_network import NeuralNetwork
df = pd.read_csv("employee_promotion.csv")

print(df.head())

numerical = [
    "Age",
    "Experience",
    "MonthlySalary",
    "ProjectsCompleted",
    "PerformanceScore"
]

for col in numerical:
    df[col] = df[col].fillna(df[col].mean())

categorical = [
    "Department",
    "Education",
    "Overtime"
]

for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])
    
df = pd.get_dummies(
    df,
    columns=categorical,
    dtype=int
)
X = df.drop("Promoted", axis=1)

y = df["Promoted"].values.reshape(-1,1)

feature_columns = X.columns.tolist()
scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

nn = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=8,
    output_size=1,
    learning_rate=0.05
)
nn.summary()
nn.train(
    X_train,
    y_train,
    epochs=5000
)
accuracy = nn.accuracy(
    X_test,
    y_test
)
print("\nAccuracy")
print(accuracy)
predictions = nn.predict(X_test)
probabilities = nn.predict_proba(X_test)
print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions
    )

)

print("\nPredictions")

for actual,predicted,prob in zip(
    y_test,
    predictions,
    probabilities
):

    print(
        f"Actual={actual[0]} "
        f"Predicted={predicted[0]} "
        f"Probability={prob[0]:.4f}"
    )
new_employee = pd.DataFrame({
    "Age":[35],
    "Experience":[10],
    "MonthlySalary":[75000],
    "ProjectsCompleted":[9],
    "PerformanceScore":[88],
    "Department":["IT"],
    "Education":["Master"],
    "Overtime":["Yes"]
})
new_employee = pd.get_dummies(

    new_employee,

    columns=categorical,

    dtype=int

)
new_employee = new_employee.reindex(

    columns=feature_columns,

    fill_value=0

)
new_employee_scaled = scaler.transform(

    new_employee

)
prediction = nn.predict(new_employee_scaled)
probability = nn.predict_proba(new_employee_scaled)
print("\n============================")
print("NEW EMPLOYEE")
print("============================")
print(new_employee)
print()
print(
    f"Promotion Probability: "
    f"{probability[0][0]:.4f}"
)
if prediction[0][0] == 1:
    print("Prediction : PROMOTED")
else:
    print("Prediction : NOT PROMOTED")
