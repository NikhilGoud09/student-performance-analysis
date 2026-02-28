import pandas as pd

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Show first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============
# data visulizartion
# ============

import matplotlib.pyplot as plt
import seaborn as sns

print("\nPlotting G3 Distribution...")

plt.figure(figsize=(8,5))
sns.histplot(df['G3'], bins=20, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.show()

print("\nCorrelation Matrix:")
print(df[['G1', 'G2', 'G3']].corr())

print("\nHeatmap of Correlation:")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(df[['G1','G2','G3']].corr(), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f")

plt.title("Correlation Between G1, G2, G3")
plt.show()

print("\nAverage G3 by Study Time:")
print(df.groupby('studytime')['G3'].mean())

plt.figure(figsize=(6,4))
df.groupby('studytime')['G3'].mean().plot(kind='bar')

plt.title("Average Final Grade (G3) by Study Time")
plt.xlabel("Study Time Level")
plt.ylabel("Average G3")
plt.show()

print("\nCorrelation Between Absences and G3:")
print(df[['absences', 'G3']].corr())

plt.figure(figsize=(6,4))
sns.scatterplot(x=df['absences'], y=df['G3'])

plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.show()

# ==============================
#  Linear Regression
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\nBuilding Linear Regression Model...")

# Select features (X) and target (y)
X = df[['G1', 'G2', 'studytime', 'absences']]
y = df['G3']

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print("Intercept:", model.intercept_)

X = df[['G1', 'studytime', 'absences']]
