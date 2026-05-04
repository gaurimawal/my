# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("AirQuality.csv", sep=';')

# -----------------------------
# 1) DATA CLEANING
# -----------------------------

# Remove unnamed columns if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Replace missing values (-200 used in dataset)
df.replace(-200, np.nan, inplace=True)

# Convert Date and Time
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S', errors='coerce').dt.time

# Fill missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print("Cleaned Data:")
print(df.head())


# -----------------------------
# 2) DATA INTEGRATION (Split & Merge)
# -----------------------------

# Split dataset into two parts
df1 = df.iloc[:, :7]
df2 = df.iloc[:, 7:]

# Merge again column-wise
df_merged = pd.concat([df1, df2], axis=1)

print("Merged Data:")
print(df_merged.head())


# -----------------------------
# 3) ERROR CORRECTING
# -----------------------------

# Remove negative values (invalid readings)
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Replace again after correction
df.fillna(df.mean(numeric_only=True), inplace=True)

print("Error Corrected Data:")
print(df.head())


# -----------------------------
# 4) DATA MODEL BUILDING
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Predict Temperature (T)
X = df.drop(['T', 'Date', 'Time'], axis=1)
y = df['T']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Model Built Successfully")


# -----------------------------
# 5) DATA VISUALIZATION
# -----------------------------

# Boxplot
plt.figure()
sns.boxplot(data=df[['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T']])
plt.title("Boxplot")
plt.show()

# Histogram
plt.figure()
df['T'].hist()
plt.title("Histogram of Temperature")
plt.show()

# Single Line Graph
plt.figure()
plt.plot(df['T'])
plt.title("Temperature Trend")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.show()

# Multiple Line Graph
plt.figure()
plt.plot(df['CO(GT)'], label='CO')
plt.plot(df['NO2(GT)'], label='NO2')
plt.legend()
plt.title("Multiple Line Graph")
plt.show()
