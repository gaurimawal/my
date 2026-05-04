# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Set style
sns.set(style="whitegrid")

# -------------------------------
# a. 1D (Linear) Data Visualization
# -------------------------------

plt.figure()
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title("Histogram - Sepal Length")
plt.show()

plt.figure()
sns.boxplot(x=df['sepal width (cm)'])
plt.title("Boxplot - Sepal Width")
plt.show()

# -------------------------------
# b. 2D (Planar) Data Visualization
# -------------------------------

plt.figure()
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df)
plt.title("2D Scatter Plot")
plt.show()

plt.figure()
sns.lineplot(data=df[['sepal length (cm)', 'sepal width (cm)']])
plt.title("Multiple Line Graph")
plt.show()

# -------------------------------
# c. 3D (Volumetric) Data Visualization
# -------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['sepal length (cm)'],
           df['sepal width (cm)'],
           df['petal length (cm)'])

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title("3D Scatter Plot")
plt.show()

# -------------------------------
# d. Temporal Data Visualization
# -------------------------------
# Iris has no time column, so simulate index as time

df['time'] = range(len(df))

plt.figure()
sns.lineplot(x='time', y='sepal length (cm)', data=df)
plt.title("Temporal Line Plot (Simulated Time)")
plt.show()
