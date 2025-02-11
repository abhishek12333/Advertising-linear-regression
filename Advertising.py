import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Load the dataset
data = pd.read_csv('Advertising.csv')

# Show dataset preview in Streamlit
st.title("Advertising Budget vs Sales - Linear Regression")
st.write("This app predicts sales based on the advertising budgets for TV, Radio, and Newspaper.")

# Display dataset information
st.subheader("Dataset Preview")
st.write(data.head())

# Exploratory Data Analysis (EDA)
st.subheader("Data Visualization")

# Plotting the relationships between advertising budgets and sales
st.write("TV vs Sales")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='TV', y='Sales', color='blue')
plt.title('TV Advertising vs Sales')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
st.pyplot()

st.write("Radio vs Sales")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Radio', y='Sales', color='green')
plt.title('Radio Advertising vs Sales')
plt.xlabel('Radio Advertising Budget')
plt.ylabel('Sales')
st.pyplot()

st.write("Newspaper vs Sales")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Newspaper', y='Sales', color='red')
plt.title('Newspaper Advertising vs Sales')
plt.xlabel('Newspaper Advertising Budget')
plt.ylabel('Sales')
st.pyplot()

# Data Preprocessing
X = data[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = data['Sales']  # Dependent variable

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Calculating the performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model evaluation metrics in Streamlit
st.subheader("Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² (Coefficient of Determination): {r2:.2f}")

# Predictions from user input
st.subheader("Predict Sales Based on Advertising Budgets")
tv_budget = st.slider('TV Advertising Budget ($)', 0, 300)
radio_budget = st.slider('Radio Advertising Budget ($)', 0, 300)
newspaper_budget = st.slider('Newspaper Advertising Budget ($)', 0, 300)

input_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], columns=['TV', 'Radio', 'Newspaper'])
predicted_sales = model.predict(input_data)

# Display predicted sales
st.write(f"Predicted Sales: ${predicted_sales[0]:.2f}")