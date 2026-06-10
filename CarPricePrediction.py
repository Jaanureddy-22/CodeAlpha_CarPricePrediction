import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset
df = pd.read_csv("car data.csv")

# Display First 5 Rows
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Create Car Age Feature
df['Car_Age'] = 2025 - df['Year']

# Drop Unnecessary Columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Convert Categorical Columns into Numerical
df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("------------------")
print("R2 Score :", round(r2, 4))
print("MAE      :", round(mae, 4))
print("RMSE     :", round(rmse, 4))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# User Prediction
print("\nEnter Car Details")

present_price = float(input("Present Price (Lakhs): "))
driven_kms = int(input("Driven Kilometers: "))
owner = int(input("Number of Previous Owners: "))
car_age = int(input("Car Age (Years): "))

fuel_type = input("Fuel Type (Petrol/Diesel/CNG): ").lower()
selling_type = input("Selling Type (Dealer/Individual): ").lower()
transmission = input("Transmission (Manual/Automatic): ").lower()

fuel_diesel = 1 if fuel_type == "diesel" else 0
fuel_petrol = 1 if fuel_type == "petrol" else 0

selling_individual = 1 if selling_type == "individual" else 0

transmission_manual = 1 if transmission == "manual" else 0

input_data = pd.DataFrame([[
    present_price,
    driven_kms,
    owner,
    car_age,
    fuel_diesel,
    fuel_petrol,
    selling_individual,
    transmission_manual
]], columns=X.columns)

predicted_price = model.predict(input_data)

print("\nPredicted Selling Price: ₹", round(predicted_price[0], 2), "Lakhs")