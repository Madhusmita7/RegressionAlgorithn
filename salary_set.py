import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

#Load the dataset
dataset = pd.read_csv(r"D:\DATA SCIENCE\Salary_Data.csv")

# Split the dataset into independent and dependent variables
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

# Split the dataset into training and testing sets(80-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

 # Predict the test set
 y_pred = regressor.predict(x_test)

# Comparision for y_test vs y_pred
compsarison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

# Visualize the training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set) ')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set) ')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f" Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f" Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model perfomance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f" Training score(R^2): {bias:.2f}")
print(f" Testing score (R^2): {variance:.2f}")
print(f" Training MSE: {train_mse:.2f}")
print(f" test MSE: {test_mse:.2f}")

#Save the trained model to disk
import pickle
filename = 'Linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as Linear_regression_model.pkl")

import _os
print(os.getcwd())
