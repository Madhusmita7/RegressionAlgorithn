import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Downloads\emp_sal.csv")

x= dataset.iloc[:,1:2].values

y= dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y, color ='red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('linear regression model(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

m = lin_reg.coef_
print(m)

c= lin_reg.intercept_
print(c)

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

#polynomial regression(non linear model)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x,y, color ='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('polymodel(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

#support vector regression model
from sklearn.svm import SVR
svr_reg = SVR(max_iter=-1)
svr_reg.fit(x,y)
svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf', degree=10,gamma='scale', C=1.0)
svr_reg.fit(x,y)
svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred

# K Nearest neighbour
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(x,y)
knn_model_pred = knn_reg.predict([[6.5]])
knn_model_pred


knn_reg = KNeighborsRegressor(n_neighbors = 3, weights = 'distance')
knn_reg.fit(x,y)
knn_model_pred = knn_reg.predict([[6.5]])
knn_model_pred

#decission tree regrressor
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x,y)
dt_model_pred = dt_reg.predict([[6.5]])
dt_model_pred

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='absolute_error', splitter='random')
dt_reg.fit(x,y)
dt_model_pred = dt_reg.predict([[6.5]])
dt_model_pred

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(x,y)
rf_model_pred = rf_reg.predict([[6.5]])
rf_model_pred