import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\Downloads\Investment.csv")
 x= dataset.iloc[:,:-1]
 y= dataset.iloc[:,4]

x= pd.get_dummies(x, dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m_slope = regressor.coef_
print(m_slope)

c_inter= regressor.intercept_
print(c_inter)

x= np.append(arr = np.ones((50,1)).astype(int),values=x, axis = 1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary() 

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]

regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary() 

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]

regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2]]

regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1]]

regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)