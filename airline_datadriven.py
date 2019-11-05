# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:27:51 2019

@author: Adarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
#from sm.tsa.statespace import sa
airline = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/Forcasting/Airlines+Data.xlsx")

# Converting the normal index of airline to time stamp 
airline.index = pd.to_datetime(airline.Month,format="%b-%y")

airline.Passengers.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
airline["Date"] = pd.to_datetime(airline.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

airline["month"] = airline.Date.dt.strftime("%b") # month extraction
#Amtrak["Day"] = Amtrak.Date.dt.strftime("%d") # Day extraction
#Amtrak["wkday"] = Amtrak.Date.dt.strftime("%A") # weekday extraction
airline["year"] = airline.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=airline,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=airline)
sns.boxplot(x="year",y="Passengers",data=airline)
#sns.factorplot("month","Passengers",data=airline,kind="box")

# Line plot for passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=airline)


# moving average for the time series to understand better about the trend character in airline
airline.Passengers.plot(label="org")
for i in range(2,24,6):
    airline["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(airline.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(airline.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(airline.Passengers,lags=12)
tsa_plots.plot_pacf(airline.Passengers,lags=12)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = airline.head(84)
Test = airline.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Train["Passengers"] = Train["Passengers"].astype('double')
# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers) # 14.23

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) # 11.84



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) # 1.61



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers) # 2.82


# Lets us use auto_arima from p
from pmdarima.arima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=4,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() # SARIMAX(1, 1, 1)x(0, 1, 1, 12)
# AIC ==> 1348.728
# BIC ==> 1362.665

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Passengers)  # 3.49


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(airline.Passengers,order=(1,1,0)).fit(transparams=True)
forecasterrors=model.forecast(steps=12)[0]  #it will give the next 12 values
