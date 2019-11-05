# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:36:03 2019

@author: Adarsh
"""

import pandas as pd
airline = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/Forcasting/Airlines+Data.xlsx")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
airline['mnth_yr'] = airline['Month'].apply(lambda x: x.strftime('%B-%Y')) 
airline=airline.drop("Month", axis=1)    
airline = airline.rename({'mnth_yr': 'Month'}, axis=1)
airline= airline[['Month','Passengers']]

p = airline["Month"][0]
p[0:3]
airline['months']= 0    

for i in range(96):
    p = airline["Month"][i]
    airline['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(airline['months']))
airline1 = pd.concat([airline,month_dummies],axis = 1)

airline1["t"] = np.arange(1,97)

airline1["t_squared"] = airline1["t"]*airline1["t"]
airline1.columns
airline1["log_pas"] = np.log(airline1["Passengers"])
airline1.Passengers.plot()
Train = airline1.head(84)
Test = airline1.tail(12)

# to change the index value in pandas data frame 
#Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear#53.19

##################### Exponential ##############################

Exp = smf.ols('log_pas~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp#46.05

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad#48.05

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea#132.81

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #26.36

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_pas~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea#140.06

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #10.51

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#### so Multi additive seasonality has the least value among the models prepared so far,10.51 rmse.

# Predicting new values 

predict_data = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/Forcasting/Predict_new.csv")
model_full = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airline1).fit()
pred_full=model_full.predict(airline1)
airline1["pred_full"]=pred_full
residuals=pd.DataFrame(np.array(airline1["Passengers"]-np.array(pred_full)))
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_passengers"] = pred_new

#ACF plot for Residuals
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals.values.squeeze(), lags=12)
sm.graphics.tsa.plot_pacf(airline1.Passengers.values.squeeze(), lags=12)

#####

#AR models for Forcasting errors
from statsmodels.tsa.arima_model import ARIMA
pas=airline["Passengers"]

model=ARIMA(residuals,order=(1,0,0)).fit(transparams=True)

forecasterrors=model.forecast(steps=12)[0]
predict_data["forecasted_passengers"] = pd.Series(pred_new)
predict_data["forecasted_errors"] = pd.Series(forecasterrors)
predict_data["improved"] = predict_data["forecasted_passengers"]+predict_data["forecasted_errors"]
predict_data["forecasted_passengers"].plot()


