# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:39:35 2019

@author: Adarsh
"""

import pandas as pd
plastic = pd.read_csv("E:/ADM/Excelr solutions/DS assignments/Forcasting/PlasticSales.csv")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = plastic["Month"][0]
p[0:3]
plastic['months']= 0    

for i in range(60):
    p = plastic["Month"][i]
    plastic['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(plastic['months']))
plastic1 = pd.concat([plastic,month_dummies],axis = 1)

plastic1["t"] = np.arange(1,61)

plastic1["t_squared"] = plastic1["t"]*plastic1["t"]
plastic1.columns
plastic1["log_pas"] = np.log(plastic1["Sales"])
plastic1.Sales.plot()
Train = plastic1.head(48)
Test = plastic1.tail(12)

# to change the index value in pandas data frame 
#Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear#260.93

##################### Exponential ##############################

Exp = smf.ols('log_pas~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp#268.69

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad#297.40

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea#235.60

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #218.19

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_pas~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea#239.65

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #160.68

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#### so Multi additive seasonality has the least value among the models prepared so far,10.51 rmse.

# Predicting new values 

predict_data = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/Forcasting/pred_plastic.xlsx")
model_full = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=plastic1).fit()
pred_full=model_full.predict(plastic1)
plastic1["pred_full"]=pred_full
residuals=pd.DataFrame(np.array(plastic1["Sales"]-np.array(pred_full)))
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_sales"] = pred_new

#ACF plot for Residuals
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals.values.squeeze(), lags=12)
sm.graphics.tsa.plot_pacf(plastic1.Sales.values.squeeze(), lags=12)

#####

#AR models for Forcasting errors
from statsmodels.tsa.arima_model import ARIMA
pas=plastic["Sales"]

model=ARIMA(residuals,order=(1,0,0)).fit(transparams=True)

forecasterrors=model.forecast(steps=12)[0]
predict_data["forecasted_sales"] = pd.Series(pred_new)
predict_data["forecasted_errors"] = pd.Series(forecasterrors)
predict_data["improved"] = predict_data["forecasted_sales"]+predict_data["forecasted_errors"]
predict_data["forecasted_sales"].plot()


