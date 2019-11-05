
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cocacola = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/Forcasting/CocaCola_Sales_Rawdata.xlsx")
quaters = ['Q1','Q2','Q3','Q4']

p = cocacola['Quarter'][0]
p[0:2]

for i in range(42):
    p = cocacola['Quarter'][i]
    cocacola['Quarter'][i] = p[0:2]

qtr_dummies = pd.get_dummies(cocacola["Quarter"])
cola = pd.concat([cocacola,qtr_dummies],axis=1)

cola['t'] = np.arange(1,43)

cola['t_squared'] = cola['t']*cola['t'] 
cola['log_sales'] = np.log(cola['Sales'])
cola.Sales.plot()
Train = cola.head(38)
Test = cola.tail(4)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

lr_ml = smf.ols('Sales~t',data=Train).fit()
pred_lr =  pd.Series(lr_ml.predict(Test['t']))
rmse_lr = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_lr))**2))
rmse_lr#591.55
#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad#475.56
##################### Exponential ##############################

Expo = smf.ols('log_sales~t',data=Train).fit()
pred_Expo = pd.Series(Expo.predict(pd.DataFrame(Test['t'])))
rmse_Expo = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Expo)))**2))
rmse_Expo#466.24

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q4+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q4','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea#1860.023

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q4+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q4','Q2','Q3','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #301.73

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Q4+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Q4','Q2','Q3']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea#1963.38

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Q4+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea #225.52

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_lr","rmse_Expo","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_lr,rmse_Expo,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#### so Multi additive seasonality has the least value among the models prepared so far,10.51 rmse.

# Predicting new values 

predict_data = pd.read_excel("E:/ADM/Excelr solutions/DS assignments/Forcasting/pred_cocaxlsx.xlsx")
model_full = smf.ols('log_sales~t+Q4+Q2+Q3',data=cola).fit()
pred_full=model_full.predict(cola)
cola["pred_full"]=pred_full
residuals=pd.DataFrame(np.array(cola["Sales"]-np.array(pred_full)))
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_sales"] = pred_new

#ACF plot for Residuals
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals.values.squeeze(), lags=4)
sm.graphics.tsa.plot_pacf(cola.Sales.values.squeeze(), lags=4)

#####

#AR models for Forcasting errors
from statsmodels.tsa.arima_model import ARIMA
pas=cocacola["Sales"]

model=ARIMA(residuals,order=(1,0,0)).fit(transparams=True)

forecasterrors=model.forecast(steps=4)[0]
predict_data["forecasted_sales"] = pd.Series(pred_new)
predict_data["forecasted_errors"] = pd.Series(forecasterrors)
predict_data["improved"] = predict_data["forecasted_sales"]+predict_data["forecasted_errors"]
predict_data["forecasted_sales"].plot()


