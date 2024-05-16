#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("AirPassengers.csv")


# In[4]:


data


# In[5]:


#converting the data to datetime  object
data["Month"] = pd.to_datetime(data["Month"])


# In[6]:


data


# In[8]:


#changing the column to index
data = data.set_index(["Month"])


# In[9]:


data


# In[10]:


data.shape


# In[11]:


# Plotting the Time Series

data.plot()


# In[12]:


#lets check if the data is stationary data or not


# In[13]:


# to find the stationarity in data, we will use a statistical test i.e. adfuller test
# Augmented Dickey Fuller test


# In[15]:


# here we will take two hypothesis

# 1) null hypothesis
# 2) Alternate Hypothesis


# In[16]:


# To reject the null hypotheses, the following
# must be true:

# 1. If the p-value after the adfuller test is
# greater than 0.05, we fail to reject the
# hypotheses.

# 2. If the p-value is less than 0.05, we can
# reject the null hypotheses and assume
# that the time series is stationary.


# h0- data is not stationary
# h1 - data is stationary


# In[17]:


from statsmodels.tsa.stattools import adfuller


# In[19]:


result = adfuller(data["#Passengers"])
print(result)

if(result[1]>0.05):
  print("The data is not stationary")

else:
  print("The data is stationary")


# In[20]:


#seasonal decompose : It plots the components of the time series data

from statsmodels.tsa.seasonal import seasonal_decompose


# In[21]:


decomposition = seasonal_decompose(data["#Passengers"])
decomposition.plot()


# ROLLING STATISTICS:
# 
# Rolling statistics is a very useful operation for time series data. Rolling mean creats a rolling window with a specified size and perform calculation on the data in this window which ofcourse rolls through the data.
# 
# Rolling stats also helps us to plot the mean and the standard deviation

# In[22]:


mean_log = data.rolling(window=12).mean()
std_log = data.rolling(window=12).std()


# In[23]:


plt.plot(data, color="blue",label="Original")
plt.plot(mean_log,color="red",label="Rolling Mean")
plt.plot(std_log,color="black",label="Rolling std")
plt.legend(loc="best")
plt.title("Rolling Mean and Rolling std")
plt.show


# In[24]:


#transformation

first_log = np.log(data)
first_log = first_log.dropna()


# In[25]:


mean_log = first_log.rolling(window=12).mean()
std_log = first_log.rolling(window=12).std()


# In[27]:


plt.plot(first_log, color="blue",label="Original")
plt.plot(mean_log,color="red",label="Rolling Mean")
plt.plot(std_log,color="black",label="Rolling std")
plt.legend(loc="best")
plt.title("Rolling Mean and Rolling std")
plt.show


# In[29]:


new_data = first_log - mean_log
new_data = new_data.dropna()


# In[30]:


mean_log = new_data.rolling(window=12).mean()
std_log = new_data.rolling(window=12).std()

plt.plot(new_data,color='blue',label='Original')
plt.plot(mean_log,color='red',label='Rolling Mean')
plt.plot(std_log,color='black',label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean and Rolling Std')
plt.show()


# In[32]:


from statsmodels.tsa.stattools import adfuller


# In[36]:


result = adfuller(new_data)
result


if(result[1]>0.05):
    print("The series is not stationary")
else:
    print("series is stationary")


# In[37]:


plt.plot(new_data)
plt.show()


# In[38]:


# Autocorrelation

from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(new_data.dropna()) #=--------> value of q


# In[40]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(new_data.dropna())  #---------> value of p


# In[41]:


train = new_data.iloc[:120]
test = new_data[121:]


# In[42]:


train


# In[43]:


test


# In[46]:


len(train)


# In[47]:


len(test)


# In[44]:


from statsmodels.tsa.arima.model import ARIMA


# In[45]:


model = ARIMA(train,order=(2,1,2))
model_fit = model.fit()


# In[48]:


new_data['predict']=model_fit.predict(start=len(train),end=len(train)+len(test)-1, dynamic=True)
new_data[['#Passengers','predict']].plot()


# In[49]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[50]:


model=SARIMAX(train,order=(2,1,2),seasonal_order=(2,1,2,12))
model=model.fit()


# In[51]:


new_data['predict']=model.predict(start=len(train), end=len(train)+len(test)-1,dynamic=True)
new_data[['#Passengers','predict']].plot()


# In[62]:


forecast = model.forecast(steps=60) #for predicting future prospect
new_data.plot()
forecast.plot()


# In[52]:


import itertools

p = range(0,8)
q = range(0,8)
d = range(0,2)

pdq_combination = list(itertools.product(p,d,q))
pdq_combination


print(len(pdq_combination)) #total combinations of p,d,q

from sklearn.metrics import *
rmse = []
order1 = []

for pdq in pdq_combination:
        #model = ARIMA(train, order=pdq).fit()
        model = ARIMA(train,order=pdq)
        model_fit=model.fit()
        pred = model_fit.predict(start=len(train),end=len(train)+len(test)-1, dynamic=True)
        error = np.sqrt(mean_squared_error(test, pred))
        order1.append(pdq)
        rmse.append(error)


results = pd.DataFrame(index=order1, data=rmse, columns=['RMSE'])

results


# In[53]:


results.sort_values("RMSE") #Least RMSE score are best hyperparamterer for (p,d,q) values


# In[54]:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train,order=(5,1,7))
model_fit = model.fit()


new_data["predict"] = model_fit.predict(start=len(train),end=len(train)+len(test)-1, dynamic=True)


# In[55]:


new_data[['#Passengers','predict']].plot()


# In[56]:


model=SARIMAX(train,order=(5,1,7),seasonal_order=(5,1,7,12))
model=model.fit()


# In[57]:


new_data['predict']=model.predict(start=len(train), end=len(train)+len(test)-1,dynamic=True)
new_data[['#Passengers','predict']].plot()


# In[59]:


# forecasting for next 5 years
forecast = model.forecast(steps = 60)
new_data.plot()
forecast.plot()


# In[60]:


# forecasting for next 10 years
forecast = model.forecast(steps = 120)
new_data.plot()
forecast.plot()


# In[ ]:




