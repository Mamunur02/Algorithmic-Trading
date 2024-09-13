#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Importing libraries
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight


# In[53]:


# A simple DNN model using binary classification
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[54]:


# Data from 2017 to 2019 about EUR/USD
# Will update later to retrieve more recent annual data using OANDA
data = pd.read_csv(r"C:\Users\mamun\Downloads\Part5_Materials\DNN_data.csv", parse_dates=["time"], index_col="time")


# In[ ]:


# Prepping data
symbol = data.columns[0]
data["returns"] = np.log(data[symbol]/data[symbol].shift())
window = 50
df = data.copy()
df["dir"] = np.where(df["returns"] > 0, 1, 0)
df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(150).mean()
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std()
df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
df["mom"] = df["returns"].rolling(3).mean()
df["vol"] = df["returns"].rolling(window).std()
df.dropna(inplace = True)
lags = 5
cols = []
features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
for f in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(f, lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
df.dropna(inplace = True)


# In[56]:


# Splitting into training and test sets
X = df[cols]
y = df["dir"]
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]


# In[57]:


# Standardising the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[58]:


# Fitting the model and giving more importance to minority classes
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
model = create_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weights_dict, validation_data=(X_test, y_test))


# In[59]:


# Evaluating the model
model.evaluate(X_train, y_train)


# In[60]:


# Showing predictions for the training set
pred = model.predict(X_train)
plt.hist(pred, bins = 50)
plt.show()


# In[61]:


# Showing predictions for the test set
model.evaluate(X_test, y_test)
pred_test = model.predict(X_test)
plt.hist(pred_test, bins = 50)
plt.show()


# In[62]:


# Predicting the positions for the test set using the model
# Possible positions are -1, 0, 1
X_test_df = pd.DataFrame(X_test, columns = cols)
X_test_df["proba"] = model.predict(X_test)
X_test_df["position"] = np.where(X_test_df.proba < 0.47, -1, np.nan)
X_test_df["position"] = np.where(X_test_df.proba > 0.53, 1, X_test_df.position) 
X_test_df["position"] = X_test_df.position.ffill().fillna(0) 
X_test_df.position.value_counts(dropna = False)


# In[63]:


# Defining and testing the strategy against a buy and hold strategy
X_test_df = X_test_df.reset_index(drop=True) 
returns_series = df[split_point:]["returns"].reset_index(drop=True) 
X_test_df["strategy"] = X_test_df["position"] * returns_series
X_test_df["creturns"] = returns_series.cumsum().apply(np.exp)
X_test_df["cstrategy"] = X_test_df["strategy"].cumsum().apply(np.exp)
X_test_df[["creturns", "cstrategy"]].plot(figsize = (12, 8))
plt.show()


# In[65]:


# Accounting for transaction costs
# May modify so the strategy is executed fewer times
ptc = 0.000059
X_test_df["trades"] = X_test_df.position.diff().abs()
X_test_df["strategy_net"] = X_test_df.strategy - X_test_df.trades * ptc
X_test_df["cstrategy_net"] = X_test_df["strategy_net"].cumsum().apply(np.exp)
X_test_df[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12, 8))
plt.show()

