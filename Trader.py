#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# In[4]:


class Trader(tpqoa.tpqoa):
    ''' A parent class that inherits from OANDA's API to create a trading object
    '''
    
    def __init__(self, conf_file, instrument, bar_length, units):
        '''
        Attributes:
        conf_file: text file
            Connects to the API
        instrument: str
            The financial instrument to be traded
        bar_length: int
            The time length we look at to trade (e.g 1 minute candlesticks)
        units: int
            The quantity of the instrument to be traded
        '''
        
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []

        #*****************add strategy-specific attributes here******************
     
        #************************************************************************
          
    def get_most_recent(self, days = 5):
        '''
        Method:
            Fetches the data from the past 5 days until now in the appropriate time intervals
        '''
        
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                break
                
    def on_success(self, time, bid, ask):
        '''
        Method:
            Starts the trading session and collects live data. One enough data has been collected it will check
            its strategy and execute trades accordingly
        '''
        
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick]) # Midpoint price
        self.tick_data = pd.concat([self.tick_data, df]) 
        
        if recent_tick - self.last_bar > self.bar_length: # Executes when we get new candlestick
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        '''
        Method:
            Concatenates the data and fills in missing values
        '''
        
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length, 
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): 
        '''
        Method:
            Where the trading strategy is defined
        '''
        
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************

        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        '''
        Method:
            Executes trades dependent on the position the strategy has given
        '''
        
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        '''
        Method:
            Reports the trade position, including time, units, price and p&l
        '''
        
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

class SMATrader(Trader):
    ''' Child class that uses a SMA trading strategy
    '''
    
    def __init__(self, conf_file, instrument, bar_length, SMA_S, SMA_L, units):
        ''' 
            Attributes:
            SMA_S: int
                Moving window for shorter SMA
            SMA_L: int
                Moving window for longer SMA
        '''
        
        super().__init__(conf_file, instrument, bar_length, units)
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L

    def define_strategy(self):
        '''
            Method:
            Defines the strategy for SMA
        '''
        
        df = self.raw_data.copy()
        df["SMA_S"] = df[self.instrument].rolling(self.SMA_S).mean() # Shorter moving average
        df["SMA_L"] = df[self.instrument].rolling(self.SMA_L).mean() # Longer moving average
        df["position"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1) # Goes long when SMA short is above SMA long and vice versa
        self.data = df.copy()

class MeanReversionTrader(Trader):
    ''' Child class that uses a Bollinger Bands strategy
    '''
    
    def __init__(self, conf_file, instrument, bar_length, SMA, dev, units):
        '''
            Attributes:
            SMA: int
                Moving window for the average
            dev: float
                How many standard deviations we use for the Bollinger Bands
        '''
        
        super().__init__(conf_file, instrument, bar_length, units)
        self.SMA = SMA
        self.dev = dev

    def define_strategy(self):
        '''
            Method:
            Defines strategy for the Bollinger Bands
        '''
        
        df = self.raw_data.copy()
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean() # Moving average
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev # Lower band
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev # Upper band
        df["distance"] = df[self.instrument] - df.SMA # Distance from the moving average and price
        df["position"] = np.where(df[self.instrument] < df.Lower, 1, np.nan) # Go long when below lower band
        df["position"] = np.where(df[self.instrument] > df.Upper, -1, df["position"]) # Go short when above upper band
        df["position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position"]) # Go neutral once returned close to average
        df["position"] = df.position.ffill().fillna(0)
        self.data = df.copy()

class ClassificationTrader(Trader):
    ''' Child class that uses logistic regression as a trading strategy
    '''
    
    def __init__(self, conf_file, instrument, bar_length, lags, units):
        '''
            Attributes:
            lags: int
                The number of price shifts we use for the features in the model
        '''
        
        super().__init__(conf_file, instrument, bar_length, units)
        self.lags = lags

    def get_most_recent(self, days = 5):
        '''
            Method:
            Trains the model on the most recent 5 days
        '''
        
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                self.define_model()
                break

    def define_model(self):
        '''
            Method:
            Trains the logistic regression model. The daily log returns (prevents underflow) 
            and its lags are the independent variable and the direction of the returns is the dependent variable
        '''
        
        df = self.raw_data.copy()
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        df.dropna(inplace=True)
        df["direction"] = np.sign(df["returns"])
        lags_range = range(1, self.lags + 1)
        lag_cols = {f"lag{lag}": df["returns"].shift(lag) for lag in lags_range}
        df = df.assign(**lag_cols)
        df.dropna(inplace=True)
        cols = [f"lag{lag}" for lag in lags_range]
        self.lm = LogisticRegression(C=1e6, max_iter=100000, multi_class="ovr")
        self.lm.fit(df[cols], df["direction"])
        

    def define_strategy(self):
        '''
            Method:
            Predicts the direction of the tick using the model
        '''
        
        df = self.raw_data.copy()
        df = pd.concat([df, self.tick_data])
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        df.dropna(inplace=True)
        df["direction"] = np.sign(df["returns"])
        lags_range = range(1, self.lags + 1)
        lag_cols = {f"lag{lag}": df["returns"].shift(lag) for lag in lags_range}
        df = df.assign(**lag_cols)
        df.dropna(inplace=True)
        cols = [f"lag{lag}" for lag in lags_range]
        df["position"] = self.lm.predict(df[cols])
        self.data = df.copy()