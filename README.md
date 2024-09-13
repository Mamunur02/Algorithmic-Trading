This is an algorithmic trading project developed on Python. 
It uses the OANDA API to fetch real time prices for financial instruments like EUR/USD and executes orders dependent on the trading strategy used. 
The strategies that are currently implement is using SMA, Bollinger Bands and Classification.

'Trader.py' is the script that creates the trading object class.
It has child classes that inherit from the parent class and differ by the strategy defined within it.

'DNNModel.py' is the script that trains a DNN model on historical data for EUR/USD from 2017 to 2019.
It then tests the model on a test set and comparest it to a simple buy and hold strategy.
