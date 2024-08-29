## TASK 1- data collection
import yfinance as yf

# creating dataset for Apple (stocks for last 5 years)
data = yf.download('AAPL', start='2019-07-4', end='2024-07-04')
data

# look at first few rows and last few rows to get general idea of change
print(data.head())
print(data.tail())

data.to_csv(f'{'AAPL'}_historical_data.csv')

## TASK 2- cleaning and preparation

# clean data using pandas
import pandas as pd

# might be pointless
data = pd.read_csv('AAPL_historical_data.csv')

# use isnull function from pandas to check for any missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# output shows no missing values

# check columns for correct data type
data_types = data.dtypes
print("Data types:\n", data_types)

# date = object which is appropriate
# open, high, close, low, adj close = float64 which is appropriate as it means numerical with decimal precision
# volume = int64 which is appropriate as uses integers

# check for outliers beginning with close
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot(data['Close'])


plt.title('Boxplot of Close')
plt.ylabel('Close')
plt.show()

# no outliers (values beyond whiskers)

# Repeat for each label
plt.figure(figsize=(8, 6))
plt.boxplot(data['Open'])
plt.title('Boxplot of Open')
plt.ylabel('Open')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(data['Low'])
plt.title('Boxplot of Low')
plt.ylabel('Low')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(data['High'])
plt.title('Boxplot of High')
plt.ylabel('High')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(data['Adj Close'])
plt.title('Boxplot of Adj Close')
plt.ylabel('Adj Close')
plt.show()

volume_data = data['Volume']
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(volume_data, vert=False)
ax.set_title(f'Boxplot of Volume for {'AAPL'}')
ax.set_xlabel('Volume')

plt.show()
# no outliers in all EXCEPT VOLUME

# setting date as the index

# PART 2~ Ensure the Date column is properly formatted as a datetime object and set it as the index of the DataFrame.
# loading dataset into dataframe
df = pd.read_csv('AAPL_historical_data.csv')

# convert 'date' column to datetime format
# this is essential so that python recognises the strings as dates rather than text
# therefore we can compare dates to calculate time differences etc
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# check its now in datetime format
print(df.index)
print(df.head())


## TASK 3
# calculate and plot the daily returns of the stock
df['Daily_Return'] = df['Close'].pct_change() * 100
df['Daily_Return']

# we use 'close' because it reflects the trader's opinions at the end of the trading day
# and is the most recent price from that day
# we can see that the stock increases in price or falls in price relative to previous day
# e.g 2019-07-11 is -0.73 (2 d.p) indicating a fall

# plot the daily returns
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Daily_Return'], color='blue', linestyle='-', linewidth=1)
plt.title('Daily Returns of AAPL Stock')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# this line graph exemplifies continuous fluctuations in the stock price of apple


# calculate 30 day moving average of the closing price
# moving averages smooth out short-term fluctuations in the data to reveal underlying trends
# providing a clearer image of the overall price trend over time as daily returns are very volatile
df['30_Day_MA'] = df['Close'].rolling(window=30).mean()

# plot 30 day moving average
plt.figure(figsize=(12, 6))
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='AAPL Close Price', color='blue')
plt.plot(df.index, df['30_Day_MA'], label='30-Day Moving Average', color='red', linestyle='--')

plt.title('AAPL Closing Price and 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# the red line is much smoother and shows a general increase with periods of decline
# we can see it shows general trend as blue line shows many declines but red line may not as in general there was a rise despite anomalies

# calculate 90 day moving average
df['90_Day_MA'] = df['Close'].rolling(window=90).mean()

# plot 90 day MA
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='AAPL Close Price', color='blue')
plt.plot(df.index, df['90_Day_MA'], label='90-Day Moving Average', color='green', linestyle='--')

plt.title('AAPL Closing Price and 90-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# even smoother and provides clearer image of the long term trend

## creating a graph which annotates the highest and lowest closing prices in the dataset.
# first identify the highest and lowest closing prices
max_close_price = df['Close'].max()
min_close_price = df['Close'].min()

# plotting closing prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='AAPL Closing Prices', color='blue')

# annotate highest closing price
plt.scatter(df.index[df['Close'] == max_close_price], max_close_price, color='red', label=f'Highest Close: {max_close_price}', zorder=5)
plt.annotate(f'Highest Close: {max_close_price}', xy=(df.index[df['Close'] == max_close_price], max_close_price),
             xytext=(-100, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

# annotate lowest closing price
plt.scatter(df.index[df['Close'] == min_close_price], min_close_price, color='green', label=f'Lowest Close: {min_close_price}', zorder=5)
plt.annotate(f'Lowest Close: {min_close_price}', xy=(df.index[df['Close'] == min_close_price], min_close_price),
             xytext=(-100, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)

# present the plot
plt.title('AAPL Closing Prices with Highest and Lowest Close Annotations')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# lowest close = 48.33...
# highest close = 221.55...


## TASK 4- Data Visualisation
# 1. Create a subplot showing:
# - The closing price with the moving averages.
# - The daily volume of the stock traded.

# a subplot refers to creating multiple plots in a single plot space
# good for visualising side by side

# creating the subplot for the closing price with moving averages and daily volume
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# firstly plotting the closing price and moving averages
ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
ax1.plot(df.index, df['30_Day_MA'], label='30-Day MA', color='red')
ax1.plot(df.index, df['90_Day_MA'], label='90-Day MA', color='green')
ax1.set_title('Apple Stock Closing Price and Moving Averages')
ax1.set_ylabel('Price')
ax1.legend()

# plotting the daily volume
ax2.bar(df.index, df['Volume'], color='gray')
ax2.set_title('Apple Stock Daily Trading Volume')
ax2.set_ylabel('Volume')
ax2.set_xlabel('Date')

# annotating the highest and lowest close price
max_close = df['Close'].max()
min_close = df['Close'].min()
ax1.annotate(f'Highest Close: ${max_close:.2f}', xy=(df.index[-1], max_close), xytext=(-100, 20),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))
ax1.annotate(f'Lowest Close: ${min_close:.2f}', xy=(df.index[0], min_close), xytext=(-100, -30),
             textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))

# annotate highest and lowest volume?
max_volume_date = data['Volume'].idxmax()
max_volume = data.loc[max_volume_date, 'Volume']

min_volume_date = data['Volume'].idxmin()
min_volume = data.loc[min_volume_date, 'Volume']

ax2.annotate(f'Highest Volume: {max_volume:.0f}', xy=(max_volume_date, max_volume),
             xytext=(-100, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))

ax2.annotate(f'Lowest Volume: {min_volume:.0f}', xy=(min_volume_date, min_volume),
             xytext=(-100, -30), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'))


plt.tight_layout()
plt.show()


## TASK 5- Advanced Analysis (Market Indicators)
# - Calculate and plot the Relative Strength Index (RSI) with a 14-day period.
# RSI is a number between 0 and 100, used by traders to evaluate how much a stock has gone up or down recently

# calculate RSI

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Assuming 'data' is your DataFrame containing necessary columns like 'Close'
data['RSI'] = calculate_rsi(data)

# plotting
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['RSI'], label='RSI', color='purple', linewidth=1.5)
plt.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
plt.title('Relative Strength Index (RSI) - Apple Inc.', fontsize=16, fontweight='bold')
plt.ylabel('RSI', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.show()


# - Calculate and plot the Moving Average Convergence Divergence (MACD) along with its signal line.
# calculate the MACD and signal line
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    exp1 = data['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

macd, signal = calculate_macd(data)

# plot
plt.figure(figsize=(14, 7))
plt.plot(data.index, macd, label='MACD', color='blue')
plt.plot(data.index, signal, label='Signal Line', color='red')
plt.title('Moving Average Convergence Divergence (MACD) - Apple Inc.', fontsize=16, fontweight='bold')
plt.ylabel('MACD', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Ensure tight layout
plt.show()


# - Calculate and plot Bollinger Bands with a 20-day moving average and 2 standard deviations.
# Calculate the 20-day moving average and standard deviation
data['20_Day_MA'] = data['Close'].rolling(window=20).mean()
data['Upper_Band'] = data['20_Day_MA'] + 2 * data['Close'].rolling(window=20).std()
data['Lower_Band'] = data['20_Day_MA'] - 2 * data['Close'].rolling(window=20).std()

# Plotting Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['20_Day_MA'], label='20-Day MA', color='red')

plt.fill_between(data.index, data['Upper_Band'], data['Lower_Band'], color='gray', alpha=0.2, label='Bollinger Bands')
plt.title('Bollinger Bands (20-Day MA, 2 Std Dev) - Apple Inc.', fontsize=16, fontweight='bold')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Ensure tight layout
plt.show()



## TRADING SIGNALS
#- Generate buy/sell signals based on the RSI (e.g., RSI < 30: buy, RSI > 70: sell).
#- Generate buy/sell signals based on MACD crossovers (e.g., MACD crossing above the signal line: buy, MACD crossing below the signal line: sell).
import numpy as np

# using RSI from previous task
data['RSI Signal'] = np.where(data['RSI'] < 30, 'Buy', np.where(data['RSI'] > 70, 'Sell', 'Hold'))

print(data[['RSI', 'RSI Signal']].head())

plt.figure(figsize=(14, 7))
plt.plot(data.index, data['RSI'], label='RSI', color='orange')
plt.axhline(30, linestyle='--', alpha=0.5, color='red')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.scatter(data[data['RSI Signal'] == 'Buy'].index, data[data['RSI Signal'] == 'Buy']['RSI'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(data[data['RSI Signal'] == 'Sell'].index, data[data['RSI Signal'] == 'Sell']['RSI'], marker='v', color='red', label='Sell Signal', alpha=1)
plt.title('Relative Strength Index (RSI) with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

#- Generate buy/sell signals based on MACD crossovers (e.g., MACD crossing above the signal line: buy, MACD crossing below the signal line: sell).
# Calculate MACD and signal line
data['MACD'], data['Signal Line'] = calculate_macd(data)

# Plot MACD and signal line
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['MACD'], label='MACD')
plt.plot(data.index, data['Signal Line'], label='Signal Line', linestyle='--')
plt.title('MACD with Signal Line')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# - Generate buy/sell signals based on Bollinger Bands (e.g., price crossing below the lower band: buy, price crossing above the upper band: sell).
# Initialize signal column
data['Signal'] = 0

# Generate signals
# Buy signal: Price crosses below Lower Band
data.loc[data['Close'] < data['Lower_Band'], 'Signal'] = 1

# Sell signal: Price crosses above Upper Band
data.loc[data['Close'] > data['Upper_Band'], 'Signal'] = -1

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
plt.plot(data['Date'], data['20_Day_MA'], label='20-Day MA', color='red')
plt.fill_between(data['Date'], data['Upper_Band'], data['Lower_Band'], color='green', alpha=0.2, label='Bollinger Bands')

# Plot buy signals
plt.plot(data[data['Signal'] == 1]['Date'], data[data['Signal'] == 1]['Close'], '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(data[data['Signal'] == -1]['Date'], data[data['Signal'] == -1]['Close'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Bollinger Bands (20-Day MA, 2 Std Dev)')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Displaying signals
print(data[data['Signal'] != 0][['Date', 'Close', 'Signal']])



### TASK 6
# implement an additional advanced analysis

# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Calculate rolling annualized volatility (252 trading days)
rolling_volatility = data['Daily Return'].rolling(window=252).std() * np.sqrt(252)

# Plotting the volatility
plt.figure(figsize=(12, 6))
plt.plot(rolling_volatility, color='blue')
plt.title('Rolling Annualized Volatility of AAPL')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()
