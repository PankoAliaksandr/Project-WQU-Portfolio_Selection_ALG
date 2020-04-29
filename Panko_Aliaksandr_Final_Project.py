import pandas as pd
import datetime
import numpy as np
from pandas_datareader import data as pdr
from matplotlib import pyplot
from scipy import stats

# Get S&P 500 companies' tickers
data = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
table = data[0]
tickers = table[0]
tickers = tickers.drop(tickers.index[0])

# Determine Start and End dates
end_date = datetime.date.today()
start_date_real = datetime.date(end_date.year-10,
                                        end_date.month,
                                        end_date.day)
# Adjust to avoid NAs
start_date = start_date_real - datetime.timedelta(days=40)
stock_data = pdr.get_data_yahoo(tickers[1], start_date, end_date)

# Create initial data frame to store all stocks' data
df_sp500 = pd.DataFrame(index=stock_data.index, columns=tickers)

# Download S&P 500 data
for ticker in tickers:
    # To let you know that the process is going
    print ticker
    if ticker == 'BRK.B':
        stock_symbol = 'BRK-B'
    elif ticker == 'BF.B':
        stock_symbol = 'BF-B'
    else:
        stock_symbol = ticker
    stock_data = pdr.get_data_yahoo(stock_symbol, start_date, end_date)
    df_sp500[ticker] = stock_data['Adj Close']

# Delete columns with NAN
columns_names = df_sp500.columns
for column in columns_names:
    if np.isnan(df_sp500[column]).any():
        df_sp500.drop(column, axis=1, inplace=True)

# Get DJIA companies' tickers
data = pd.read_html(
        'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
table = data[1]
tickers = table[2]
tickers = tickers.drop(tickers.index[0])

# Create data frame for DJIA data
df_DJIA = pd.DataFrame(index=stock_data.index, columns=tickers)

# Download DJIA data
for ticker in tickers:
    # To let you know that the process is going
    print ticker
    stock_data = pdr.get_data_yahoo(ticker, start_date, end_date)
    df_DJIA[ticker] = stock_data['Adj Close']

start_date_time_real = pd.to_datetime(start_date_real)
ma_window = len(df_sp500.index[df_sp500.index <= start_date_time_real])

# Calculate MA for S&P 500 stocks
df_sp500_MA = pd.DataFrame(index=df_sp500.index, columns=df_sp500.columns)

for column in df_sp500.columns:
    df_sp500_MA[column] = pd.rolling_mean(df_sp500[column], ma_window)

# Delete NAs
df_sp500_MA = df_sp500_MA.iloc[ma_window - 1:]

# Calculate MA for DJIA stocks
df_DJIA_MA = pd.DataFrame(index=df_DJIA.index, columns=df_DJIA.columns)

for column in df_DJIA.columns:
    df_DJIA_MA[column] = pd.rolling_mean(df_DJIA[column], ma_window)

# Delete NAs
df_DJIA_MA = df_DJIA_MA.iloc[ma_window - 1:]

# Trend analysis for S&P 500 stocks
period_df_sp500 = pd.DataFrame(index=['Start Period Price',
                                      'End Period Price',
                                      'Slope'], columns=df_sp500_MA.columns)

# Trend analysis for DJIA stocks
period_df_DJIA = pd.DataFrame(index=['Start Period Price',
                                     'End Period Price',
                                     'Slope'], columns=df_DJIA_MA.columns)


all_period = df_sp500_MA.index
previous_period_end_date = all_period[0]

# Create the list to store portfolio returns in different years
portfolio_returns = [0] * 9

# Loop for every year
for period_number in range(1, 10):
    # Period for analysis
    period_start_date = previous_period_end_date
    period_start_date_time = pd.to_datetime(period_start_date)
    period_end_date = datetime.date(period_start_date.year + 1,
                                    period_start_date.month,
                                    period_start_date.day)
    period_end_date_time = pd.to_datetime(period_end_date)

    # Period for strategy implementation
    new_period_start_date_time = period_end_date_time
    new_period_end_date = datetime.date(period_start_date.year + 2,
                                        period_start_date.month,
                                        period_start_date.day)
    new_period_end_date_time = pd.to_datetime(new_period_end_date)

    # Time window for analysis period
    x = all_period[np.logical_and(
            all_period >= period_start_date_time,
            all_period < period_end_date_time)]

    # Time window for implementation period
    x_new = all_period[np.logical_and(
            all_period >= new_period_start_date_time,
            all_period < new_period_end_date_time)]

    # Make trend analysis for S&P 500 stocks
    for column in df_sp500_MA.columns:
        y = df_sp500_MA.loc[x][column]
        y_new = df_sp500_MA.loc[x_new][column]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(1, len(y)+1), y)

        period_df_sp500[column]['Start Period Price'] = y_new[0]
        period_df_sp500[column]['End Period Price'] = y_new[len(
                y_new) - 1]
        period_df_sp500[column]['Slope'] = slope

    # Make trend analysis for DJIA stocks
    for column in df_DJIA_MA.columns:
        y = df_DJIA_MA.loc[x][column]
        y_new = df_DJIA_MA.loc[x_new][column]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(1, len(y)+1), y)
        period_df_DJIA[column]['Start Period Price'] = y_new[0]
        period_df_DJIA[column]['End Period Price'] = y_new[len(
                y_new) - 1]
        period_df_DJIA[column]['Slope'] = slope

    # Find 15 stocks from S&P 500 with the largest absolute value of slope
    slopes = period_df_sp500.iloc[[2]].values
    abs_slopes = abs(slopes)[0]
    largest_15_abs_slopes_index = abs_slopes.argsort()[-15:][::-1]

    tickers = period_df_sp500.columns
    selected_stocks = dict()
    for i in largest_15_abs_slopes_index:
        selected_stocks[tickers[i]] = 1

    # Find 5 stocks from DJIA with the largest absolute value of slope
    slopes = period_df_DJIA.iloc[[2]].values
    abs_slopes = abs(slopes)[0]
    largest_5_abs_slopes_index = abs_slopes.argsort()[-5:][::-1]

    tickers = period_df_DJIA.columns
    for i in largest_5_abs_slopes_index:
        if tickers[i] in selected_stocks.keys():
            selected_stocks[tickers[i]] = selected_stocks[tickers[i]] + 1
        else:
            selected_stocks[tickers[i]] = 1

    # Calculate returns
    for stock in selected_stocks.keys():
        if stock in period_df_sp500.columns:
            new_df_sp500 = period_df_sp500
        else:
            new_df_sp500 = period_df_DJIA

        start_period_price = new_df_sp500.iloc[[0]][stock].values[0]
        end_period_price = new_df_sp500.iloc[[1]][stock].values[0]
        slope = new_df_sp500.iloc[[2]][stock].values[0]
        stock_weight = 0.05 * selected_stocks[stock]

        if slope >= 0:
            # Go long
            portfolio_returns[period_number - 1] = portfolio_returns[
                    period_number - 1] + stock_weight*(
                    end_period_price - start_period_price) / start_period_price
        else:
            # Go short
            portfolio_returns[period_number - 1] = portfolio_returns[
                    period_number - 1] + stock_weight*(
                    start_period_price - end_period_price) / start_period_price
    previous_period_end_date = period_end_date

# Calculate cumulative returns
cumulative_returns = [0]*10
cumulative_returns[0] = 1
returns_product = 1
for i in range(0, len(portfolio_returns)):
    returns_product = returns_product * (1 + portfolio_returns[i])
    cumulative_returns[i+1] = returns_product

# Plot cumulative returns
pyplot.title("Cumulative Returns")
pyplot.xlabel("Period")
pyplot.plot(range(1, 11), cumulative_returns)
pyplot.show()

# Calculate CAGR
cagr = (cumulative_returns[9] / cumulative_returns[0])**(1 / 9.0) - 1
print cagr
