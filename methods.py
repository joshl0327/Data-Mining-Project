#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import fix_yahoo_finance as yf
import numpy as np
import pandas_datareader.data as web
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
import quandl
import datetime
from stocker import Stocker


def get_stocks(names, start, end):
    return yf.download(names,start, end)
    

def get_close(stocks):
    stocks_close = stocks['Adj Close']
    stocks_close.plot()
    plt.show()
    return stocks_close

def get_returns(stocks):
    stocks = stocks.apply(lambda x: x / x[0])
    stocks.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
    return stocks

def get_spy(location,start,end):
    spy = web.DataReader("SPY", location, start, end)
    return spy

def add_spy(spy, stocks):
    stocks = stocks.join(spy['Adj Close'])
    stocks.rename(columns={'Adj Close':'SPY'}, inplace=True)
    return stocks

def reset_plot():
    # Restore default parameters
    plt.rcParams.update(plt.rcParamsDefault)

    # Adjust a few parameters to liking
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['text.color'] = 'k'

def get_stocker(names):
    stocker_data = []
    for x in names:
        stock = Stocker(x)
        stocker_data.append(stock)
    return stocker_data

def plot_stocker(stocker_list, start, end):
    for x in stocker_list:
        x.plot_stock(start_date = start, end_date = end, stats = ['Daily Change', 'Adj. Volume'], plot_type='pct')
    return

def get_multiple_close(columns, stocks, start_date, end_date):
    stock_plot = stocks[0].make_df(start_date, end_date)
    stock_plot = stock_plot[['Date','Adj. Close']].rename(columns={'Adj. Close': columns[0]})
    stock_plot.set_index('Date', inplace=True, drop=False)
    stock_plot.iloc[0].head()
    
    #Adds in other stocks to df
    i = 1
    while (i<len(stocks)):
        stock_close = stocks[i].make_df(start_date, end_date)
        stock_close = stock_close[['Date','Adj. Close']].rename(columns={'Adj. Close': columns[i]})
        stock_close.set_index('Date', inplace=True)
        stock_plot = stock_plot.join(stock_close[columns[i]])
        i+=1
    return stock_plot

def dpc(stock_close):
    # Daily returns
    daily_pct_change = stock_close.pct_change()
    # Replace NA values with 0
    daily_pct_change.fillna(0, inplace=True)
    
    return daily_pct_change
    

def volatility(stock_plot):
    reset_plot()
    columns = list(stock_plot.columns.values)
    i = 1;

    while (i < len(columns)):
        plt.style.use('fivethirtyeight');
        daily_pct_change = dpc(stock_plot[columns[i]])
        width = 1
        if columns[i] == 'SPY':
            width = 3;
        # Define the minumum of periods to consider 
        min_periods = 75 

        # Calculate the volatility
        vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods) 
        
        plt.plot(stock_plot['Date'], vol, label = columns[i], linewidth = width, alpha = 0.8)
        plt.xlabel('Date'); plt.title('Stock Volatility'); 
        
        #vol.plot()
        plt.legend(prop={'size':10})
        plt.grid(color = 'k', alpha = 0.4);
        i+=1
    plt.show();
    

def plot_multiple(columns, stock_plot, plot_type='basic'):
    
    reset_plot()
    i = 0
    while (i < len(columns)):
        # Percentage y-axis        
        plt.style.use('fivethirtyeight');
        width = 1
        if columns[i] == 'SPY':
            width = 3;
        
        if plot_type == 'returns': 
            s= stock_plot.loc[:,columns[i]].first_valid_index()
            plt.plot(stock_plot['Date'], stock_plot.loc[:,columns[i]].transform(lambda x: x / x[s]), 
                     label = columns[i], linewidth = width,alpha = 0.8)          
            plt.xlabel('Date'); plt.ylabel('Returns'); plt.title('Stock History - Returns');
            plt.axhline(y = 1, color = "black", lw = 2);

        # Stat y-axis
        elif plot_type == 'basic':
            plt.plot(stock_plot['Date'], stock_plot.loc[:,columns[i]], label = columns[i], linewidth = width,alpha = 0.8)
            plt.xlabel('Date'); plt.ylabel('US $'); plt.title('Stock History - Stock Price'); 

        plt.legend(prop={'size':10})
        plt.grid(color = 'k', alpha = 0.4);
        i+=1
    plt.show();
       
def model_stocker(stocker_list):
    for x in stocker_list:
        model, model_data = x.create_prophet_model()
        model.plot_components(model_data)
        plt.show()
    return

def changepoint_stocker(stocker_list):
    for x in stocker_list:
        x.changepoint_date_analysis()

def candlestick(name,start,end):
    reset_plot()
    stock = quandl.get("WIKI/"+ name, start_date=start, end_date=end)
    stock["20d"] = np.round(stock["Adj. Close"].rolling(window = 20, center = False).mean(), 2)
    stock["50d"] = np.round(stock["Adj. Close"].rolling(window = 50, center = False).mean(), 2)
    stock["200d"] = np.round(stock["Adj. Close"].rolling(window = 200, center = False).mean(), 2) 
    pandas_candlestick_ohlc(stock.loc[start:end,:], otherseries = ["20d", "50d", "200d"], adj=True)

def pandas_candlestick_ohlc(dat, stick = "day", adj = False, otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param adj: A boolean indicating whether to use adjusted prices
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    fields = ["Open", "High", "Low", "Close"]
    if adj:
        fields = ["Adj. " + s for s in fields]
    transdat = dat.loc[:,fields]
    transdat.columns = pd.Index(["Open", "High", "Low", "Close"])
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
 
    plt.show()