#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sharedtools.Portfolio import Portfolio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

def visualize_trade(portfolio_directory):
    #trades
    a=Portfolio(initial_capital=1000000, commission_rate=17.5, portfolio_name='Technical')
    a.load(portfolio_directory)
    nav=a.historical[['Datetime','NAV']].copy()[1:]
    nav.Datetime=pd.to_datetime(nav.Datetime)
    trades=a.trades[['Datetime','Quantity','Price']].copy()
    trades.Datetime=pd.to_datetime(trades.Datetime)
    sell_date=trades[trades.Quantity<0].Datetime
    sell_price=trades[trades.Quantity<0].Price
    buy_date=trades[trades.Quantity>0].Datetime
    buy_price=trades[trades.Quantity>0].Price

    data=pd.read_pickle('../../../Data/Static/CoinbaseProBTCUSD/alldays_from_2015-06-01 00:00:00_timeReadable.pickle')
    data=data.rename(columns={'c':'Close'})#
    alltimes=list(data.timeRead)
    data.timeRead=pd.to_datetime(data.timeRead)

    #setup
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig = plt.figure(figsize=(15,10))
    color1 = plt.cm.tab20(0)#blue
    color2 = plt.cm.tab20(4)#green
    color3 = plt.cm.tab20(6)#red
    color4 = plt.cm.tab20(3)#orange
    color5 = plt.cm.tab20(1)#light blue
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8)
    
    #trades
    gap=np.mean(data.Close)/5
    plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    plt.plot(data.timeRead, data.Close, color=color4,linewidth=1.5)
    plt.plot_date(sell_date,sell_price+gap, color=color3, markersize=4, marker='v')
    plt.plot_date(buy_date,buy_price-gap, color=color2, markersize=4, marker='^')
    plt.ylabel("BTC")
    plt.title(f'Performance of {portfolio_directory[3:]}')
    
    #position
    plt.subplot2grid((4, 1), (2, 0), rowspan=1)
    plt.plot(trades.Datetime,trades.Quantity.cumsum(),color=color5,drawstyle='steps-post')
    plt.ylabel('Pos in BTC')

    #NAV
    plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    plt.plot(nav.Datetime,nav.NAV,color=color1)
    plt.xlabel("Date")
    plt.ylabel("NAV")
    
    plt.show()
    print (a.statistics)
    
    

