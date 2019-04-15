#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# To visualize trades

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
    
    
# surface plot
def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def surface_plot(strategy_name,data_folder_directory,X_name,Y_name,Z_name,Z_type):

    #setup
    plt.rc('font', family='TT Commons')
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['legend.fontsize']= 15
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.titleweight'] = "bold"
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    sbcolA = "#01C38D"
    sbcolC = "#EDEFF1"
    sbcolB = "#191E29"
    warnings.filterwarnings("ignore")
    show_axis = False


    #data
    X=[]
    Y=[]
    Z=[]
    pickles = list_files(data_folder_directory,'pkl')
    for p in pickles:
        filename=f'{data_folder_directory}/{p}'
        a=Portfolio(initial_capital=1000000, commission_rate=0, portfolio_name='SurfacePlot')
        a.load(filename[:-4])
        X.append(a.others[X_name])
        Y.append(a.others[Y_name])
        Z.append(a.statistics[Z_type][Z_name])

    df = pd.DataFrame({X_name:X,Y_name:Y,Z_name:Z})

    #plot
#     plt.close("all")
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(df[X_name], df[Y_name], df[Z_name], cmap=plt.cm.Greens, linewidth=0.2)
    ax.set_xlabel(X_name)
    ax.set_ylabel(Y_name)
    ax.set_zlabel(Z_name)
    ax.set_title(strategy_name)
    plt.show()

