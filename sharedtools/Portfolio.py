import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from sharedtools.PerformanceHTML import PerformanceReport

class Portfolio(object):
    def __init__(self, initial_capital, commission_rate, portfolio_name, benchmark=None, others=None):
        
        '''
        Portfolio is a generic object for varied portfolio simulations.
        The object tracks and stores all trades and historical details.
        
        Output dataframes:
        [trades] all historical trades
        [portfolio] current open positions
        [tradesum] all trades that have been closed and consolidated
        [historical_portfolio] historical open positions for all datetime
        [historical] historical portfolio aggregated information for all datetime
        
        Output dictionarys:
        [statistics] performance statistics
        [others] other miscellaneous data
        
        
        :param initial_capital: initial capital  in $ to trade
        :type initial_capital: float
        :param commission_rate: commission rate in basis points
        :type commission_rate: float
        :param portfolio_name: name of portfolio
        :type portfolio_name: str
        :param benchmark: a dataframe consisting benchmark time series. \
                        Dataframe columns should be set to 'Datetime' and 'Benchmark'.\
                            Default is None.
        :type benchmark: a list, or an array
        :param others: other miscellaneous data
        :type portfolio_name: dict
    
        '''
        
        #parameters
        self.initial_capital = initial_capital
        self.nav = initial_capital
        self.cash = initial_capital
        self.portfolio_name=portfolio_name
        self.commission_rate = commission_rate
        self.benchmark=benchmark
        self.others=others
        
        #dataframes
        self.trades=pd.DataFrame(columns=['Datetime','Ticker','Quantity','Price','Commission_Expense','Margin','Direction'])
        self.portfolio=pd.DataFrame(columns=['Open_Date','Ticker','Quantity','Last_Price','Position_Value',
                                'Avg_Open_Price','Outlay','Realized_Pnl','Unrealized_Pnl','Other'])
        self.tradesum=pd.DataFrame(columns=['Open_Date','Ticker','Avg_Open_Price','Outlay',
                                            'Realized_Pnl','Close_Date','Other'])
        self.historical_portfolio=pd.DataFrame(columns=['Open_Date','Ticker','Quantity','Last_Price','Position_Value',
                                'Avg_Open_Price','Outlay','Realized_Pnl','Unrealized_Pnl','Other'])
        self.historical=pd.DataFrame(columns=['Datetime','Position_Value','Cash','Realized_Pnl','Unrealized_Pnl',
                                              'NAV','Number_Positions','Drawdown'])
        self.historical.loc[0]=[np.nan,0,self.cash,0,0,self.nav,0,0]
        self.statistics={}
        self.others={}

        
    def trade(self,date,price,ticker,quantity,margin=100):
        
        '''
        Record trades into [trade] dataframe.
        If any position in [portfolio] is closed, calculate PnL and record into [tradesum].
        Else, update information in [portfolio].
        
        :param date: datetime of the trade
        :type date: datetime
        :param price: order price of the trade
        :type price: float
        :param ticker: asset to be traded
        :type ticker: str
        :param quantity: quantity of the trade
        :type quantity: float
        :param margin: margin rate% of the trade, not in use yet
        :type margin: float
        
        '''
        
        if ticker not in self.portfolio.Ticker.values:#new open
            comm = float(quantity*price*self.commission_rate)/10000
            self.cash -= price*quantity+comm
            self.trades.loc[len(self.trades)]=[date,ticker,quantity,price,comm,margin,'Open']
            self.portfolio.loc[len(self.portfolio)]=[date,ticker,quantity,price,quantity*price,price,
                                                     quantity*price,0,0,np.nan]
        else:#update
            pf_last=self.portfolio.loc[self.portfolio.Ticker==ticker]
            quantity_last=float(pf_last.Quantity)
            quantity_now=quantity_last+quantity
            avg_open=float(pf_last.Avg_Open_Price)
            r_pnl_last=float(pf_last.Realized_Pnl)
            
            if quantity_last*quantity_now <0:#position flip
                #part close
                quantity_mid=0
                comm = abs(float((quantity_mid-quantity_last)*price*self.commission_rate)/10000)
                self.cash -= price*(quantity_mid-quantity_last)+comm
                r_pnl=(quantity_mid-quantity_last)*(avg_open-price)-comm
                
                
                self.trades.loc[len(self.trades)]=[date,ticker,quantity_mid-quantity_last,price,comm,margin,'Close']
                self.tradesum.loc[len(self.tradesum)]=[pf_last.Open_Date.tolist()[0],ticker,avg_open,
                                                       float(pf_last.Outlay)+(quantity_mid-quantity_last)*price,
                                                       r_pnl_last+r_pnl,date,np.nan]
                self.portfolio=self.portfolio[self.portfolio.Ticker!=ticker].reset_index(drop=True)
                
                #open new
                quantity_rest=quantity_now-quantity_mid
                comm = abs(float(quantity_rest*price*self.commission_rate)/10000)
                self.cash -= price*quantity_rest+comm
                self.trades.loc[len(self.trades)]=[date,ticker,quantity_rest,price,comm,margin,'Open']
                self.portfolio.loc[len(self.portfolio)]=[date,ticker,quantity_rest,price,quantity_rest*price,
                                                         price,quantity_rest*price,0,0,np.nan]
            elif abs(quantity_now)<abs(quantity_last):#close
                comm = abs(float((quantity_now-quantity_last)*price*self.commission_rate)/10000)
                self.cash -= price*quantity+comm
                r_pnl=quantity*(avg_open-price)-comm
                if quantity_now==0:#full close
                    self.trades.loc[len(self.trades)]=[date,ticker,quantity,price,comm,margin,'Close']
                    self.tradesum.loc[len(self.tradesum)]=[pf_last.Open_Date.tolist()[0],ticker,avg_open,
                                                           float(pf_last.Outlay)+quantity*price,
                                                           r_pnl_last+r_pnl,date,np.nan]
                    self.portfolio=self.portfolio[self.portfolio.Ticker!=ticker].reset_index(drop=True)
                
                else:#partial close
                    u_pnl=-quantity_now*(avg_open-price)
                    self.trades.loc[len(self.trades)]=[date,ticker,quantity,price,comm,margin,'Close']
                    self.portfolio.loc[self.portfolio.Ticker==ticker,
                                       ['Quantity','Last_Price','Position_Value',
                                        'Outlay','Realized_Pnl','Unrealized_Pnl']]=\
                                       quantity_now,price,quantity_now*price,\
                                       float(pf_last.Outlay)+quantity*price,r_pnl_last+r_pnl,u_pnl
                    
            else:#further open
                comm = abs(float(quantity*price*self.commission_rate)/10000)
                self.cash -= price*quantity+comm
                u_pnl=-quantity_last*(avg_open-price)
                avg_open_new=(avg_open*quantity_last+quantity*price)/quantity_now
                self.trades.loc[len(self.trades)]=[date,ticker,quantity,price,comm,margin,'Open']
                self.portfolio.loc[self.portfolio.Ticker==ticker,
                                       ['Quantity','Last_Price','Position_Value',
                                        'Avg_Open_Price','Outlay','Unrealized_Pnl']]=\
                                       quantity_now,price,quantity_now*price,\
                                       avg_open_new,float(pf_last.Outlay)+quantity*price,u_pnl
                
    
    def update_portfolio(self,date,ticker,price):#per ticker
        
        '''
        Update [portfolio] with specified updated price.
        
        :param date: datetime to be updated
        :type date: datetime
        :param price: order price to be updated
        :type price: float
        :param ticker: asset to be updated
        :type ticker: str
        
        '''
        index=self.portfolio.index[self.portfolio.Ticker==ticker][0]
        quantity_now=self.portfolio.at[index,'Quantity']
        avg_open=self.portfolio.at[index,'Avg_Open_Price']
        self.portfolio.at[index,['Last_Price','Position_Value','Unrealized_Pnl']]=[price,
                                                                                   quantity_now*price,
                                                                                   -quantity_now*(avg_open-price)]
#         quantity_now=float(self.portfolio.loc[self.portfolio.Ticker==ticker].Quantity)
#         avg_open=float(self.portfolio.loc[self.portfolio.Ticker==ticker].Avg_Open_Price)
#         self.portfolio.loc[self.portfolio.Ticker==ticker,
#                            ['Last_Price','Position_Value','Unrealized_Pnl']]=price,quantity_now*price,-quantity_now*(avg_open-price)
        
    
    def save_historical(self,date):#per day
        
        '''
        Save current [portfolio] dataframe into [historical_portfolio] dataframe.
        Also, update aggregated information in [historical] dataframe.
        
        :param date: datetime to be saved and aggregated
        :type date: datetime
    
        '''
        
        self.historical_portfolio=self.historical_portfolio.append(self.portfolio,ignore_index=True)
        
        pos_value=self.portfolio.Position_Value.sum()
        r_pnl=self.portfolio.Realized_Pnl.sum()+self.tradesum.Realized_Pnl.sum()
        u_pnl=self.portfolio.Unrealized_Pnl.sum()
        self.nav=pos_value+self.cash
        npos=len(self.portfolio)
        dd=min(0,(self.nav-self.historical.NAV.max())/self.historical.NAV.max())
        
        index=len(self.historical)
        self.historical.at[index,['Datetime','Position_Value','Cash',
                                 'Realized_Pnl','Unrealized_Pnl','NAV',
                                 'Number_Positions','Drawdown']]=[date,pos_value,self.cash,r_pnl,
                                                                  u_pnl,self.nav,npos,dd]
    
    def calculate_stats(self,annualized_ratio,form='dict'):#per simulation
        
        '''
        Calculate performance statistics of the simulation.
        Output is either a HTML or a dictionary with results printed.
        
        :param annualized_ratio: the scalar to annualize standard deviation. \
        Eg. If data is BTC minute date, ratio = 60*24*365.
        :type annualized_ratio: float
        :param form: format to save the statistics, 'dict' or 'html'
        :type form: str
        :return: If form = 'html', results will be saved into a html report in the same folder.\
        When benchmark is not provided, it will print 'No benchmark provided.'.\
        If form = 'dict', results will be saved into [statistics] dictionary and also printed out.\
        When benchmark is not provided, only statistics without using benchmark are calculated.
        :rtype: html or dict
        
        '''
        

        if form=='html':
            if self.benchmark is not None:# to be updated with html object, benchmark to be a list/array
                combine=pd.merge(self.benchmark,self.historical[['Datetime','NAV']],on='Datetime',how='right')
                combine.rename(columns={'NAV':'Curve','Datetime':'date'}, inplace=True)
                combine.to_csv(f'./BenchmarkAnalysis_{self.portfolio_name}.csv')
                report = PerformanceReport(f'./BenchmarkAnalysis_{self.portfolio_name}.csv')
                report.generate_html_report()
            
            else:
                print ('No benchmark provided.')
            
        else:
            #trades
            total_trades=len(self.trades)
            total_closed=len(self.tradesum)
            r_pnl=self.portfolio.Realized_Pnl.sum()+self.tradesum.Realized_Pnl.sum()
            u_pnl=self.portfolio.Unrealized_Pnl.sum()
            total_pnl=r_pnl+u_pnl
            pnls=self.tradesum.Realized_Pnl.tolist()+self.portfolio.Realized_Pnl.tolist()
            total_commission=self.trades.Commission_Expense.sum()
            try:
                avg_trade_pnl=total_pnl/total_trades#
                p_win=len([x for x in pnls if x>=0])/len(pnls)#
                p_loss=len([x for x in pnls if x<0])/len(pnls)#
            except:
                avg_trade_pnl,p_win,p_loss=0,0,0
            #time series
            returns=self.historical.NAV.pct_change().tolist()[1:]
            total_ret=self.historical.NAV.iloc[-1]/self.historical.NAV.iloc[0]-1
            duration=(self.historical.Datetime.iloc[-1] - 
                      self.historical.Datetime.iloc[1])/datetime.timedelta(days=1)/365
            CAGR= ((self.nav/self.initial_capital)**(1/duration))-1
            std_ann=round(np.std(returns)*np.sqrt(annualized_ratio),2)
            mdd=self.historical.Drawdown.min()
            SR=CAGR/std_ann
            
            self.statistics['Trades']={}
            self.statistics['Trades']['Total Trades']=total_trades
            self.statistics['Trades']['Total Realized Pnl']=r_pnl
            self.statistics['Trades']['Total Unrealized Pnl']=u_pnl
            self.statistics['Trades']['Average Pnl per Trade']=avg_trade_pnl
            self.statistics['Trades']['Total Pnl']=total_pnl
            self.statistics['Trades']['Total Commission Expense']=total_commission
            self.statistics['Trades']['Lossing Rate']=p_loss
            self.statistics['Trades']['Winning Rate']=p_win
            self.statistics['Time Series']={}
            self.statistics['Time Series']['Starting NAV']=self.initial_capital
            self.statistics['Time Series']['Ending NAV']=self.nav
            self.statistics['Time Series']['Total Return']=total_ret
            self.statistics['Time Series']['Duration']=duration
            self.statistics['Time Series']['CAGR']=CAGR
            self.statistics['Time Series']['STD ANN']=std_ann
            self.statistics['Time Series']['Max Drawdown']=mdd
            self.statistics['Time Series']['Sharpe Ratio']=SR
            
            if self.benchmark is not None:
                total_ret_bm=self.benchmark.Benchmark.iloc[-1]/self.benchmark.Benchmark[0]-1
                IR=(total_ret-total_ret_bm)/std_ann
                self.statistics['Benchmark']={}
                self.statistics['Benchmark']['Total Return']=total_ret_bm
                self.statistics['Benchmark']['Information Ratio']=IR
                
         
        
            print('- Trade Analysis:')
            print(f"Total Trades: {total_trades} Total Pnl: {total_pnl} Average PnL per Trade: {avg_trade_pnl}")
            print(f"Total Realized Pnl: {r_pnl} Total Unrealized Pnl: {u_pnl} \
                  Total Commission Expense: {total_commission}")
            print(f'Winning Rate: {round(p_win*100,2)}% Lossing Rate: {round(p_loss*100,2)}%')
            
            print('- Time Series Analysis:')
            print(f'Starting NAV: {self.initial_capital} Ending NAV: {self.nav}')
            print(f'Total Return: {round(total_ret*100,2)}% across {duration} years')
            print(f'CAGR: {round(CAGR*100,2)}% ST DEV ANN: {std_ann} Sharpe Ratio: {SR}')
            print(f'Max Drawdown: {round(mdd*100,2)} %')
            plt.plot(self.historical.NAV, linewidth=0.8)
            plt.show()
        
        
    def get_position(self,ticker):
        
        '''
        Fetch the current position of the asset.
        
        :param ticker: the asset to be fetched
        :type ticker: str
        '''
        
        return self.portfolio[self.portfolio.Ticker==ticker]
    
    def save(self,directory):
        
        '''
        Save the object into a pickle file.
        
        :param directory: full directory to save the object, excluding format (.pkl)
        :type directory: str
        '''
        
        
        savedict={}
        savedict['Portfolio Name']=self.portfolio_name
        savedict['Initial Capital']=self.initial_capital
        savedict['NAV']=self.nav
        savedict['Cash']=self.cash
        savedict['Commission Rate']=self.commission_rate
        savedict['Benchmark']=self.benchmark
        savedict['Trade']=self.trades
        savedict['Portfolio']=self.portfolio
        savedict['Tradesum']=self.tradesum
        savedict['Historical Portfolio']=self.historical_portfolio
        savedict['Historical']=self.historical
        savedict['Statistics']=self.statistics
        savedict['Others']=self.others
        pickle.dump(savedict,open(directory+".pkl","wb"))
    
    def load(self,directory):
        
        '''
        Load back the object from a pickle file.
        
        :param directory: full directory to load the object, excluding format (.pkl)
        :type directory: str
        '''
        
        savedict= pickle.load(open(directory+".pkl","rb"))
        self.portfolio_name=savedict['Portfolio Name']
        self.initial_capital=savedict['Initial Capital']
        self.nav=savedict['NAV']
        self.cash=savedict['Cash']
        self.commission_rate=savedict['Commission Rate']
        self.benchmark=savedict['Benchmark']
        self.trades=savedict['Trade']
        self.portfolio=savedict['Portfolio']
        self.tradesum=savedict['Tradesum']
        self.historical_portfolio=savedict['Historical Portfolio']
        self.historical=savedict['Historical']
        self.statistics=savedict['Statistics']
        self.others=savedict['Others']
    
    
    
    
