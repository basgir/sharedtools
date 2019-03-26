- Installation
- How to build a portfolio
- How to view an existing portfolio

# Installation 
```python
pip install git+https://github.com/swissborgcanada/sharedtools
from sharedtools.jupyter import Portfolio
```
# How to build a portfolio

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
    :param benchmark: a dataframe consisting benchmark time series. 
                    Dataframe columns should be set to 'Datetime' and 'Benchmark'.
                    Default is None.
    :type benchmark: a list, or an array
    :param others: other miscellaneous data
    :type portfolio_name: dict
    
## method/trade
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

## method/update_portfolio
Update [portfolio] with specified updated price.
        
    :param date: datetime to be updated
    :type date: datetime
    :param price: order price to be updated
    :type price: float
    :param ticker: asset to be updated
    :type ticker: str

## method/save_historical
Save current [portfolio] dataframe into [historical_portfolio] dataframe.
Also, update aggregated information in [historical] dataframe.
        
    :param date: datetime to be saved and aggregated
    :type date: datetime

## method/calculate_stats
Calculate performance statistics of the simulation.
Output is either a HTML or a dictionary with results printed.
        
    :param annualized_ratio: the scalar to annualize standard deviation. 
      Eg. If data is BTC minute date, ratio = 60*24*365.
    :type annualized_ratio: float
    :param form: format to save the statistics, 'dict' or 'html'
    :type form: str
    :return: If form = 'html', results will be saved into a html report in the same folder.
      When benchmark is not provided, it will print 'No benchmark provided.'.
      If form = 'dict', results will be saved into [statistics] dictionary and also printed out.
      When benchmark is not provided, only statistics without using benchmark are calculated.
    :rtype: html or dict
## method/get_position
Fetch the current position of the asset.
        
    :param ticker: the asset to be fetched
    :type ticker: str
## method/save
Save the object into a pickle file.
        
    :param directory: full directory to save the object, excluding format (.pkl)
    :type directory: str
## method/load
Load back the object from a pickle file.
        
    :param directory: full directory to load the object, excluding format (.pkl)
    :type directory: str
## method/output_excel
Save all information from a pickle file to a Excelsheet in the same folder.
        
    :param directory: full directory to load the object, excluding format (.pkl)
    :type directory: str


# How to view an existing portfolio

## First specify the directory of portfolio pickle file you want to investigate
Eg. In the Jupyter terminal, directory is in format of *'../strategy_name/simulation_specifications/variable_specifications'*.<br>
```python
directory='../RSI/Pickle_day_longshort2/w0_p5'
```

## To load an existing portfolio
```python
a=Portfolio(initial_capital=1000000, commission_rate=17.5, portfolio_name='MeanReversion')
a.load(directory)
```
## To see performance statistics
```python
a.statistics
```
## To see historical aggregated PnL information
```python
a.historical
```
## To see all historical trades
```python
a.trades
```
## To see trades summary
```python
a.tradesum
```
## To export all information into xlsx. format
```python
a.output_excel(directory)
```
