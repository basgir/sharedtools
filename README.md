# Installation 
```python
pip install git+https://github.com/swissborgcanada/sharedtools
from sharedtools.jupyter import Portfolio
```

# How to use Portfolio

## Please first specify the directory of portfolio pickle file you want to investigate
In the Jupyter terminal, directory is in format of *'../strategy_name/simulation_specifications/variable_specifications'*.<br>
```python
directory='../RSI/Pickle_day_longshort2/w0_p5'
```
**For RSI strategies**
* strategy_name = 'RSI'
* simulation_specifications can = 
    * 'Pickle_day_long2' for daily long-only simulations
    * 'Pickle_day_longshort2' for daily long-short simulations
    * 'Pickle_hour_long2' for hourly long-only simulations
    * 'Pickle_hour_longshort2' for hourly long-short simulations
* variable_specifications = 'w0_pX'
    * where X, the RSI window size, can be any integer number in range [5,60]
    
**For Contrarian strategies**
* strategy_name = 'Contrarian'
* simulation_specifications can = 'Pickle_day_limitlong' for daily long-only simulations
* variable_specifications = 'rX_pY'
    * where X, the sigma ratio, can be any number in [0.1, 0.2, ..., 1.0]
    * where Y, SMA window size, can be any integer number in range [5,60]

## To load a previous portfolio
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
