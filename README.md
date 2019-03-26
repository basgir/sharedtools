# Installation 
```python
pip install git+https://github.com/swissborgcanada/sharedtools
from sharedtools.jupyter import Portfolio
```

# How to use Portfolio

## Please first specify the directory of portfolio pickle file you want to investigate
Eg. In the Jupyter terminal, directory is in format of *'../strategy_name/simulation_specifications/variable_specifications'*.<br>
```python
directory='../RSI/Pickle_day_longshort2/w0_p5'
```

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
