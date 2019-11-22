- Installation
- How to build a portfolio
- How to view an existing portfolio

# Installation 
```python
pip install git+https://github.com/swissborgcanada/sharedtools
from sharedtools.Portfolio import Portfolio
```
# Basic usage

Portfolio is a generic object for varied portfolio simulations.
The object tracks and stores all trades and historical details.
        
    
    # Init portfolio
    # Portfolio(<initial balance>, <transaction fee rate>, <portfolio name>)
    portfolio = Portfolio(10, 0.001, 'test') 
    
    # Trade
    d = datetime(2019, 1, 1)
    portfolio.trade(d, 'ETHBTC', -100, 1) 
    
    # Update daily holding urealized PNL 
    # update_portfolio(<asset>, <price>)
    portfolio.update_portfolio('ETHBTC', 1)
    
    # Save portfolio daily nav
    portfolio.save_historical(d)
    
    # Calculate performance statistics of the simulation
    # calculate_stats(<annualized_ratio>)
    portfolio.calculate_stats(354)
    
## Dataframe in portfolio
    # Current open positions
    portfolio.holding
    
    # Trades summary with matched open and close information
    portfolio.tradesum
    
    # All historical trades
    portfolio.trades
    
    # Historical aggregated PnL information
    portfolio.historical
    
    # Portfolop performance statistics (after called calculate_stats method)
    portfolio.statistics
    
    

# How to view an existing portfolio
To be continued

