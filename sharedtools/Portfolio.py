import json
import datetime
import uuid
from decimal import Decimal
import pandas as pd
import numpy as np


def day_difference(later_date, ealier_date):
    """
    Day difference between two dates

    :param
        @later_date date object
        @ealier_date date object

    :return: int
    """
    delta = later_date - ealier_date
    return delta.days


class Portfolio(object):
    def __init__(self, capital_balance: float, commission_rate: float, portfolio_name: str = '',
                 daily_interest: float = 0.0002, benchmark: list = None):
        """"
        Portfolio is a generic object for varied portfolio simulations.
        The object tracks and stores all trades and historical details.

        Output data frames:
        [trades] all historical trades
        [portfolio] current open positions
        [tradesum] all trades that have been closed and consolidated
        [historical_portfolio] historical open positions for all datetime
        [historical] historical portfolio aggregated information for all datetime

        Output dictionaries:
        [statistics] performance statistics

        Args:
            capital_balance: initial capital  in $ to trade
            commission_rate: commission rate in basis points
            daily_interest: daily interest for short an asset
            portfolio_name: name of portfolio
            benchmark: a dataframe consisting benchmark time series. \
                       Dataframe columns should be set to 'Datetime' and 'benchmark'.\
                       Default is None.
        """
        self.portfolio_name = portfolio_name
        self.commission_rate = commission_rate
        self.benchmark = benchmark
        self.trade_count = 0
        self.capital_balance = capital_balance
        self.daily_interest = daily_interest
        self.statistics = {}
        self.total_nav = capital_balance

        self.trades = pd.DataFrame(columns=['position_id', 'datetime', 'ticker', 'quantity', 'price',
                                            'commission', 'margin', 'direction'])

        self.holding = pd.DataFrame(columns=['position_id', 'ticker', 'open_date', 'avg_open_price', 'quantity',
                                             'last_price', 'notional_value',  'realized_pnl', 'unrealized_pnl'])

        self.position = pd.DataFrame(columns=['position_id', 'ticker',
                                              'open_quantity', 'open_price', 'open_notional', 'open_date',
                                              'close_quantity', 'close_price', 'close_notional', 'close_date',
                                              'commission', 'interest', 'realized_pnl'])

        self.historical = pd.DataFrame(columns=['datetime', 'notional_value', 'cash', 'realized_pnl', 'unrealized_pnl',
                                                'nav', 'number_positions', 'drawdown'])

    @staticmethod
    def _generate_position_id():
        return str(uuid.uuid4())[:18]

    def _open_new_position(self, date, price: float, ticker: str, quantity: float, margin: float = 100):
        notional_value = quantity * price
        comm = abs(notional_value * self.commission_rate)
        position_id = self._generate_position_id()

        self.capital_balance -= (notional_value - comm)

        self.trades.at[len(self.trades), self.trades.columns.tolist()] = \
            [position_id, date, ticker, quantity, price, comm, margin, 'open']
        self.trade_count += 1

        self.holding.at[len(self.holding), self.holding.columns.tolist()] = \
            [position_id,  ticker,  date, price, quantity, price, notional_value, -comm, 0]

        self.position.at[len(self.holding), ['position_id', 'ticker', 'open_quantity',
                                             'open_price', 'open_notional', 'open_date',
                                             'commission', 'realized_pnl']] = \
            [position_id, ticker, quantity, price, notional_value, date, comm, -comm]

    def _open_further_position(self, position_id, date, price: float, quantity: float, margin: float = 100):
        open_notional_value = quantity * price
        comm = abs(open_notional_value * self.commission_rate)

        curr_trade_sum = self.position.loc[self.position.position_id == position_id]
        curr_open_quantity = curr_trade_sum['open_quantity']
        curr_open_price = curr_trade_sum['open_price']
        curr_commission = curr_trade_sum['commission']
        curr_r_pnl = curr_trade_sum['realized_pnl']
        ticker = curr_trade_sum['ticker']

        new_open_quantity = curr_open_quantity + quantity
        average_open_price = (curr_open_price * curr_open_quantity + open_notional_value) / abs(new_open_quantity)
        new_open_notional = new_open_quantity * average_open_price

        self.capital_balance -= (open_notional_value - comm)

        self.trades.at[len(self.trades), self.trades.columns.tolist()] = \
            [position_id, date, ticker, quantity, price, comm, margin, 'open']
        self.trade_count += 1

        self.holding.loc[self.holding.position_id == position_id,
                         ['quantity', 'last_price', 'notional_value', 'avg_open_price', 'realized_pnl',
                          'unrealized_pnl']] = \
            [new_open_quantity, price, new_open_notional, average_open_price, price, quantity * price,
             price, open_amount, -comm, 0]

        self.position.loc[self.position.position_id == position_id,
                          ['open_quantity', 'open_price', 'open_notional',
                           'commission', 'realized_pnl']] = \
            [new_open_quantity, average_open_price, new_open_notional,  curr_commission + comm, curr_r_pnl - comm]







    def _close_full_position(self, position_id,  date, price: float, quantity: float, margin: float = 100):

        close_comm = abs(quantity * price * self.commission_rate)
        close_amt = quantity * price - close_comm

        curr_trade_sum = self.position.loc[self.position.position_id == position_id]
        curr_commission = curr_trade_sum['commission']
        curr_r_pnl = curr_trade_sum['realized_pnl']
        # TODO check if off by one
        curr_interest = curr_trade_sum['interest']
        avg_open_price = curr_trade_sum['open_price']
        ticker = curr_trade_sum['ticker']

        r_pnl = quantity * (price - avg_open_price) - close_comm
        self.position.loc[self.position.position_id == position_id,
                          ['close_quantity', 'close_price', 'close_notional',
                           'close_date', 'commission', 'realized_pnl']] = \
            [quantity, price, quantity * price, date, curr_commission + close_comm, curr_r_pnl + r_pnl]

        # Update to trade
        self.trades.at[len(self.trades), self.trades.columns.tolist()] = [
            position_id, date, ticker, quantity, price, close_comm, margin, 'close'
        ]
        self.trade_count += 1

        # Drop current holding
        self.holding = self.holding[self.holding.position_id != position_id].reset_index(drop=True)
        self.capital_balance += (close_amt - curr_interest)

    def _close_partial_position(self, position_id,  date, price: float, quantity: float, margin: float = 100):
        close_comm = abs(quantity * price * self.commission_rate)
        close_amt = quantity * price - close_comm

        curr_trade_sum = self.position.loc[self.position.position_id == position_id]
        curr_commission = curr_trade_sum['commission']
        curr_r_pnl = curr_trade_sum['realized_pnl']
        # TODO check if off by one
        curr_interest = curr_trade_sum['interest']
        avg_open_price = curr_trade_sum['open_price']
        ticker = curr_trade_sum['ticker']

        r_pnl = quantity * (price - avg_open_price) - close_comm
        self.position.loc[self.position.position_id == position_id,
                          ['close_quantity', 'close_price', 'close_notional',
                           'close_date', 'commission', 'realized_pnl']] = \
            [quantity, price, quantity * price, date, curr_commission + close_comm, curr_r_pnl + r_pnl]

        # Update to trade
        self.trades.at[len(self.trades), self.trades.columns.tolist()] = [
            position_id, date, ticker, quantity, price, close_comm, margin, 'close'
        ]
        self.trade_count += 1

        # Drop current holding
        self.capital_balance += (close_amt - curr_interest)


    def trade(self, date, price: float, ticker: str, quantity: float, margin: float = 100):
        """
        Record trades into [trade] dataframe.
        If any position in [portfolio] is closed,  calculate PnL and record into [tradesum].
        Else,  update information in [portfolio].

        Args:
        date: datetime of the trade
        price: order price of the trade
        ticker: asset to be traded
        quantity: quantity of the trade
        margin: margin rate% of the trade,  not in use yet
        """

        if ticker not in self.holding.ticker.values:  # new open
            self._open_new_position(date, price, ticker, quantity)

        else:  # update
            ticker_holding = self.holding.loc[self.holding.Ticker == ticker]
            # e.g. curr_holding_qty = -1, quantity = 2, new_qty = 1
            # e.g. curr_holding_qty = 1, quantity = -2, new_qty = 1
            position_id = ticker_holding.quantity
            curr_holding_qty = float(ticker_holding.quantity)
            new_qty = curr_holding_qty + quantity
            curr_holding_avg_open = float(ticker_holding.avg_open_price)
            curr_holding_r_pnl = float(ticker_holding.realized_pnl)

            if curr_holding_qty * new_qty < 0:  # position flip
                self._close_full_position(position_id, date, price, -curr_holding_qty)
                self._open_new_position(date, price, ticker, new_qty)

            elif abs(new_qty) < abs(curr_holding_qty):  # close

                if new_qty == 0:  # full close
                    self._close_full_position(position_id, date, price, quantity)

                else:  # partial close
                    self._close_partial_position(position_id, date, price, quantity)

            else:  # further open
                comm = abs(float(quantity * price * self.commission_rate))
                curr_holding_open_amount = price * quantity + comm
                self.capital_balance -= curr_holding_open_amount

                u_pnl = -curr_holding_qty * (curr_holding_avg_open - price)
                avg_open_new = (curr_holding_avg_open * curr_holding_qty + quantity * price) / new_qty
                self.trades.at[len(self.trades), self.trades.columns.tolist()] = \
                    [date, ticker, quantity, price, comm, margin, 'open']
                self.trade_count += 1
                # Update current holding
                index = self.holding.index[self.holding.Ticker == ticker][0]
                self.holding.at[index, ['quantity', 'last_price', 'notional_value',
                                        'avg_open_price', 'outlay', 'realized_pnl', 'unrealized_pnl']] = \
                    [new_qty, price, new_qty * price, avg_open_new,
                     float(ticker_holding.outlay) + quantity * price, curr_holding_r_pnl - comm, u_pnl]

    def update_portfolio(self, ticker: str, price: float, curr_date):  # per ticker
        """
        Update [portfolio] with specified updated price.

        Args:
            curr_date: datetime to be updated
            price: order price to be updated
            ticker: asset to be updated
        """
        index = self.holding.index[self.holding.Ticker == ticker][0]
        quantity_now = self.holding.at[index, 'quantity']
        interest = 0
        if quantity_now < 0:
            open_date = self.holding.at[index, 'open_date']
            interest = 0.0002 * day_difference(curr_date, open_date)

        avg_open = self.holding.at[index, 'avg_open_price']
        self.holding.at[index, ['last_price', 'notional_value', 'unrealized_pnl']] = \
            [price, quantity_now * price, quantity_now * (price - avg_open) - interest * price]

    def calculate_stats(self, annualized_ratio):  # per simulation
        '''
        Calculate performance statistics of the simulation.
        Output is either a HTML or a dictionary with results printed.

        :param annualized_ratio: the scalar to annualize standard deviation. \
        Eg. If data is BTC minute date,  ratio = 60*24*365.
        :type annualized_ratio: float
        :param form: format to save the statistics,  'dict' or 'html'
        :type form: str
        :return: If form = 'html',  results will be saved into a html report in the same folder.\
        When benchmark is not provided,  it will print 'No benchmark provided.'.\
        If form = 'dict',  results will be saved into [statistics] dictionary and also printed out.\
        When benchmark is not provided,  only statistics without using benchmark are calculated.
        :rtype: html or dict
        '''
        # trades
        total_trades = len(self.trades)

        r_pnl = self.holding.realized_pnl.sum() + self.tradesum.realized_pnl.sum()
        u_pnl = self.holding.unrealized_pnl.sum()

        total_pnl = r_pnl + u_pnl
        pnls = self.tradesum.realized_pnl.tolist() + self.holding.realized_pnl.tolist()
        total_commission = self.trades.Commission_Expense.sum()
        try:
            avg_trade_pnl = total_pnl / total_trades
            p_win = len([x for x in pnls if x >= 0]) / len(pnls)
            p_loss = len([x for x in pnls if x < 0]) / len(pnls)
        except:
            avg_trade_pnl, p_win, p_loss = 0, 0, 0
        # time series
        returns = self.historical.NAV.pct_change().tolist()[1:]

        if len(self.historical) > 1:
            total_ret = self.historical.NAV.iloc[-1] / self.historical.NAV.iloc[0] - 1

            duration = (self.historical.Datetime.iloc[-1] -
                        self.historical.Datetime.iloc[1]) / datetime.timedelta(days=1) / 365
            # Compund annual growth rate
            CAGR = ((self.total_nav / self.capital_balance) ** (1 / duration)) - 1
            std_ann = round(np.std(returns) * np.sqrt(annualized_ratio), 2)
            mdd = self.historical.Drawdown.min()
            SR = CAGR / std_ann
            t = np.sqrt(len(returns)) * np.mean(returns) / np.std(returns)

        else:
            total_ret = duration = CAGR = std_ann = mdd = SR = t = 0

        # TODO: look deeper into nan value
        if np.isnan(std_ann):
            std_ann, SR, t = None, None, None

        self.statistics['trades'] = {}
        self.statistics['trades']['total_trades'] = total_trades
        self.statistics['trades']['total_realized_pnl'] = r_pnl
        self.statistics['trades']['total_unrealized_pnl'] = u_pnl
        self.statistics['trades']['average_pnl_per_trade'] = avg_trade_pnl
        self.statistics['trades']['total_pnl'] = total_pnl
        self.statistics['trades']['total_commission'] = total_commission
        self.statistics['trades']['lossing_rate'] = p_loss
        self.statistics['trades']['winning_rate'] = p_win
        self.statistics['time_series'] = {}
        self.statistics['time_series']['starting_nav'] = self.capital_balance
        self.statistics['time_series']['ending_nav'] = self.total_nav
        self.statistics['time_series']['total_return'] = total_ret
        self.statistics['time_series']['duration'] = duration
        self.statistics['time_series']['CAGR'] = CAGR
        self.statistics['time_series']['std_ann'] = std_ann
        self.statistics['time_series']['max_drawdown'] = mdd
        self.statistics['time_series']['sharpe_ratio'] = SR
        self.statistics['time_series']['t_statistics'] = t

        if self.benchmark is not None:
            total_ret_bm = self.benchmark.benchmark.iloc[-1] / self.benchmark.benchmark[0] - 1
            IR = (total_ret - total_ret_bm) / std_ann
            self.statistics['benchmark'] = {}
            self.statistics['benchmark']['total_return'] = total_ret_bm
            self.statistics['benchmark']['information_ratio'] = IR

        print('- Trade Analysis:')
        print(
            f"Total trades: {total_trades} Total Pnl: {total_pnl} Average PnL per Trade: {avg_trade_pnl}")
        print(f"Total Realized Pnl: {r_pnl} Total Unrealized Pnl: {u_pnl} \
                Total Commission Expense: {total_commission}")
        print(f'Winning Rate: {round(p_win*100, 2)}% Lossing Rate: {round(p_loss*100, 2)}%')

        print('- Time Series Analysis:')
        print(f'Starting NAV: {self.capital_balance} Ending NAV: {self.total_nav}')
        print(f'Total Return: {round(total_ret*100, 2)}% across {duration} years')
        print(f'CAGR: {round(CAGR*100, 2)}% ST DEV ANN: {std_ann} Sharpe Ratio: {SR}')
        print(f'Max Drawdown: {round(mdd*100, 2)} %')
        print(f't-Statistics: {t}')

    def get_position(self, ticker: str):
        """
        Fetch the current position of the asset.
        Args:
            ticker: the asset to be fetched
        """
        return self.holding[self.holding.Ticker == ticker]

    def save_historical(self, date):  # per day
        """
        Save current [portfolio] dataframe into [historical_portfolio] dataframe.
        Also, update aggregated information in [historical] dataframe.

        Args:
            date: datetime to be saved and aggregated

        """

        pos_value = self.holding.notional_value.sum()
        r_pnl = self.holding.realized_pnl.sum() + self.tradesum.realized_pnl.sum()
        u_pnl = self.holding.unrealized_pnl.sum()
        self.total_nav = pos_value + self.capital_balance
        npos = len(self.holding)
        dd = min(0, (self.total_nav - self.historical.NAV.max()) / self.historical.NAV.max())

        index = len(self.historical)
        self.historical.at[index, self.historical.columns.tolist()] = [date, pos_value, self.capital_balance, r_pnl,
                                                                       u_pnl, self.total_nav, npos, dd]

    def save(self, directory: str):
        """Save current portfolio into a json file

        Args:
            directory: file path
        """
        save_dict = {
            'initial_capital': self.capital_balance,
            'NAV': self.total_nav,
            'total_cash': self.capital_balance,
            'commission_rate': self.commission_rate,
            'benchmark': self.benchmark,
            'trades': self.trades.to_json(orient='records'),
            'holding': self.holding.to_json(orient='records'),
            'tradesum': self.tradesum.to_json(orient='records'),
            'historical': self.historical.to_json(orient='records'),
            'statistics':  self.statistics
        }


        with open(directory, 'w') as fp:
            json.dump(save_dict, fp)

    @classmethod
    def load(cls, directory):
        """
        Load back the object from a json file.

        Args:
            directory: file path
        """
        with open(directory, 'r') as fp:

            portfolio = Portfolio(0, 0)
            loaded_dict = json.load(fp)
            portfolio.portfolio_name = loaded_dict['portfolio_name']
            portfolio.capital_balance = loaded_dict['initial_capital']
            portfolio.nav = loaded_dict['NAV']
            portfolio.cash = loaded_dict['total_cash']
            portfolio.commission_rate = loaded_dict['commission_rate']
            portfolio.benchmark = loaded_dict['benchmark']

            portfolio.trades = pd.DataFrame.from_dict(json.loads(loaded_dict['trades']))
            portfolio.holding = pd.DataFrame.from_dict(json.loads(loaded_dict['holding']))
            portfolio.tradesum = pd.DataFrame.from_dict(json.loads(loaded_dict['tradesum']))
            portfolio.historical = pd.DataFrame.from_dict(json.loads(loaded_dict['historical']))
            portfolio.statistics = loaded_dict['statistics']

            return portfolio

if __name__ == '__main__':
    from  datetime import datetime
    d = datetime(2019,1,1)
    test_portfolio = Portfolio(10, 0.001, 'test')
    test_portfolio.trade(d, 0.02, 'ETHBTC', 20)
    import ipdb
    ipdb.set_trace()
