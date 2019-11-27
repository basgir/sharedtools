import json
import datetime
import uuid
import math
import pandas as pd
import numpy as np


def round_down(n, decimals=8):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


class Portfolio(object):
    def __init__(self, cash: float, commission_rate: float, portfolio_name: str = '',
                 daily_interest: float = 0.0002, benchmark: list = None):
        """"Portfolio is a generic object for varied portfolio simulations.
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
            cash: initial capital  in $ to trade
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
        self.initial_cash = cash
        self.cash = cash
        self.daily_interest = daily_interest
        self.statistics = {}
        self.total_nav = cash

        self.trades = pd.DataFrame(columns=['position_id', 'datetime', 'ticker', 'quantity', 'price',
                                            'commission', 'margin', 'direction'])

        self.holding = pd.DataFrame(columns=['position_id', 'ticker', 'open_date', 'avg_open_price', 'quantity',
                                             'last_price', 'notional_value',  'realized_pnl', 'unrealized_pnl'])

        self.tradesum = pd.DataFrame(columns=['position_id', 'ticker',
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
        comm = round_down(abs(notional_value * self.commission_rate))
        position_id = self._generate_position_id()

        self.cash -= (notional_value + comm)

        self.trades.at[len(self.trades), self.trades.columns.tolist()] = \
            [position_id, date, ticker, quantity, price, comm, margin, 'open']
        self.trade_count += 1

        self.tradesum.at[len(self.tradesum),
                         ['position_id', 'ticker',
                          'open_quantity', 'open_price', 'open_notional', 'open_date',
                          'close_quantity', 'close_price', 'close_notional', 'close_date',
                          'commission', 'interest', 'realized_pnl']] = \
            [position_id, ticker,
             quantity, price, abs(notional_value), date,
             0, 0, 0, None,
             comm, 0, -comm]

        self.holding.at[len(self.holding), self.holding.columns.tolist()] = \
            [position_id,  ticker,  date, price, quantity, price, notional_value, -comm, 0]

    def _open_further_position(self, position_id: str, date, price: float, quantity: float, margin: float = 100):
        open_notional_value = quantity * price
        comm = round_down(abs(open_notional_value * self.commission_rate))

        trade_sum = self.tradesum.loc[self.tradesum.position_id == position_id]
        open_quantity = trade_sum['open_quantity'].tolist()[0]
        open_notional = trade_sum['open_notional'].tolist()[0]
        close_quantity = trade_sum['close_quantity'].tolist()[0]
        curr_commission = trade_sum['commission'].tolist()[0]
        curr_r_pnl = trade_sum['realized_pnl'].tolist()[0]
        ticker = trade_sum['ticker'].tolist()[0]

        new_open_notional = open_notional + price * abs(quantity)
        new_open_quantity = open_quantity + quantity
        average_open_price = round_down(new_open_notional / abs(new_open_quantity))

        self.cash -= (open_notional_value + comm)

        self.trades.at[len(self.trades), self.trades.columns.tolist()] = \
            [position_id, date, ticker, quantity, price, comm, margin, 'open']
        self.trade_count += 1

        self.tradesum.loc[self.tradesum.position_id == position_id,
                          ['open_quantity', 'open_price', 'open_notional', 'commission', 'realized_pnl']] = \
            [new_open_quantity, average_open_price, new_open_notional, curr_commission + comm, curr_r_pnl - comm]

        curr_holding_qty = new_open_quantity + close_quantity
        self.holding.loc[self.holding.position_id == position_id,
                         ['quantity', 'last_price', 'notional_value', 'avg_open_price',
                          'realized_pnl', 'unrealized_pnl']] = \
            [curr_holding_qty, price, curr_holding_qty * price, average_open_price,
             curr_r_pnl - comm, curr_holding_qty * (price - average_open_price)]

    def _close_position(self, position_id: str,  date, price: float, quantity: float, margin: float = 100):
        close_notional_value = quantity * -1 * price
        close_comm = round_down(abs(close_notional_value * self.commission_rate))
        self.cash += (close_notional_value - close_comm)

        trade_sum = self.tradesum.loc[self.tradesum.position_id == position_id]
        commission = trade_sum['commission'].tolist()[0]
        r_pnl = trade_sum['realized_pnl'].tolist()[0]
        close_notional = trade_sum['close_notional'].tolist()[0]
        close_quantity = trade_sum['close_quantity'].tolist()[0]
        avg_open_price = trade_sum['open_price'].tolist()[0]
        ticker = trade_sum['ticker'].tolist()[0]

        new_close_notional = close_notional + price * abs(quantity)
        new_close_quantity = close_quantity + quantity
        average_close_price = round_down(new_close_notional / abs(new_close_quantity))

        # Update to trade
        self.trades.at[len(self.trades), self.trades.columns.tolist()] = [
            position_id, date, ticker, quantity, price, close_comm, margin, 'close'
        ]
        self.trade_count += 1

        r_pnl = round_down(quantity * -1 * (average_close_price - avg_open_price) + r_pnl - close_comm)
        self.tradesum.loc[self.tradesum.position_id == position_id,
                          ['close_quantity', 'close_price', 'close_notional',
                           'close_date', 'commission', 'realized_pnl']] = \
            [new_close_quantity, average_close_price, new_close_notional,
             date, commission + close_comm, r_pnl]

    def _close_full_position(self, position_id: str,  date, price: float, quantity: float, margin: float = 100):
        self._close_position(position_id,  date, price, quantity, margin)

        # Drop current holding
        self.holding = self.holding[self.holding.position_id != position_id].reset_index(drop=True)

    def _close_partial_position(self, position_id,  date, price: float, quantity: float, margin: float = 100):
        self._close_position(position_id, date, price, quantity, margin)

        trade_sum = self.tradesum.loc[self.tradesum.position_id == position_id]
        curr_r_pnl = trade_sum['realized_pnl'].tolist()[0]
        average_open_price = trade_sum['open_price'].tolist()[0]
        open_quantity = trade_sum['open_quantity'].tolist()[0]
        close_quantity = trade_sum['close_quantity'].tolist()[0]
        curr_holding_qty = open_quantity + close_quantity

        self.holding.loc[self.holding.position_id == position_id,
                         ['quantity', 'last_price', 'notional_value',
                          'avg_open_price', 'realized_pnl', 'unrealized_pnl']] = \
            [curr_holding_qty, price, curr_holding_qty * price, average_open_price, curr_r_pnl,
             curr_holding_qty * (price - average_open_price)]

    def trade(self, date,  ticker: str, quantity: float, price: float, margin: float = 100):
        """Record trades into [trade] dataframe.
        If any position in [portfolio] is closed,  calculate PnL and record into [tradesum].
        Else,  update information in [portfolio].

        Args:
        date: datetime of the trade
        price: order price of the trade
        ticker: asset to be traded
        quantity: quantity of the trade
        margin: margin rate% of the trade,  not in use yet
        """
        ticker = ticker.upper()
        if ticker not in self.holding.ticker.values:  # new open
            self._open_new_position(date, price, ticker, quantity, margin)

        else:  # update
            ticker_holding = self.holding.loc[self.holding.ticker == ticker]
            position_id = ticker_holding.position_id.tolist()[0]

            curr_holding_qty = float(ticker_holding.quantity.tolist()[0])
            new_qty = curr_holding_qty + quantity

            if curr_holding_qty * new_qty < 0:  # position flip
                self._close_full_position(position_id, date, price, -curr_holding_qty, margin)
                self._open_new_position(date, price, ticker, new_qty, margin)

            elif abs(new_qty) < abs(curr_holding_qty):  # close

                if new_qty == 0:  # full close
                    self._close_full_position(position_id, date, price, quantity, margin)

                else:  # partial close
                    self._close_partial_position(position_id, date, price, quantity, margin)

            else:  # further open
                self._open_further_position(position_id, date, price, quantity, margin)

    def update_portfolio(self, ticker: str, price: float):  # per ticker
        """
        Update portfolio with specified updated price. Call on every ticker by the end of the day

        Args:
            ticker: asset to be updated
            price: order price to be updated
        """
        # Update
        ticker = ticker.upper()
        if ticker.upper() in self.holding.ticker.values:
            holding_index = self.holding.index[self.holding.ticker == ticker][0]
            position_id = self.holding.at[holding_index, 'position_id']

            tradesum_index_lst = self.tradesum.index[self.tradesum.position_id == position_id]
            if not len(tradesum_index_lst):
                raise Exception(f'Internal error: position id {position_id} not in tradesum dataframe')
            elif len(tradesum_index_lst) != 1:
                raise Exception(f'Internal error: mulitple position id {position_id} in tradesum dataframe')

            tradesum_index = tradesum_index_lst[0]

            # Update tradesum realized pnl, daily interest and capital_balance
            open_quantity = self.tradesum.at[tradesum_index, 'open_quantity']
            if open_quantity < 0:
                curr_interest = self.tradesum.at[tradesum_index, 'interest']
                realized_pnl = self.tradesum.at[tradesum_index, 'realized_pnl']
                interest = abs(open_quantity) * price * self.daily_interest
                new_realized_pnl = realized_pnl - interest
                self.tradesum.at[tradesum_index, 'interest'] = curr_interest + interest
                self.tradesum.at[tradesum_index, 'realized_pnl'] = new_realized_pnl
                self.cash -= interest

                # Update current holding realized pnl
                self.holding.at[holding_index, 'realized_pnl'] = new_realized_pnl

            holding_quantity = self.holding.at[holding_index, 'quantity']
            avg_open = self.holding.at[holding_index, 'avg_open_price']

            self.holding.at[holding_index, ['last_price', 'notional_value', 'unrealized_pnl']] = \
                [price, holding_quantity * price, round_down(holding_quantity * (price - avg_open))]

    def save_historical(self, date):
        """Save current [portfolio] dataframe into [historical_portfolio] dataframe.
        Also, update aggregated information in [historical] dataframe.

        Args:
            date: datetime to be saved and aggregated
        """

        pos_value = self.holding.notional_value.sum()
        r_pnl = self.tradesum.realized_pnl.sum()
        u_pnl = self.holding.unrealized_pnl.sum()
        self.total_nav = pos_value + self.cash
        position_number = len(self.holding)

        draw_down = round_down(min(0, (self.total_nav - self.historical.nav.max()) / self.historical.nav.max()))
        index = len(self.historical)
        self.historical.at[index, self.historical.columns.tolist()] = [date, pos_value, self.cash, r_pnl,
                                                                       u_pnl, self.total_nav, position_number, draw_down]

    def calculate_stats(self, annualized_ratio: float):  # per simulation
        """Calculate performance statistics of the simulation.
        Output is either a HTML or a dictionary with results printed.

        Args:
            annualized_ratio: the scalar to annualize standard deviation.
            Eg. If data is BTC minute date,  ratio = 60*24*365.
        """
        # trades
        total_trades = len(self.trades)

        r_pnl = self.tradesum.realized_pnl.sum()
        u_pnl = self.holding.unrealized_pnl.sum()

        total_pnl = r_pnl + u_pnl
        pnls = self.tradesum[abs(self.tradesum.open_quantity) == abs(self.tradesum.close_quantity)].realized_pnl.tolist() \
               + (self.holding.realized_pnl + self.holding.unrealized_pnl).tolist()
        total_commission = self.trades.commission.sum()
        try:
            avg_trade_pnl = total_pnl / total_trades
            p_win = len([x for x in pnls if x >= 0]) / len(pnls)
            p_loss = len([x for x in pnls if x < 0]) / len(pnls)
        except:
            avg_trade_pnl, p_win, p_loss = 0, 0, 0
        # time series
        returns = self.historical.nav.pct_change().tolist()[1:]

        if len(self.historical) > 1:
            total_ret = self.historical.nav.iloc[-1] / self.historical.nav.iloc[0] - 1

            duration = (self.historical.datetime.iloc[-1] -
                        self.historical.datetime.iloc[0]) / datetime.timedelta(days=1) / 365

            # Compund annual growth rate
            CAGR = ((self.total_nav / self.cash) ** (1 / duration)) - 1
            std_ann = round(np.std(returns) * np.sqrt(annualized_ratio), 2)
            mdd = self.historical.drawdown.min()
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
        self.statistics['time_series']['starting_nav'] = self.cash
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
        print(f'Starting nav: {self.initial_cash} Ending nav: {self.total_nav}')
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
        return self.holding[self.holding.ticker == ticker]

    def save(self, directory: str):
        """Save current portfolio into a json file

        Args:
            directory: file path
        """
        save_dict = {
            'portfolio_name': self.portfolio_name,
            'initial_cash': self.initial_cash,
            'total_nav': self.total_nav,
            'cash': self.cash,
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
        """Load back the object from a json file.

        Args:
            directory: file path
        """
        with open(directory, 'r') as fp:

            portfolio = Portfolio(0, 0)
            loaded_dict = json.load(fp)
            portfolio.portfolio_name = loaded_dict['portfolio_name']
            portfolio.initial_cash = loaded_dict['initial_cash']
            portfolio.total_nav = loaded_dict['total_nav']
            portfolio.cash = loaded_dict['cash']
            portfolio.commission_rate = loaded_dict['commission_rate']
            portfolio.benchmark = loaded_dict['benchmark']

            portfolio.trades = pd.DataFrame.from_dict(json.loads(loaded_dict['trades']))
            portfolio.holding = pd.DataFrame.from_dict(json.loads(loaded_dict['holding']))
            portfolio.tradesum = pd.DataFrame.from_dict(json.loads(loaded_dict['tradesum']))
            portfolio.historical = pd.DataFrame.from_dict(json.loads(loaded_dict['historical']))
            portfolio.statistics = loaded_dict['statistics']

            return portfolio

