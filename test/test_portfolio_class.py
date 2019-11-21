import unittest
from sharedtools.Portfolio import Portfolio
from datetime import datetime, timedelta
import ipdb

def date_traverse(start_date, delta):
    """
    Taverse n days back

    Args:
        start_date: date object
        delta:      int

    Returns:
        date object

    """

    traverse_date = start_date - timedelta(days=delta)
    return traverse_date


class TestPortfolioClass(unittest.TestCase):

    def test_full_close(self):

        d = datetime(2019, 1, 1)
        portfolio = Portfolio(10, 0.001, 'test')
        portfolio.trade(d, 'ETHBTC', 100, 1)
        self.assertFalse(portfolio.holding.empty, 'holding should not be empty')
        self.assertFalse(portfolio.tradesum.empty, 'tradesum should not be empty')
        position_id = portfolio.holding[portfolio.holding.ticker == 'ETHBTC'].position_id[0]

        d = date_traverse(d, -1)
        portfolio.trade(d, 'ETHBTC', -100, 2)
        self.assertTrue(portfolio.holding.empty, 'holding should be empty')
        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0] * -1, trade_sum.close_quantity[0])
        self.assertEqual(trade_sum.realized_pnl[0], 99.7)

    def test_partial_close(self):
        # First day
        d1 = datetime(2019, 1, 1)
        portfolio = Portfolio(10, 0.001, 'test')
        portfolio.trade(d1, 'ETHBTC', 200, 1)
        portfolio.update_portfolio('ETHBTC', 1)
        self.assertFalse(portfolio.holding.empty, 'holding should not be empty')
        self.assertFalse(portfolio.tradesum.empty, 'tradesum should not be empty')
        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        position_id = holding_row.position_id[0]

        # Second day
        d2 = date_traverse(d1, -1)
        portfolio.trade(d2, 'ETHBTC', -100, 2)
        portfolio.update_portfolio('ETHBTC', 2)
        portfolio.save_historical(d2)

        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        self.assertEqual(holding_row.quantity[0], 100)
        self.assertEqual(holding_row.realized_pnl[0], 99.6)
        self.assertEqual(holding_row.unrealized_pnl[0], 100)

        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0], 200)
        self.assertEqual(trade_sum.close_quantity[0], -100)
        self.assertEqual(trade_sum.realized_pnl[0], 99.6)
        self.assertEqual(trade_sum.commission[0], 0.4)

    def test_further_open(self):
        # First day
        d1 = datetime(2019, 1, 1)
        portfolio = Portfolio(10, 0.001, 'test')
        portfolio.trade(d1, 'ETHBTC', 200, 1)
        portfolio.update_portfolio('ETHBTC', 1)
        portfolio.save_historical(d1)
        self.assertFalse(portfolio.holding.empty, 'holding should not be empty')
        self.assertFalse(portfolio.tradesum.empty, 'tradesum should not be empty')
        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        position_id = holding_row.position_id[0]

        # Second day
        d2 = date_traverse(d1, -1)
        portfolio.trade(d2, 'ETHBTC', 100, 2)
        portfolio.update_portfolio('ETHBTC', 2)
        portfolio.save_historical(d2)

        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        self.assertEqual(holding_row.quantity[0], 300, 'holding quantity')
        self.assertEqual(holding_row.avg_open_price[0], 400/holding_row.quantity[0], 'avg open price')
        self.assertEqual(holding_row.realized_pnl[0], -0.4, 'realized pnl')
        self.assertEqual(holding_row.unrealized_pnl[0], (2 - 400/300) * holding_row.quantity[0], 'unrealized pnl')

        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0], 300, 'tradesum open quantity')
        self.assertEqual(trade_sum.open_price[0], 400/300, 'tradesum open price')
        self.assertEqual(trade_sum.close_quantity[0], 0, 'tradesum close quantity')
        self.assertEqual(trade_sum.realized_pnl[0], -0.4)
        self.assertEqual(trade_sum.commission[0], 0.4)

    def test_full_close_short(self):

        d = datetime(2019, 1, 1)
        portfolio = Portfolio(10, 0.001, 'test')
        portfolio.trade(d, 'ETHBTC', -100, 1)
        portfolio.update_portfolio('ETHBTC', 1)
        portfolio.save_historical(d)
        self.assertFalse(portfolio.holding.empty, 'holding should not be empty')
        self.assertFalse(portfolio.tradesum.empty, 'tradesum should not be empty')
        position_id = portfolio.holding[portfolio.holding.ticker == 'ETHBTC'].position_id[0]

        d = date_traverse(d, -1)
        portfolio.update_portfolio('ETHBTC', 2)
        portfolio.save_historical(d)

        d = date_traverse(d, -1)
        portfolio.trade(d, 'ETHBTC', 100, 2)
        portfolio.update_portfolio('ETHBTC', 2)
        portfolio.save_historical(d)
        self.assertTrue(portfolio.holding.empty, 'holding should be empty')
        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0] * -1, trade_sum.close_quantity[0])
        self.assertEqual(trade_sum.realized_pnl[0], (2-1) * -100 - 100 * (1 + 2) *0.0002 - (1+2)*100*0.001)

    def test_partial_close_further_open(self):
        # First day
        d1 = datetime(2019, 1, 1)
        portfolio = Portfolio(10, 0.001, 'test')
        portfolio.trade(d1, 'ETHBTC', 200, 1)
        portfolio.update_portfolio('ETHBTC', 1)
        portfolio.save_historical(d1)
        self.assertFalse(portfolio.holding.empty, 'holding should not be empty')
        self.assertFalse(portfolio.tradesum.empty, 'tradesum should not be empty')
        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        position_id = holding_row.position_id[0]

        # Second day
        d2 = date_traverse(d1, -1)
        portfolio.trade(d2, 'ETHBTC', -100, 2)
        portfolio.update_portfolio('ETHBTC', 2)
        portfolio.save_historical(d2)

        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        self.assertEqual(holding_row.quantity[0], 100)
        self.assertEqual(holding_row.realized_pnl[0], 99.6)
        self.assertEqual(holding_row.unrealized_pnl[0], 100)

        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0], 200)
        self.assertEqual(trade_sum.close_quantity[0], -100)
        self.assertEqual(trade_sum.realized_pnl[0], 99.6)
        self.assertEqual(trade_sum.commission[0], 0.4)

        # Third day
        d3 = date_traverse(d2, -1)
        portfolio.trade(d3, 'ETHBTC', 100, 3)
        portfolio.update_portfolio('ETHBTC', 3)
        portfolio.save_historical(d3)

        holding_row = portfolio.holding.loc[portfolio.holding.ticker == 'ETHBTC']
        self.assertEqual(holding_row.quantity[0], 200)
        self.assertEqual(holding_row.realized_pnl[0], 99.3)
        self.assertEqual(holding_row.unrealized_pnl[0], 266.666668)

        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0], 300)
        self.assertEqual(trade_sum.close_quantity[0], -100)
        self.assertEqual(trade_sum.realized_pnl[0], 99.3)
        self.assertEqual(trade_sum.commission[0], 0.7)

        # Fourth day
        d = date_traverse(d3, -1)
        portfolio.trade(d, 'ETHBTC', -200, 3)
        portfolio.update_portfolio('ETHBTC', 3)
        portfolio.save_historical(d)
        self.assertTrue(portfolio.holding.empty, 'holding should be empty')
        trade_sum = portfolio.tradesum[portfolio.tradesum.position_id == position_id]
        self.assertEqual(trade_sum.open_quantity[0] * -1, trade_sum.close_quantity[0])
        self.assertEqual(portfolio.cash, trade_sum.realized_pnl[0] + 10)
        portfolio.calculate_stats(365)

if __name__ == '__main__':
    unittest.main()