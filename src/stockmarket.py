from typing import Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 12})


class MarkerOperator:
    def __init__(self, estimator, features_names: list, initial_cash: float = 1000, initial_stocks: int = 0,
                 open_col: str = 'Abertura', close_col: str = 'Fech.', min_col: str = 'Mínimo',
                 max_col: str = 'Máximo', price_ref_col: str = 'Fech.',
                 daily_negotiable_perc: float = 0.5, min_stocks_op: int = 1, broker_taxes: float = 0) -> None:
        # Input parameters:
        self.estimator = estimator
        self.initial_cash = initial_cash
        self.initial_stocks = initial_stocks
        self.features_names = features_names
        self.open_col = open_col
        self.close_col = close_col
        self.min_col = min_col
        self.max_col = max_col
        self.price_ref_col = price_ref_col
        # Market policies:
        self.daily_negotiable_perc = daily_negotiable_perc
        self.min_stocks_op = min_stocks_op
        self.broker_taxes = broker_taxes
        # Output:
        self.op_results = None

    def run(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        operation actions:
            0 - hold position
            1 - buy stocks
            2 - sell stocks
        """
        y_pred = self.estimator.predict(df_data[self.features_names])
        df_tmp = df_data.copy()
        df_tmp['op'] = y_pred
        df_tmp['cash'] = float(self.initial_cash)
        df_tmp['n_stocks'] = int(self.initial_stocks)
        df_tmp['wealth'] = df_tmp['cash'] + df_tmp['n_stocks'] * df_tmp[self.price_ref_col]
        # Determining the performance of a baseline model performance:
        wealth_init = df_tmp['wealth'].values[0]
        n_stocks_init = int(wealth_init/df_tmp[self.price_ref_col].values[0])
        df_tmp['baseline_wealth'] = (wealth_init - n_stocks_init*df_tmp[self.price_ref_col].values[0]) + n_stocks_init * df_tmp[self.price_ref_col]

        # Appending a new row:
        new_row = pd.DataFrame([{col:np.nan for col in df_tmp.columns}])
        new_row.index = [df_tmp.index[-1] + timedelta(days=1)]
        df_tmp = df_tmp.append(new_row)

        for i, (idx, op_action) in enumerate(zip(df_tmp.index[:-1], y_pred)):
            idx_next = df_tmp.index[i+1]
            current_cash = df_tmp.loc[idx, 'cash']
            current_n_stocks = df_tmp.loc[idx, 'n_stocks']
            stock_price = df_tmp.loc[idx, self.price_ref_col]
            if op_action == 1:
                cash_updated, n_stocks_bought = self.buy_stocks_(current_cash, stock_price)
                df_tmp.loc[idx_next, 'cash'] = cash_updated
                df_tmp.loc[idx_next, 'n_stocks'] = df_tmp.loc[idx, 'n_stocks'] + n_stocks_bought
                if n_stocks_bought == 0:
                    df_tmp.loc[idx, 'op'] = 0
            elif op_action == 2:
                cash_updated, n_stocks_sold = self.sell_stocks_(current_cash, current_n_stocks, stock_price)
                df_tmp.loc[idx_next, 'cash'] = cash_updated
                df_tmp.loc[idx_next, 'n_stocks'] = df_tmp.loc[idx, 'n_stocks'] - n_stocks_sold
                if n_stocks_sold == 0:
                    df_tmp.loc[idx, 'op'] = 0
            else:
                df_tmp.loc[idx_next, 'cash'] = df_tmp.loc[idx, 'cash']
                df_tmp.loc[idx_next, 'n_stocks'] = df_tmp.loc[idx, 'n_stocks']
            # Updating the wealth value:
            df_tmp.loc[idx, 'wealth'] = df_tmp.loc[idx, 'cash'] + df_tmp.loc[idx, 'n_stocks'] * df_tmp.loc[idx, self.close_col]

        # Updating the last row:
        df_tmp.loc[idx_next, 'wealth'] = df_tmp.loc[idx_next, 'cash'] + df_tmp.loc[idx_next, 'n_stocks'] * df_tmp.loc[idx, self.close_col]
        self.op_results = df_tmp
        return df_tmp
    
    def buy_stocks_(self, cash_value, stock_price):
        cash_available = self.daily_negotiable_perc * cash_value
        n_stocks_available = cash_available/stock_price
        if n_stocks_available > self.min_stocks_op:
            n_stocks_bought = int(n_stocks_available/self.min_stocks_op) * self.min_stocks_op
        else:
            n_stocks_bought = 0
        cash_updated = cash_value - (n_stocks_bought * stock_price) - self.broker_taxes
        return cash_updated, n_stocks_bought

    def sell_stocks_(self, cash_value, n_stocks, stock_price):
        n_stocks_available = int(self.daily_negotiable_perc * n_stocks)
        if n_stocks_available > self.min_stocks_op:
            n_stocks_sold = int(n_stocks_available/self.min_stocks_op) * self.min_stocks_op
        else:
            n_stocks_sold = 0
        cash_updated = cash_value + (n_stocks_sold * stock_price) - self.broker_taxes
        return cash_updated, n_stocks_sold

    def plot_wealth(self, figsize=(12, 8), grad_threshold=10):
        init_w = round(self.op_results['wealth'].values[0], 2)
        final_w = round(self.op_results['wealth'].values[-1], 2)
        variation_perc_ = round(100*(self.op_results['wealth'].values[-1] - self.op_results['wealth'].values[0]) / self.op_results['wealth'].values[0], 2)
        variation_base_perc_ = round(100*(self.op_results['baseline_wealth'].values[-2] - self.op_results['baseline_wealth'].values[0]) / self.op_results['baseline_wealth'].values[0], 2)
        plt.figure(figsize=figsize)
        plt.title(f"Initial wealth: {init_w}; Final wealth: {final_w}\nModel gain: {variation_perc_}% Base gain: {variation_base_perc_}%")
        self.op_results['wealth'].plot(label='wealth', zorder=2)
        plt.plot(self.op_results['n_stocks'] * self.op_results[self.close_col], label='stock value', zorder=2)
        self.op_results['cash'].plot(label='cash', zorder=2)
        self.op_results['baseline_wealth'].plot(label='baseline', color='k', alpha=0.5, zorder=2)
        plt.xlim([self.op_results.index.min(), self.op_results.index.max()])
        plt.ylabel('$')
        plt.legend(loc='best', prop={'size': 10}, ncol=2)
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_operations(self, figsize=(12, 10)):
        plt.figure(figsize=figsize)
        plt.subplot(3, 1, 1)
        self.op_results[self.price_ref_col].plot(label=self.price_ref_col, zorder=2)
        plt.scatter(self.op_results[self.op_results['op'] == 0].index, self.op_results[self.op_results['op'] == 0][self.price_ref_col], label='hold', alpha=0.5, zorder=3)
        plt.scatter(self.op_results[self.op_results['op'] == 1].index, self.op_results[self.op_results['op'] == 1][self.price_ref_col], label='buy', alpha=0.5, zorder=3)
        plt.scatter(self.op_results[self.op_results['op'] == 2].index, self.op_results[self.op_results['op'] == 2][self.price_ref_col], label='sell', alpha=0.5, zorder=3)
        plt.xlim([self.op_results.index.min(), self.op_results.index.max()])
        plt.ylabel('stock price [$]')
        plt.legend(loc='best', prop={'size': 10}, ncol=2)
        plt.grid()
        plt.subplot(3, 1, 2)
        self.op_results['macd'].plot(label='macd', zorder=2)
        self.op_results['signal'].plot(label='signal', zorder=2)
        plt.bar(self.op_results.index, self.op_results['histogram'], label='Histogram', zorder=2)
        plt.xlim([self.op_results.index.min(), self.op_results.index.max()])
        plt.ylabel('Trend indicators')
        plt.legend(loc='best', prop={'size': 10}, ncol=2)
        plt.grid()
        plt.subplot(3, 1, 3)
        self.op_results['williams_r'].plot(label='williams_r', zorder=2)
        plt.xlim([self.op_results.index.min(), self.op_results.index.max()])
        plt.ylabel('Williams %R')
        plt.grid()
        plt.tight_layout()
        plt.show()
