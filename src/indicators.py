from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators
import ta


def stock_preprocessing(df_data: pd.DataFrame, open_col: str = 'Abertura', close_col: str = 'Fech.',
                        min_col: str = 'Mínimo', max_col: str = 'Máximo', williams_r_lbp=14) -> pd.DataFrame:

    # MACD, Signal, Histogram:
    df_data['macd'] = ta.trend.macd(df_data[close_col])
    df_data['signal'] = ta.trend.macd_signal(df_data[close_col])
    df_data['histogram'] = ta.trend.macd_diff(df_data[close_col])
    # William %R
    df_data['williams_r'] = ta.momentum.williams_r(df_data[max_col], df_data[min_col],
                                                   df_data[close_col], lbp=williams_r_lbp)

    return df_data

def get_macd_signal_hist(close_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # MACD:
    exp1 = close_series.ewm(span=12, adjust=False).mean()
    exp2 = close_series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    # Signal 
    signal = macd.ewm(span=9, adjust=False).mean()
    # Histogram
    histogram = macd - signal

    return macd, signal, histogram
