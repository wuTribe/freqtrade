
import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pda
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (DecimalParameter,
                                IStrategy)


class SuperTrendSMA(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 100  # inactive
    }

    stoploss = -0.99  # inactive

    trailing_stop = False

    buy_stoch_rsi = DecimalParameter(0.5, 1, decimals=3, default=0.8, space="buy")
    sell_stoch_rsi = DecimalParameter(0, 0.5, decimals=3, default=0.2, space="sell")

    timeframe = '15m'

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 90

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'sma21': {},
            'supertrend_1': {},
        },
        'subplots': {
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        # 为每对交易对分配tf，以便可以为策略下载和缓存它们。
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']

        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)

        supertrend_length = 22
        supertrend_multiplier = 3.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length,
                                    multiplier=supertrend_multiplier)
        dataframe['supertrend_1'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_1'] = superTrend[
            'SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['sma21']) &
                    (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['fastk_rsi'] > self.sell_stoch_rsi.value) &
                    (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe