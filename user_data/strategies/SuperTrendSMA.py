
import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pda
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (DecimalParameter,
                                IStrategy)


# This class is a sample. Feel free to customize it.
class SuperTrendSMA(IStrategy):
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100  # inactive
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99  # inactive

    # Trailing stoploss
    trailing_stop = False

    buy_stoch_rsi = DecimalParameter(0.5, 1, decimals=3, default=0.8, space="buy")
    sell_stoch_rsi = DecimalParameter(0, 0.5, decimals=3, default=0.2, space="sell")

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

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
            'ema90': {},
            'supertrend_1': {},
            'supertrend_2': {},
            'supertrend_3': {},
        },
        'subplots': {
            "SUPERTREND DIRECTION": {
                'supertrend_direction_1': {},
                'supertrend_direction_2': {},
                'supertrend_direction_3': {},
            },
            "STOCH RSI": {
                'stoch_rsi': {},
            }
        }
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Momentum Indicators
        # ------------------------------------

        # # Stochastic RSI
        dataframe['stoch_rsi'] = ta.momentum.stochrsi(dataframe['close'])

        # Overlap Studies
        # ------------------------------------

        # # EMA - Exponential Moving Average
        dataframe['ema90'] = ta.trend.ema_indicator(dataframe['close'], 90)

        # Supertrend
        supertrend_length = 20
        supertrend_multiplier = 3.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length,
                                    multiplier=supertrend_multiplier)
        dataframe['supertrend_1'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_1'] = superTrend[
            'SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]

        supertrend_length = 20
        supertrend_multiplier = 4.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length,
                                    multiplier=supertrend_multiplier)
        dataframe['supertrend_2'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_2'] = superTrend[
            'SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]

        supertrend_length = 40
        supertrend_multiplier = 8.0
        superTrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=supertrend_length,
                                    multiplier=supertrend_multiplier)
        dataframe['supertrend_3'] = superTrend['SUPERT_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]
        dataframe['supertrend_direction_3'] = superTrend[
            'SUPERTd_' + str(supertrend_length) + "_" + str(supertrend_multiplier)]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    ((dataframe['supertrend_direction_1'] + dataframe['supertrend_direction_2'] + dataframe[
                        'supertrend_direction_3']) >= 1) &
                    (dataframe['stoch_rsi'] < self.buy_stoch_rsi.value) &
                    (dataframe['close'] > dataframe['ema90']) &
                    (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    ((dataframe['supertrend_direction_1'] + dataframe['supertrend_direction_2'] + dataframe[
                        'supertrend_direction_3']) < 1) &
                    (dataframe['stoch_rsi'] > self.sell_stoch_rsi.value) &
                    (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe