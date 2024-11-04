import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pda
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (IStrategy, merge_informative_pair)


class SuperTrendSMA(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 100  # inactive
    }

    stoploss = -0.99  # inactive

    trailing_stop = False

    timeframe = '15m'

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
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
            'sma21_1h': {'color': 'blue', 'width': 2},
            'supertrend': {'color': 'green', 'width': 2},
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
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['sma21'] = ta.SMA(informative, timeperiod=21)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # 使用 pandas_ta 计算 Supertrend
        supertrend_length = 22
        supertrend_multiplier = 3.0
        supertrend = pda.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                    length=supertrend_length, multiplier=supertrend_multiplier)

        # 将 Supertrend 值和方向添加到 DataFrame
        dataframe['supertrend'] = supertrend[f'SUPERT_{supertrend_length}_{supertrend_multiplier}']
        dataframe['supertrend_direction'] = supertrend[f'SUPERTd_{supertrend_length}_{supertrend_multiplier}']

        # 在趋势首次变化时生成买卖信号
        dataframe['buy_signal'] = (dataframe['supertrend_direction'] == 1) & (dataframe['supertrend_direction'].shift(1) == -1)
        dataframe['sell_signal'] = (dataframe['supertrend_direction'] == -1) & (dataframe['supertrend_direction'].shift(1) == 1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['buy_signal'])  # Supertrend 买入信号
                    & (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['sell_signal'])  # Supertrend 卖出信号
                    & (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
