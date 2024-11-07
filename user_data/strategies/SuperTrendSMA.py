from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pda
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade, Order
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_absolute)


class SuperTrendSMA(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 100  # inactive
    }

    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True

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
        "entry": "GTC",
        "exit": "GTC"
    }

    plot_config = {
        'main_plot': {
            'sma21_1h': {'color': 'blue', 'width': 2},
            'supertrend': {'color': 'black', 'width': 2},
            'take_profit_price': {'color': 'green', 'width': 2},
            'stop_loss_price': {'color': 'red', 'width': 2},
        },
        'subplots': {
        }
    }

    # 每次都会调用
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     # 获取分析过的 dataframe
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #     current_candle = dataframe.iloc[-1]  # 获取当前最新的蜡烛数据
    #
    #     # 获取止损价格（假设为当前蜡烛的 take_profit_price）
    #     short = trade.is_short
    #     stop_rate = current_rate
    #     if short :
    #         stop_rate += current_candle['atr'] * 3
    #     else:
    #         stop_rate -= current_candle['atr'] * 3
    #
    #     # 使用 stoploss_from_absolute 计算止损百分比
    #     stop_loss_percentage = stoploss_from_absolute(stop_rate=stop_rate, current_rate=current_rate, is_short=short)
    #
    #     # 返回止损百分比，负值表示下跌时触发止损
    #     return -abs(stop_loss_percentage)
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # 获取分析过的 dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # 根据入场位置计算的目标点位出场
        stop_loss_price = dataframe.iloc[int((trade.open_date - dataframe['date'].iloc[0]) / timedelta(minutes=15)) - 1]['stop_loss_price']

        # 使用 stoploss_from_absolute 计算止损百分比
        stop_loss_percentage = stoploss_from_absolute(stop_rate=stop_loss_price, current_rate=current_rate, is_short=trade.is_short)

        # 返回止损百分比，负值表示下跌时触发止损
        return -abs(stop_loss_percentage)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        # 为每对交易对分配tf，以便可以为策略下载和缓存它们。
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['sma21'] = ta.SMA(informative, timeperiod=21)
        informative['sma21_slope'] = informative['sma21'] - informative['sma21'].shift(1)
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
        dataframe['buy_signal'] = (dataframe['supertrend_direction'] == 1) & (
                dataframe['supertrend_direction'].shift(1) == -1)
        dataframe['sell_signal'] = (dataframe['supertrend_direction'] == -1) & (
                dataframe['supertrend_direction'].shift(1) == 1)

        # 计算 ATR
        atr_period = 22
        stop_loss_atr_multiplier = 4.0
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period)

        # 从配置文件中读取手续费，如果配置中没有 "fee" 则默认 0.1%
        fee_rate = self.config.get("fee", 0.001)
        # 设定每次交易的固定亏损金额，例如 100 美元
        fixed_loss_amount = 100.0
        # 计算每笔交易的总手续费预算

        # 做多逻辑
        long_entry_price = dataframe['close']
        long_stop_loss_price = long_entry_price - dataframe['atr'] * stop_loss_atr_multiplier  # 做多止损价格
        long_stop_loss_percentage = abs((long_entry_price - long_stop_loss_price) / long_entry_price)
        long_investment_amount = fixed_loss_amount / (long_stop_loss_percentage + fee_rate * 2)

        # 手续费调整后的止盈价格
        long_take_profit_percentage = (fixed_loss_amount * 1.5 / long_investment_amount) + (fee_rate * 2)
        long_take_profit_price_adj = long_entry_price * (1 + long_take_profit_percentage)

        # 做空逻辑
        short_entry_price = dataframe['close']
        short_stop_loss_price = short_entry_price + dataframe['atr'] * stop_loss_atr_multiplier  # 做空止损价格
        short_stop_loss_percentage = abs((short_stop_loss_price - short_entry_price) / short_entry_price)
        short_investment_amount = fixed_loss_amount / (long_stop_loss_percentage + fee_rate * 2)

        # 手续费调整后的止盈价格
        short_take_profit_percentage = (fixed_loss_amount * 1.5 / short_investment_amount) + (fee_rate * 2)
        short_take_profit_price_adj = short_entry_price * (1 - short_take_profit_percentage)

        # 将计算结果添加到 DataFrame
        # 做多信号时的各项指标
        dataframe.loc[dataframe['buy_signal'], 'investment_amount'] = long_investment_amount // 1
        dataframe.loc[dataframe['buy_signal'], 'stop_loss_price'] = long_stop_loss_price
        dataframe.loc[dataframe['buy_signal'], 'stop_loss_percentage'] = long_stop_loss_percentage
        dataframe.loc[dataframe['buy_signal'], 'take_profit_price'] = long_take_profit_price_adj
        dataframe.loc[dataframe['buy_signal'], 'take_profit_percentage'] = long_take_profit_percentage

        # 做空信号时的各项指标
        dataframe.loc[dataframe['sell_signal'], 'investment_amount'] = short_investment_amount // 1
        dataframe.loc[dataframe['sell_signal'], 'stop_loss_price'] = short_stop_loss_price
        dataframe.loc[dataframe['sell_signal'], 'stop_loss_percentage'] = short_stop_loss_percentage
        dataframe.loc[dataframe['sell_signal'], 'take_profit_price'] = short_take_profit_price_adj
        dataframe.loc[dataframe['sell_signal'], 'take_profit_percentage'] = short_take_profit_percentage

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 设置斜率的最小值，只有当斜率大于某个阈值时才买入
        slope_threshold = 0.1  # 根据需要调整斜率阈值
        dataframe.loc[
            (
                    (1 > 0)
                    # & (dataframe['atr'] >= 2)
                    # & (dataframe['sma21_slope_1h'] > slope_threshold) # 判断 sma21_1h 是否有足够的上升斜率
                    & (dataframe['buy_signal'])  # Supertrend 买入信号
                    & (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    # 首次入场调用
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        return current_candle['investment_amount']


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                    **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # 根据入场位置计算的目标点位出场
        take_profit_price = dataframe.iloc[int((trade.open_date - dataframe['date'].iloc[0]) / timedelta(minutes=15)) - 1]['take_profit_price']
        if trade.is_short:
            if take_profit_price >= current_rate:
                return 'short_target_price'
        else:
            if take_profit_price <= current_rate:
                return 'long_target_price'
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, entry_tag: Optional[str], side: str, **kwargs,
                 ) -> float:
        return 1.0
