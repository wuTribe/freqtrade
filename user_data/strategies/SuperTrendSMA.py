from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np  # noqa
import pandas as pd  # noqa
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame
from sympy.physics.units import volume

from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_absolute)


# 计算 Chandelier Exit 指标
def chandelier_exit(dataframe, atr_period=22, atr_multiplier=3.0, use_close=True):
    # 计算 ATR
    dataframe['atr'] = ta.ATR(dataframe, timeperiod=atr_period).round(3)

    # 计算最高和最低点
    if use_close:
        highest = dataframe['close'].rolling(window=atr_period).max()
        lowest = dataframe['close'].rolling(window=atr_period).min()
    else:
        highest = dataframe['high'].rolling(window=atr_period).max()
        lowest = dataframe['low'].rolling(window=atr_period).min()

    # 计算初始 long 和 short 停损线
    dataframe['long_stop'] = highest - dataframe['atr'] * atr_multiplier
    dataframe['short_stop'] = lowest + dataframe['atr'] * atr_multiplier

    # 创建临时列表以逐步更新 `long_stop` 和 `short_stop`
    long_stops = [dataframe['long_stop'].iloc[0]]
    short_stops = [dataframe['short_stop'].iloc[0]]

    # 循环逐行更新 `long_stop` 和 `short_stop`
    for i in range(1, len(dataframe)):
        prev_long_stop = long_stops[-1]  # 前一行的 long_stop
        prev_short_stop = short_stops[-1]  # 前一行的 short_stop

        # 更新 long_stop
        current_long_stop = dataframe['long_stop'].iloc[i]
        if dataframe['close'].iloc[i - 1] > prev_long_stop:
            long_stops.append(np.maximum(current_long_stop, prev_long_stop))
        else:
            long_stops.append(current_long_stop)

        # 更新 short_stop
        current_short_stop = dataframe['short_stop'].iloc[i]
        if dataframe['close'].iloc[i - 1] < prev_short_stop:
            short_stops.append(np.minimum(current_short_stop, prev_short_stop))
        else:
            short_stops.append(current_short_stop)

    # 将更新后的 long_stops 和 short_stops 列赋值回 dataframe
    dataframe['long_stop'] = long_stops
    dataframe['short_stop'] = short_stops

    # 确定方向
    dataframe['dir'] = np.where(
        dataframe['close'] > dataframe['short_stop'], 1,
        np.where(dataframe['close'] < dataframe['long_stop'], -1, np.nan)
    )
    dataframe['dir'] = dataframe['dir'].ffill().fillna(1)  # 初始方向设为1

    # 生成买卖信号
    dataframe['buy_signal'] = (dataframe['dir'] == 1) & (dataframe['dir'].shift(1) == -1)
    dataframe['sell_signal'] = (dataframe['dir'] == -1) & (dataframe['dir'].shift(1) == 1)

    # 清理临时列
    # dataframe.drop(columns=['long_stop_prev', 'short_stop_prev'], inplace=True)

    return dataframe

class SuperTrendSMA(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 100  # inactive
    }

    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True

    timeframe = '5m'

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
            'long_stop': {'color': '#26a69a', 'width': 2},
            'short_stop': {'color': '#ef5350', 'width': 2},
        },
        'subplots': {
            "obv": {
                'obv': {'color': '#26a69a', 'width': 2},
                'obv_sma': {'color': '#ef5350', 'width': 2},
            }
        }
    }

    fee_rate = 0.001
    fixed_loss_amount = 100

    def bot_start(self, **kwargs) -> None:
        # 从配置文件中读取手续费，如果配置中没有 "fee" 则默认 0.1%
        self.fee_rate = self.config.get("fee", 0.001)
        # 设定每次交易的固定亏损金额，例如 100 美元
        self.fixed_loss_amount = self.config.get("dry_run_wallet", 400) / 400
        pass

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # 获取分析过的 dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # 根据入场位置计算的目标点位出场
        minutes_ = int((trade.open_date - dataframe['date'].iloc[0]) / timedelta(minutes=15)) - 1
        if minutes_ < 0:
            return self.stoploss
        stop_loss_price = dataframe.iloc[minutes_]['stop_loss_price_15m']

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
        sma_timeperiod = 21
        informative['sma21'] = ta.SMA(informative, timeperiod=sma_timeperiod)
        informative['sma21_slope'] = informative['sma21'] - informative['sma21'].shift(1)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        inf_tf = '4h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['sma21'] = ta.SMA(informative, timeperiod=sma_timeperiod)
        informative['sma21_slope'] = informative['sma21'] - informative['sma21'].shift(1)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)


        inf_tf = '15m'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        sma_timeperiod = 21
        informative['sma21'] = ta.SMA(informative, timeperiod=sma_timeperiod)
        informative['sma21_slope'] = informative['sma21'] - informative['sma21'].shift(1)
        # 超级趋势计算
        informative = chandelier_exit(informative, atr_period=22, atr_multiplier=3.0, use_close=True)
        # 设置买卖信号
        informative['buy_signal'] = informative['buy_signal']
        informative['sell_signal'] = informative['sell_signal']

        # 计算每笔交易的总手续费预算
        stop_loss_atr_multiplier = 4.0

        # 做多逻辑
        long_entry_price = informative['close']
        long_stop_loss_price = long_entry_price - informative['atr'] * stop_loss_atr_multiplier  # 做多止损价格
        long_stop_loss_percentage = abs((long_entry_price - long_stop_loss_price) / long_entry_price)
        long_investment_amount = self.fixed_loss_amount / (long_stop_loss_percentage + self.fee_rate * 2)

        # 手续费调整后的止盈价格
        long_take_profit_percentage = (self.fixed_loss_amount * 1.5 / long_investment_amount) + (self.fee_rate * 2)
        long_take_profit_price_adj = long_entry_price * (1 + long_take_profit_percentage)

        # 做空逻辑
        short_entry_price = informative['close']
        short_stop_loss_price = short_entry_price + informative['atr'] * stop_loss_atr_multiplier  # 做空止损价格
        short_stop_loss_percentage = abs((short_stop_loss_price - short_entry_price) / short_entry_price)
        short_investment_amount = self.fixed_loss_amount / (long_stop_loss_percentage + self.fee_rate * 2)

        # 手续费调整后的止盈价格
        short_take_profit_percentage = (self.fixed_loss_amount * 1.5 / short_investment_amount) + (self.fee_rate * 2)
        short_take_profit_price_adj = short_entry_price * (1 - short_take_profit_percentage)

        # 将计算结果添加到 DataFrame
        # 做多信号时的各项指标
        informative.loc[informative['buy_signal'], 'investment_amount'] = long_investment_amount // 1
        informative.loc[informative['buy_signal'], 'stop_loss_price'] = long_stop_loss_price
        informative.loc[informative['buy_signal'], 'stop_loss_percentage'] = long_stop_loss_percentage
        informative.loc[informative['buy_signal'], 'take_profit_price'] = long_take_profit_price_adj
        informative.loc[informative['buy_signal'], 'take_profit_percentage'] = long_take_profit_percentage

        # 做空信号时的各项指标
        informative.loc[informative['sell_signal'], 'investment_amount'] = short_investment_amount // 1
        informative.loc[informative['sell_signal'], 'stop_loss_price'] = short_stop_loss_price
        informative.loc[informative['sell_signal'], 'stop_loss_percentage'] = short_stop_loss_percentage
        informative.loc[informative['sell_signal'], 'take_profit_price'] = short_take_profit_price_adj
        informative.loc[informative['sell_signal'], 'take_profit_percentage'] = short_take_profit_percentage
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 设置斜率的最小值，只有当斜率大于某个阈值时才买入
        slope_threshold = 0.1  # 根据需要调整斜率阈值
        dataframe.loc[
            (
                    (1 > 0)
                    # & (dataframe['atr'] >= 2)
                    & (dataframe['sma21_slope_4h'] > 0) # 判断 sma21_1h 是否有足够的上升斜率
                    & (dataframe['buy_signal_15m'])  # Supertrend 买入信号
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
        return current_candle['investment_amount_15m']

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                    **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # 根据入场位置计算的目标点位出场
        minutes_ = int((trade.open_date - dataframe['date'].iloc[0]) / timedelta(minutes=15)) - 1
        if minutes_ < 0:
            take_profit_price = dataframe.iloc[-1].squeeze()['take_profit_price_15m']
        else:
            take_profit_price = dataframe.iloc[minutes_]['take_profit_price_15m']
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
