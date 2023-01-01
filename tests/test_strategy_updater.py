# pragma pylint: disable=missing-docstring, protected-access, invalid-name

from freqtrade.strategy.strategyupdater import StrategyUpdater


def test_strategy_updater(default_conf, caplog) -> None:
    modified_code2 = StrategyUpdater.update_code(StrategyUpdater, """
ticker_interval = '15m'
buy_some_parameter = IntParameter(space='buy')
sell_some_parameter = IntParameter(space='sell')
""")
    modified_code1 = StrategyUpdater.update_code(StrategyUpdater, """
class testClass(IStrategy):
    def populate_buy_trend():
        pass
    def populate_sell_trend():
        pass
    def check_buy_timeout():
        pass
    def check_sell_timeout():
        pass
    def custom_sell():
        pass
""")
    modified_code3 = StrategyUpdater.update_code(StrategyUpdater, """
use_sell_signal = True
sell_profit_only = True
sell_profit_offset = True
ignore_roi_if_buy_signal = True
forcebuy_enable = True
""")
    modified_code4 = StrategyUpdater.update_code(StrategyUpdater, """
dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")
dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
""")
    modified_code5 = StrategyUpdater.update_code(StrategyUpdater, """
def confirm_trade_exit(sell_reason: str):
    pass
    """)
    modified_code6 = StrategyUpdater.update_code(StrategyUpdater, """
order_time_in_force = {
    'buy': 'gtc',
    'sell': 'ioc'
}
order_types = {
    'buy': 'limit',
    'sell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
}
unfilledtimeout = {
    'buy': 1,
    'sell': 2
}
""")

    modified_code7 = StrategyUpdater.update_code(StrategyUpdater, """
def confirm_trade_exit(sell_reason):
    if (sell_reason == 'stop_loss'):
        pass
""")

    assert "populate_entry_trend" in modified_code1
    assert "populate_exit_trend" in modified_code1
    assert "check_entry_timeout" in modified_code1
    assert "check_exit_timeout" in modified_code1
    assert "custom_exit" in modified_code1
    assert "INTERFACE_VERSION = 3" in modified_code1

    assert "timeframe" in modified_code2
    # check for not editing hyperopt spaces
    assert "space='buy'" in modified_code2
    assert "space='sell'" in modified_code2

    assert "use_exit_signal" in modified_code3
    assert "exit_profit_only" in modified_code3
    assert "exit_profit_offset" in modified_code3
    assert "ignore_roi_if_entry_signal" in modified_code3
    assert "force_entry_enable" in modified_code3

    assert "enter_long" in modified_code4
    assert "exit_long" in modified_code4
    assert "enter_tag" in modified_code4

    assert "exit_reason" in modified_code5

    assert "'entry': 'gtc'" in modified_code6
    assert "'exit': 'ioc'" in modified_code6
    assert "'entry': 'limit'" in modified_code6
    assert "'exit': 'market'" in modified_code6
    assert "'entry': 1" in modified_code6
    assert "'exit': 2" in modified_code6

    assert "exit_reason" in modified_code7
    assert "exit_reason == 'stop_loss'" in modified_code7
