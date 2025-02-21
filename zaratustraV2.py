import logging
import numpy as np
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime, timezone
from typing import Optional
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

class ZaratustraV2(IStrategy):
    """
    Enhanced version of Zaratustra strategy with improved risk management,
    dynamic indicators, and market regime filters.
    """

    # Strategy parameters
    minimal_roi = {
        "0": 0.253,
        "16": 0.075,
        "70": 0.025,
        "112": 0
    }

    exit_profit_only = True
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.2
    trailing_stop_positive_offset = 0.276
    trailing_only_offset_is_reached = False
    ignore_roi_if_entry_signal = True
    can_short = True
    use_exit_signal = True
    stoploss = -0.349
    startup_candle_count: int = 200
    timeframe = '5m'
    process_only_new_candles = True

    # DCA Parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 1.5

    # Hyperoptable parameters
    cooldown_lookback = IntParameter(2, 48, default=3, space="protection", optimize=True)
    stop_duration = IntParameter(12, 240, default=224, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=False, space="protection", optimize=True)
    adx_threshold = IntParameter(20, 40, default=21, space="buy", optimize=True)
    rsi_threshold = IntParameter(30, 70, default=64, space="buy", optimize=True)
    atr_multiplier = DecimalParameter(1.0, 5.0, default=4.097, space="sell", optimize=True)

    @property
    def protections(self):
        prot = []
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })
        return prot

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        adjusted_stake = proposed_stake / self.max_dca_multiplier
        return max(adjusted_stake, min_stake)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        atr = dataframe['atr'].iloc[-1]
        
        if trade.entry_side == "buy":
            if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if current_profit < -0.05 * (atr/current_rate):
                return trade.stake_amount * 0.6
        else:
            if current_profit > 0.10 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 2)
            if current_profit < -0.05 * (atr/current_rate):
                return trade.stake_amount * 0.6
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Basic Indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # Market Trend
        dataframe['market_trend'] = np.where(dataframe['close'] > dataframe['ema_200'], 1, -1)
        
        # ADX System
        dataframe['dx'] = ta.SMA(ta.DX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['adx'] = ta.SMA(ta.ADX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['pdi'] = ta.SMA(ta.PLUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['mdi'] = ta.SMA(ta.MINUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        
        # Volatility Filter
        dataframe['volatility'] = dataframe['atr'] / dataframe['close']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry Conditions
        dataframe.loc[
            (
                (dataframe['market_trend'] == 1) &
                (qtpylib.crossed_above(dataframe['dx'], dataframe['pdi'])) &
                (dataframe['adx'] > self.adx_threshold.value) &
                (dataframe['pdi'] > dataframe['mdi']) &
                (dataframe['rsi'] > self.rsi_threshold.value) &
                (dataframe['volatility'] < 0.05)  # Max 5% volatility
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Long')

        # Short Entry Conditions
        dataframe.loc[
            (
                (dataframe['market_trend'] == -1) &
                (qtpylib.crossed_above(dataframe['dx'], dataframe['mdi'])) &
                (dataframe['adx'] > self.adx_threshold.value) &
                (dataframe['mdi'] > dataframe['pdi']) &
                (dataframe['rsi'] < (100 - self.rsi_threshold.value)) &
                (dataframe['volatility'] < 0.05)  # Max 5% volatility
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'ZaratustraDCA Entry Short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Dynamic Stop Loss
        dataframe['stoploss'] = dataframe['atr'] * self.atr_multiplier.value

        # Exit Long
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['dx'], dataframe['adx'])) &
                (dataframe['adx'] > self.adx_threshold.value)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Long')

        # Exit Short
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['dx'], dataframe['adx'])) &
                (dataframe['adx'] <= self.adx_threshold.value)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'ZaratustraDCA Exit Short')

        return dataframe  # Pastikan ini ada

