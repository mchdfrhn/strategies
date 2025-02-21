import logging
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta

logger = logging.getLogger(__name__)

class ShinonV1(IStrategy):
    """
    Strategy ini menggunakan EMA untuk menentukan trend dan mengonfirmasi trend tersebut
    menggunakan timeframe yang lebih tinggi (informative timeframe).

    Timeframe utama: 5m
    Timeframe konfirmasi: 1h

    Entry Conditions:
      - Long Entry:
          * Main timeframe: close > EMA13, close > EMA21, dan EMA13 > EMA21.
          * Higher timeframe: informative_1h_close > informative_1h_ema13, 
            informative_1h_close > informative_1h_ema21, dan informative_1h_ema13 > informative_1h_ema21.
      - Short Entry:
          * Main timeframe: close < EMA13, close < EMA21, dan EMA13 < EMA21.
          * Higher timeframe: informative_1h_close < informative_1h_ema13,
            informative_1h_close < informative_1h_ema21, dan informative_1h_ema13 < informative_1h_ema21.

    Exit Conditions:
      - Long exit: Jika harga turun di bawah EMA13 pada main timeframe.
      - Short exit: Jika harga naik di atas EMA13 pada main timeframe.
    """

    timeframe = '5m'
    informative_timeframe = '1h'
    startup_candle_count: int = 50

    minimal_roi = {
        "0": 0.05,
        "10": 0.03,
        "20": 0.01,
        "30": 0
    }
    stoploss = -0.03
    trailing_stop = False

    can_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Hitung EMA pada timeframe utama menggunakan kolom 'close'
        dataframe['ema13'] = ta.EMA(dataframe['close'], timeperiod=13)
        dataframe['ema21'] = ta.EMA(dataframe['close'], timeperiod=21)

        # Ambil dataframe dari timeframe yang lebih tinggi (informative timeframe)
        informative, _ = self.dp.get_analyzed_dataframe(metadata['pair'], self.informative_timeframe)
        
        # Pastikan kolom 'close' ada. Jika tidak, ubah nama kolom menjadi lowercase.
        if 'close' not in informative.columns:
            informative.columns = [x.lower() for x in informative.columns]

        # Hitung EMA pada informative timeframe menggunakan kolom 'close'
        informative['ema13'] = ta.EMA(informative['close'], timeperiod=13)
        informative['ema21'] = ta.EMA(informative['close'], timeperiod=21)

        # Gabungkan informative dataframe ke dataframe utama
        dataframe = merge_informative_pair(
            dataframe, 
            informative, 
            self.timeframe, 
            self.informative_timeframe, 
            ffill=True
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry pada main timeframe
        conditions_long = []
        conditions_long.append(dataframe['close'] > dataframe['ema13'])
        conditions_long.append(dataframe['close'] > dataframe['ema21'])
        conditions_long.append(dataframe['ema13'] > dataframe['ema21'])
        # Konfirmasi dari timeframe lebih tinggi
        conditions_long.append(dataframe['informative_1h_close'] > dataframe['informative_1h_ema13'])
        conditions_long.append(dataframe['informative_1h_close'] > dataframe['informative_1h_ema21'])
        conditions_long.append(dataframe['informative_1h_ema13'] > dataframe['informative_1h_ema21'])

        dataframe.loc[
            np.all(conditions_long, axis=0),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long: EMA trend confirmed on main & 1h')

        # Short Entry pada main timeframe
        conditions_short = []
        conditions_short.append(dataframe['close'] < dataframe['ema13'])
        conditions_short.append(dataframe['close'] < dataframe['ema21'])
        conditions_short.append(dataframe['ema13'] < dataframe['ema21'])
        # Konfirmasi dari timeframe lebih tinggi
        conditions_short.append(dataframe['informative_1h_close'] < dataframe['informative_1h_ema13'])
        conditions_short.append(dataframe['informative_1h_close'] < dataframe['informative_1h_ema21'])
        conditions_short.append(dataframe['informative_1h_ema13'] < dataframe['informative_1h_ema21'])

        dataframe.loc[
            np.all(conditions_short, axis=0),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short: EMA trend confirmed on main & 1h')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit untuk posisi long: harga turun di bawah EMA13 (main timeframe)
        exit_long_condition = dataframe['close'] < dataframe['ema13']
        dataframe.loc[
            exit_long_condition,
            ['exit_long', 'exit_tag']
        ] = (1, 'Exit Long: Price below EMA13')

        # Exit untuk posisi short: harga naik di atas EMA13 (main timeframe)
        exit_short_condition = dataframe['close'] > dataframe['ema13']
        dataframe.loc[
            exit_short_condition,
            ['exit_short', 'exit_tag']
        ] = (1, 'Exit Short: Price above EMA13')

        return dataframe
