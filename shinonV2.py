import logging
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from pandas import DataFrame

logger = logging.getLogger(__name__)

class ShinonV2(IStrategy):
    # Timeframe dan Risk Management
    timeframe = '15m'
    stoploss = -0.15  # Fallback, tidak dipakai karena custom_stoploss aktif
    trailing_stop = True
    trailing_stop_positive = 0.07
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True  # Mengaktifkan custom stoploss dinamis
    minimal_roi = {"0": 100}  # Exit didasarkan pada custom_exit

    # Hyperparameters untuk optimasi
    bb_length = IntParameter(10, 30, default=20, space='buy')
    bb_dev = DecimalParameter(1.4, 2.2, default=1.8, space='buy')
    volume_filter = DecimalParameter(1.0, 1.5, default=1.1, space='buy')
    trend_filter = CategoricalParameter([True, False], default=True, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bb_length = self.bb_length.value
        bb_dev = self.bb_dev.value
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=bb_length, stds=bb_dev)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']

        # ATR sebagai pengukur volatilitas
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Filter Volume
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)

        # Filter Trend: ADX dan Directional Indicators
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Entry Long: harga harus berada di bawah bb_lower (tanpa toleransi) dan volume minimal 10% di atas MA volume.
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['volume'] > dataframe['volume_ma'] * self.volume_filter.value) &
                ((dataframe['adx'] > 30) if self.trend_filter.value else True) &
                (dataframe['plus_di'] > dataframe['minus_di'])
            ),
            'enter_long'
        ] = 1

        # Entry Short: harga harus berada di atas bb_upper (tanpa toleransi) dan volume minimal 10% di atas MA volume.
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['volume'] > dataframe['volume_ma'] * self.volume_filter.value) &
                ((dataframe['adx'] > 30) if self.trend_filter.value else True) &
                (dataframe['minus_di'] > dataframe['plus_di'])
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit tambahan jika harga sudah melewati bb_middle dan terjadi sinyal pembalikan momentum
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middle']) &
                (qtpylib.crossed_above(dataframe['minus_di'], dataframe['plus_di']))
            ),
            'exit_long'
        ] = 1

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_middle']) &
                (qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di']))
            ),
            'exit_short'
        ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Custom Stop Loss berbasis ATR.
        Untuk trade long, SL dihitung sebagai ATR relatif terhadap harga (atr/current_rate)
        dengan batas antara 3% (min) dan 7% (max). Untuk trade short, nilai SL positif.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_row = dataframe.iloc[-1]
        atr = last_row['atr']
        risk_pct = atr / current_rate
        # Batasan stop loss: minimum 3%, maksimum 7%
        min_sl = 0.03
        max_sl = 0.07
        dynamic_sl = max(min_sl, min(risk_pct, max_sl))

        if trade.is_short:
            return dynamic_sl
        else:
            return -dynamic_sl

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Custom Take Profit dinamis berbasis ATR.
        Dengan rasio risk reward 4:1, target profit ditentukan sebagai 4 kali nilai stop loss dinamis.
        Batas target profit ditetapkan antara 5% dan 15%.
        Trade akan di-exit jika current_profit mencapai atau melebihi target tersebut.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return False

        last_row = dataframe.iloc[-1]
        atr = last_row['atr']
        risk_pct = atr / current_rate

        # Target profit: 4 kali nilai stop loss dinamis
        target_profit = 4 * max(0.03, min(risk_pct, 0.07))
        # Batas target profit: minimal 5% dan maksimal 15%
        target_profit = max(0.05, min(target_profit, 0.15))

        if current_profit >= target_profit:
            return True
        return False

    def get_ticker_info(self, pair: str):
        """Contoh akses data tambahan"""
        _, df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return df.iloc[-1] if df is not None else None
