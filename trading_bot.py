import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trading_bot')

class TechnicalIndicators:
    """Class to implement technical indicators without relying on TA-Lib"""
    
    @staticmethod
    def sma(data, period):
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(data, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD, MACD Signal and MACD Histogram"""
        fast_ema = TechnicalIndicators.ema(data, fast_period)
        slow_ema = TechnicalIndicators.ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def heikin_ashi(df):
        """Convert regular OHLC data to Heikin Ashi candles"""
        ha_df = pd.DataFrame(index=df.index)
        
        # Calculate Heikin Ashi values
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Initialize first Heikin Ashi open with first candle's open
        ha_df['ha_open'] = np.nan  # Initialize with NaN values
        ha_df.loc[ha_df.index[0], 'ha_open'] = df['open'].iloc[0]
        
        # Calculate ha_open for the rest of the candles using .loc instead of chained assignment
        for i in range(1, len(df)):
            ha_df.loc[ha_df.index[i], 'ha_open'] = (ha_df['ha_open'].iloc[i-1] + ha_df['ha_close'].iloc[i-1]) / 2
            
        ha_df['ha_high'] = df.apply(lambda x: max(x['high'], x['open'], x['close']), axis=1)
        ha_df['ha_low'] = df.apply(lambda x: min(x['low'], x['open'], x['close']), axis=1)
        
        return ha_df
    
    @staticmethod
    def atr(df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

class BinanceFuturesBot:
    def __init__(self, config_path='config.json'):
        """
        Initialize the Binance Futures Trading Bot
        
        Args:
            config_path (str): Path to the configuration file
        """
        try:
            self.load_config(config_path)
            self.client = Client(self.config['api_key'], self.config['api_secret'])
            
            # Set leverage immediately after initialization
            try:
                self.client.futures_change_leverage(
                    symbol=self.config['symbol'],
                    leverage=self.config.get('leverage', 1)  # Default to 1 if not set
                )
                logger.info(f"Leverage set to {self.config['leverage']}x")
            except Exception as e:
                logger.error(f"Error setting leverage: {e}")
                self.config['leverage'] = 1  # Set default leverage if failed
            
            self.positions = {}
            self.last_update_time = 0
            self.orders = []
            self.indicators = TechnicalIndicators()
            self.is_trading = False
            
            # Create necessary directories
            os.makedirs('data', exist_ok=True)
            
            logger.info(f"Bot initialized for symbol: {self.config['symbol']}")
        except Exception as e:
            logger.error(f"Error in initialization: {e}")
            raise
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as file:
                self.config = json.load(file)
                
            # Set defaults for any missing config values
            self.config.setdefault('symbol', 'STXUSDT')
            self.config.setdefault('leverage', 10)
            self.config.setdefault('stop_loss_percent', 1)
            self.config.setdefault('take_profit_min', 2)
            self.config.setdefault('take_profit_max', 3)
            self.config.setdefault('position_size_percent', 100)  # % of available balance
            self.config.setdefault('sma_period', 200)
            self.config.setdefault('use_heikin_ashi', True)
            self.config.setdefault('macd_fast', 12)
            self.config.setdefault('macd_slow', 26)
            self.config.setdefault('macd_signal', 9)
            self.config.setdefault('use_volatility_filter', True)
            self.config.setdefault('atr_period', 14)
            self.config.setdefault('atr_multiplier', 1.0)  # Para SL
            
            # Asegurar que tp_atr_multiplier sea al menos el doble de atr_multiplier
            min_tp_multiplier = self.config['atr_multiplier'] * 2
            default_tp_multiplier = max(min_tp_multiplier, 2)  # Al menos 2.0 o el doble del SL
            self.config.setdefault('tp_atr_multiplier', default_tp_multiplier)
            
            self.config.setdefault('timeframe', '1m')  # Añadir timeframe por defecto
            
            # Verificar y ajustar la relación riesgo/recompensa si es necesario
            if self.config['tp_atr_multiplier'] < self.config['atr_multiplier'] * 2:
                logger.warning(f"La relación riesgo/recompensa configurada ({self.config['tp_atr_multiplier'] / self.config['atr_multiplier']:.2f}) es menor que 2.0")
                logger.warning("Ajustando tp_atr_multiplier para mantener una relación mínima de 1:2")
                self.config['tp_atr_multiplier'] = self.config['atr_multiplier'] * 2
            
            logger.info("Configuration loaded successfully")
            logger.info(f"Relación riesgo/recompensa: 1:{self.config['tp_atr_multiplier'] / self.config['atr_multiplier']:.2f}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Create a default config
            self.config = {
                'api_key': 'tu_api_key',
                'api_secret': 'tu_api_secret',
                'symbol': 'STXUSDT',
                'leverage': 10,
                'stop_loss_percent': 1,
                'take_profit_min': 2,
                'take_profit_max': 3,
                'position_size_percent': 100,
                'sma_period': 200,
                'use_heikin_ashi': True,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'use_volatility_filter': True,
                'atr_period': 14,
                'atr_multiplier': 1.5,
                'tp_atr_multiplier': 2.0,  # Asegurar relación 1:2
                'timeframe': '1m'  # Añadir timeframe por defecto
            }
            # Save the default config
            with open(config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
            logger.info(f"Created default configuration at {config_path}")
    
    def save_config(self, config_path='config.json'):
        """Save the current configuration to a file"""
        try:
            with open(config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def set_leverage(self):
        """Set leverage for the trading pair"""
        try:
            self.client.futures_change_leverage(
                symbol=self.config['symbol'], 
                leverage=self.config['leverage']
            )
            logger.info(f"Leverage set to {self.config['leverage']}x for {self.config['symbol']}")
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage: {e}")
    
    def get_historical_data(self, interval='1m', limit=500):
        """
        Fetch historical klines/candlestick data
        
        Args:
            interval (str): Candle interval (1m, 5m, 15m, etc.)
            limit (int): Number of candles to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            klines = self.client.futures_klines(
                symbol=self.config['symbol'],
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert values to numeric
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            df.set_index('timestamp', inplace=True)
            
            return df
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for the strategy
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        # Prepare DataFrame for indicators
        analysis_df = df.copy()
        
        # Convert to Heikin Ashi if configured
        if self.config['use_heikin_ashi']:
            ha_df = TechnicalIndicators.heikin_ashi(analysis_df)
            for col in ha_df.columns:
                analysis_df[col] = ha_df[col]
            
            # For the strategy, we'll use Heikin Ashi values
            price_series = analysis_df['ha_close']
            is_bullish = analysis_df['ha_close'] > analysis_df['ha_open']
            is_bearish = analysis_df['ha_close'] < analysis_df['ha_open']
        else:
            # Use regular candles
            price_series = analysis_df['close']
            is_bullish = analysis_df['close'] > analysis_df['open']
            is_bearish = analysis_df['close'] < analysis_df['open']
        
        # Calculate SMA 200
        analysis_df['sma200'] = TechnicalIndicators.sma(price_series, self.config['sma_period'])
        
        # Calculate EMA 50 (for trend filter)
        analysis_df['ema50'] = TechnicalIndicators.ema(price_series, 50)
        
        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            price_series, 
            self.config['macd_fast'], 
            self.config['macd_slow'], 
            self.config['macd_signal']
        )
        analysis_df['macd'] = macd_line
        analysis_df['macd_signal'] = signal_line
        analysis_df['macd_hist'] = histogram
        
        # Calculate ATR
        analysis_df['atr'] = TechnicalIndicators.atr(df, self.config['atr_period'])
        
        # Calculate average ATR (for volatility filter)
        analysis_df['avg_atr'] = analysis_df['atr'].rolling(window=100).mean()
        
        # Trading signals
        analysis_df['is_bullish'] = is_bullish
        analysis_df['is_bearish'] = is_bearish
        
        # MACD crossover signals
        analysis_df['macd_bull_cross'] = (analysis_df['macd'] > analysis_df['macd_signal']) & (analysis_df['macd'].shift(1) <= analysis_df['macd_signal'].shift(1))
        analysis_df['macd_bear_cross'] = (analysis_df['macd'] < analysis_df['macd_signal']) & (analysis_df['macd'].shift(1) >= analysis_df['macd_signal'].shift(1))
        
        # Volatility filter
        if self.config['use_volatility_filter']:
            analysis_df['volatility_ok'] = analysis_df['atr'] <= analysis_df['avg_atr'] * self.config['atr_multiplier']
        else:
            analysis_df['volatility_ok'] = True
        
        # Final signals
        analysis_df['long_signal'] = (
            (price_series > analysis_df['sma200']) & 
            analysis_df['macd_bull_cross'] & 
            analysis_df['is_bullish'] & 
            analysis_df['volatility_ok']
        )
        
        analysis_df['short_signal'] = (
            (price_series < analysis_df['sma200']) & 
            analysis_df['macd_bear_cross'] & 
            analysis_df['is_bearish'] & 
            analysis_df['volatility_ok']
        )
        
        return analysis_df
    
    def get_account_balance(self):
        """Get futures account balance"""
        try:
            futures_account = self.client.futures_account()
            for asset in futures_account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            return 0
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return 0
    
    def get_position_info(self):
        """Get current position information for the trading pair"""
        try:
            positions = self.client.futures_position_information(symbol=self.config['symbol'])
            for position in positions:
                if position['symbol'] == self.config['symbol']:
                    position_size = float(position['positionAmt'])
                    entry_price = float(position['entryPrice'])
                    leverage = float(position.get('leverage', self.config['leverage']))
                    unrealized_pnl = float(position['unRealizedProfit'])
                    mark_price = float(position.get('markPrice', 0))
                    
                    self.positions = {
                        'symbol': position['symbol'],
                        'position_size': position_size,
                        'side': 'LONG' if position_size > 0 else 'SHORT' if position_size < 0 else 'NONE',
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'leverage': leverage,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percent': (unrealized_pnl / (abs(position_size) * entry_price / leverage)) * 100 if position_size != 0 and entry_price != 0 else 0
                    }
                    return self.positions
            
            # No position found, return default values
            self.positions = {
                'symbol': self.config['symbol'],
                'position_size': 0,
                'side': 'NONE',
                'entry_price': 0,
                'mark_price': 0,
                'leverage': self.config['leverage'],
                'unrealized_pnl': 0,
                'pnl_percent': 0
            }
            return self.positions
            
        except BinanceAPIException as e:
            logger.error(f"Error getting position info: {e}")
            # Return default values on error
            self.positions = {
                'symbol': self.config['symbol'],
                'position_size': 0,
                'side': 'NONE',
                'entry_price': 0,
                'mark_price': 0,
                'leverage': self.config['leverage'],
                'unrealized_pnl': 0,
                'pnl_percent': 0
            }
            return self.positions
    
    def calculate_position_size(self):
        """Calculate position size based on available balance and risk settings"""
        balance = self.get_account_balance()
        position_size_usdt = balance * (self.config['position_size_percent'] / 100)
        
        # Get current price
        ticker = self.client.futures_symbol_ticker(symbol=self.config['symbol'])
        current_price = float(ticker['price'])
        
        # Calculate quantity in base asset
        quantity = position_size_usdt / current_price
        
        # Get symbol info for precision
        exchange_info = self.client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == self.config['symbol']), None)
        
        if symbol_info:
            # Find the quantity precision
            quantity_precision = 0
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    quantity_precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
                    break
            
            # Round quantity to the correct precision
            quantity = round(quantity, quantity_precision)
        
        return quantity
    
    def place_order(self, side, quantity, order_type="MARKET"):
        """
        Place an order on Binance Futures
        
        Args:
            side (str): 'BUY' or 'SELL'
            quantity (float): Order quantity
            order_type (str): Order type (MARKET, LIMIT, etc.)
            
        Returns:
            dict: Order response
        """
        try:
            order = self.client.futures_create_order(
                symbol=self.config['symbol'],
                side=side,
                type=order_type,
                quantity=quantity
            )
            logger.info(f"Placed {side} order for {quantity} {self.config['symbol']} at market price")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_price_precision(self, symbol):
        exchange_info = self.client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        
        if symbol_info:
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
                    price_precision = len(str(tick_size).rstrip('0').split('.')[1]) if '.' in str(tick_size) else 0
                    return price_precision
        return 2  # Valor por defecto si no se encuentra información
    
    def place_take_profit_order(self, side, quantity, entry_price):
        """Place a take profit order based on ATR"""
        try:
            opposite_side = "SELL" if side == "BUY" else "BUY"
            
            # Ensure entry_price is valid
            if entry_price <= 0:
                logger.error("Entry price is invalid (0 or negative). Cannot place TP order.")
                return None
            
            # Calculate ATR for the current market
            atr_value = TechnicalIndicators.atr(self.get_historical_data(limit=100), self.config['atr_period']).iloc[-1]
            
            # Asegurar que la relación riesgo/recompensa sea al menos 1:2
            sl_multiplier = self.config['atr_multiplier']
            min_tp_multiplier = sl_multiplier * 2  # Relación mínima 1:2
            
            # Usar el multiplicador configurado o el mínimo calculado, el que sea mayor
            tp_multiplier = max(self.config.get('tp_atr_multiplier', sl_multiplier * 1.5), min_tp_multiplier)
            
            # Calculate the TP price
            tp_price = entry_price + (atr_value * tp_multiplier) if side == "BUY" else entry_price - (atr_value * tp_multiplier)
            
            # Get the price precision
            price_precision = self.get_price_precision(self.config['symbol'])
            
            # Round the TP price to the correct precision
            tp_price = round(tp_price, price_precision)
            
            # Ensure the tp_price is valid
            if tp_price <= 0:
                logger.error("Calculated TP price is invalid (0 or negative). Cannot place TP order.")
                return None
            
            # Calcular la relación riesgo/recompensa
            sl_price = entry_price - (atr_value * sl_multiplier) if side == "BUY" else entry_price + (atr_value * sl_multiplier)
            
            if side == "BUY":
                risk = entry_price - sl_price
                reward = tp_price - entry_price
            else:
                risk = sl_price - entry_price
                reward = entry_price - tp_price
                
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Verificar que la relación sea al menos 1:2
            if risk_reward_ratio < 2.0:
                logger.warning(f"La relación riesgo/recompensa calculada ({risk_reward_ratio:.2f}) es menor que 2.0")
                logger.warning("Ajustando el precio de TP para mantener una relación mínima de 1:2")
                
                # Recalcular el precio de TP para mantener la relación 1:2
                if side == "BUY":
                    tp_price = entry_price + (2.0 * (entry_price - sl_price))
                else:
                    tp_price = entry_price - (2.0 * (sl_price - entry_price))
                
                # Redondear nuevamente al precio correcto
                tp_price = round(tp_price, price_precision)
                
                # Recalcular la relación
                if side == "BUY":
                    reward = tp_price - entry_price
                else:
                    reward = entry_price - tp_price
                
                risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Log the details for debugging
            logger.info(f"Attempting to place TP order: Side={opposite_side}, Quantity={quantity}, StopPrice={tp_price}, Entry Price={entry_price}")
            logger.info(f"TP Distance: {abs(tp_price - entry_price):.4f} ({abs(tp_price/entry_price - 1) * 100:.2f}%)")
            logger.info(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}")
            
            # Create the TP order
            order = self.client.futures_create_order(
                symbol=self.config['symbol'],
                side=opposite_side,
                type="TAKE_PROFIT_MARKET",
                quantity=quantity,
                stopPrice=tp_price,
                workingType="MARK_PRICE",
                reduceOnly=True
            )
            
            logger.warning(f"Placed take profit order at {tp_price} (R/R: {risk_reward_ratio:.2f})")
            return order
        except Exception as e:
            logger.error(f"Error placing take profit order: {str(e)}")
            # Try with a different precision if the error is related to precision
            if "Precision is over the maximum" in str(e):
                try:
                    # Get exchange info to find the correct precision
                    exchange_info = self.client.futures_exchange_info()
                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == self.config['symbol']), None)
                    
                    if symbol_info:
                        # Find the price filter
                        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                        if price_filter:
                            tick_size = float(price_filter['tickSize'])
                            # Calculate the correct precision
                            correct_precision = len(str(tick_size).rstrip('0').split('.')[1]) if '.' in str(tick_size) else 0
                            
                            # Round the TP price to the correct precision
                            tp_price = round(tp_price, correct_precision)
                            
                            logger.info(f"Retrying TP order with corrected precision: {correct_precision}, Price: {tp_price}")
                            
                            # Create the TP order with corrected precision
                            order = self.client.futures_create_order(
                                symbol=self.config['symbol'],
                                side=opposite_side,
                                type="TAKE_PROFIT_MARKET",
                                quantity=quantity,
                                stopPrice=tp_price,
                                workingType="MARK_PRICE",
                                reduceOnly=True
                            )
                            
                            logger.warning(f"Placed take profit order at {tp_price} with corrected precision")
                            return order
                except Exception as retry_error:
                    logger.error(f"Error retrying take profit order with corrected precision: {str(retry_error)}")
            return None
    
    def place_stop_loss_order(self, side, quantity, entry_price):
        """Place a stop loss order based on ATR"""
        try:
            opposite_side = "SELL" if side == "BUY" else "BUY"
            
            # Asegúrate de que entry_price no sea 0
            if entry_price <= 0:
                logger.error("Entry price is invalid (0 or negative). Cannot place SL order.")
                return None
            
            # Calcular ATR para el mercado actual
            atr_value = TechnicalIndicators.atr(self.get_historical_data(limit=100), self.config['atr_period']).iloc[-1]
            
            # Calcular el precio de SL
            sl_price = entry_price - (atr_value * self.config['atr_multiplier']) if side == "BUY" else entry_price + (atr_value * self.config['atr_multiplier'])
            
            # Obtener la precisión del precio
            price_precision = self.get_price_precision(self.config['symbol'])
            
            # Redondear el precio de SL
            sl_price = round(sl_price, price_precision)
            
            # Asegúrate de que sl_price sea válido
            if sl_price <= 0:
                logger.error("Calculated SL price is invalid (0 or negative). Cannot place SL order.")
                return None
            
            # Log the details for debugging
            logger.info(f"Attempting to place SL order: Side={opposite_side}, Quantity={quantity}, StopPrice={sl_price}, Entry Price={entry_price}")
            
            # Crear la orden de SL
            order = self.client.futures_create_order(
                symbol=self.config['symbol'],
                side=opposite_side,
                type="STOP_MARKET",
                quantity=quantity,
                stopPrice=sl_price,
                workingType="MARK_PRICE",
                reduceOnly=True
            )
            
            logger.warning(f"Placed stop loss order at {sl_price}")
            return order
        except Exception as e:
            logger.error(f"Error placing stop loss order: {str(e)}")
            # Try with a different precision if the error is related to precision
            if "Precision is over the maximum" in str(e):
                try:
                    # Get exchange info to find the correct precision
                    exchange_info = self.client.futures_exchange_info()
                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == self.config['symbol']), None)
                    
                    if symbol_info:
                        # Find the price filter
                        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                        if price_filter:
                            tick_size = float(price_filter['tickSize'])
                            # Calculate the correct precision
                            correct_precision = len(str(tick_size).rstrip('0').split('.')[1]) if '.' in str(tick_size) else 0
                            
                            # Round the SL price to the correct precision
                            sl_price = round(sl_price, correct_precision)
                            
                            logger.info(f"Retrying SL order with corrected precision: {correct_precision}, Price: {sl_price}")
                            
                            # Create the SL order with corrected precision
                            order = self.client.futures_create_order(
                                symbol=self.config['symbol'],
                                side=opposite_side,
                                type="STOP_MARKET",
                                quantity=quantity,
                                stopPrice=sl_price,
                                workingType="MARK_PRICE",
                                reduceOnly=True
                            )
                            
                            logger.warning(f"Placed stop loss order at {sl_price} with corrected precision")
                            return order
                except Exception as retry_error:
                    logger.error(f"Error retrying stop loss order with corrected precision: {str(retry_error)}")
            return None
    
    def track_trade_history(self, entry_order, tp_order, sl_order, side, quantity, entry_price):
        """Registra información sobre la operación actual para seguimiento"""
        trade_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': self.config['symbol'],
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'tp_price': tp_order['stopPrice'] if tp_order else None,
            'sl_price': sl_order['stopPrice'] if sl_order else None,
            'leverage': self.config['leverage'],
            'status': 'OPEN',
            'exit_price': None,
            'pnl': None,
            'pnl_percent': None,
            'duration': None,
            'exit_reason': None
        }
        
        # Guardar en archivo JSON
        trade_history_file = 'trade_history.json'
        try:
            trade_history = []
            
            # Si el archivo existe y no está vacío, intentar cargarlo
            if os.path.exists(trade_history_file) and os.path.getsize(trade_history_file) > 0:
                try:
                    with open(trade_history_file, 'r') as f:
                        trade_history = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("El archivo trade_history.json está vacío o corrupto. Creando uno nuevo.")
                    trade_history = []
            
            trade_history.append(trade_info)
            
            with open(trade_history_file, 'w') as f:
                json.dump(trade_history, f, indent=4)
                
            logger.info(f"Trade registrado: {side} {quantity} {self.config['symbol']} a {entry_price}")
        except Exception as e:
            logger.error(f"Error al registrar trade: {e}")
        
        return trade_info
    
    def check_closed_positions(self):
        """Verifica si alguna posición se ha cerrado y actualiza el historial de trades"""
        # Obtener historial de órdenes recientes
        try:
            # Verificar si tenemos una posición abierta
            position = self.get_position_info()
            
            # Cargar historial de trades
            trade_history_file = 'trade_history.json'
            trade_history = []
            
            # Si el archivo existe y no está vacío, intentar cargarlo
            if os.path.exists(trade_history_file) and os.path.getsize(trade_history_file) > 0:
                try:
                    with open(trade_history_file, 'r') as f:
                        trade_history = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("El archivo trade_history.json está vacío o corrupto. Creando uno nuevo.")
                    trade_history = []
            
            # Buscar trades abiertos
            for i, trade in enumerate(trade_history):
                if trade['status'] == 'OPEN':
                    # Si ya no tenemos posición pero el trade está marcado como abierto, significa que se cerró
                    if position['side'] == 'NONE':
                        # Obtener historial de órdenes recientes para encontrar la orden de cierre
                        orders = self.client.futures_get_all_orders(symbol=self.config['symbol'], limit=20)
                        
                        # Filtrar órdenes ejecutadas de TP o SL
                        closed_orders = [order for order in orders if 
                                        order['status'] == 'FILLED' and 
                                        (order['type'] == 'TAKE_PROFIT_MARKET' or order['type'] == 'STOP_MARKET')]
                        
                        # Ordenar por tiempo de ejecución (más reciente primero)
                        closed_orders.sort(key=lambda x: x['updateTime'], reverse=True)
                        
                        if closed_orders:
                            # Tomar la orden más reciente que coincida con la dirección del trade
                            for order in closed_orders:
                                # Verificar si esta orden cerró nuestra posición
                                if (trade['side'] == 'BUY' and order['side'] == 'SELL') or \
                                   (trade['side'] == 'SELL' and order['side'] == 'BUY'):
                                    
                                    # Obtener precio de salida
                                    if 'avgPrice' in order:
                                        exit_price = float(order['avgPrice'])
                                    else:
                                        # Si no hay avgPrice, intentar obtener de otras fuentes
                                        try:
                                            # Obtener trades recientes para encontrar el precio de salida
                                            account_trades = self.client.futures_account_trades(symbol=self.config['symbol'], limit=10)
                                            # Filtrar por orderId
                                            matching_trades = [t for t in account_trades if t['orderId'] == order['orderId']]
                                            if matching_trades:
                                                exit_price = float(matching_trades[0]['price'])
                                            else:
                                                # Si no se encuentra, usar el precio actual
                                                ticker = self.client.futures_symbol_ticker(symbol=self.config['symbol'])
                                                exit_price = float(ticker['price'])
                                                logger.warning(f"No se pudo obtener precio de salida exacto, usando precio actual: {exit_price}")
                                        except Exception as e:
                                            logger.error(f"Error obteniendo precio de salida: {e}")
                                            # Usar el precio de TP o SL como aproximación
                                            exit_price = float(trade['tp_price']) if order['type'] == 'TAKE_PROFIT_MARKET' else float(trade['sl_price'])
                                    
                                    entry_price = trade['entry_price']
                                    quantity = trade['quantity']
                                    
                                    # Calcular PnL
                                    if trade['side'] == 'BUY':
                                        pnl = (exit_price - entry_price) * quantity
                                        pnl_percent = ((exit_price / entry_price) - 1) * 100 * self.config['leverage']
                                    else:
                                        pnl = (entry_price - exit_price) * quantity
                                        pnl_percent = ((entry_price / exit_price) - 1) * 100 * self.config['leverage']
                                    
                                    # Actualizar trade
                                    trade_history[i]['status'] = 'CLOSED'
                                    trade_history[i]['exit_price'] = exit_price
                                    trade_history[i]['pnl'] = pnl
                                    trade_history[i]['pnl_percent'] = pnl_percent
                                    trade_history[i]['exit_reason'] = 'TP' if order['type'] == 'TAKE_PROFIT_MARKET' else 'SL'
                                    
                                    # Calcular duración
                                    start_time = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
                                    end_time = datetime.now()
                                    duration = (end_time - start_time).total_seconds() / 60  # en minutos
                                    trade_history[i]['duration'] = duration
                                    
                                    # Guardar actualización
                                    with open(trade_history_file, 'w') as f:
                                        json.dump(trade_history, f, indent=4)
                                    
                                    # Mostrar resumen de la operación
                                    result = "GANANCIA" if pnl > 0 else "PÉRDIDA"
                                    logger.warning(f"\n===== OPERACIÓN CERRADA =====")
                                    logger.warning(f"Símbolo: {trade['symbol']}")
                                    logger.warning(f"Dirección: {trade['side']}")
                                    logger.warning(f"Entrada: {entry_price}")
                                    logger.warning(f"Salida: {exit_price}")
                                    logger.warning(f"Cantidad: {quantity}")
                                    logger.warning(f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
                                    logger.warning(f"Razón de salida: {trade_history[i]['exit_reason']}")
                                    logger.warning(f"Duración: {duration:.1f} minutos")
                                    logger.warning(f"Resultado: {result}")
                                    logger.warning(f"=============================\n")
                                    
                                    # Notificar al usuario
                                    print(f"\n===== OPERACIÓN CERRADA: {result} =====")
                                    print(f"Símbolo: {trade['symbol']} | Dirección: {trade['side']}")
                                    print(f"Entrada: {entry_price} | Salida: {exit_price}")
                                    print(f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
                                    print(f"Razón: {trade_history[i]['exit_reason']} | Duración: {duration:.1f} min")
                                    print("=======================================\n")
                                    
                                    break
                        else:
                            # Si no se encuentran órdenes de cierre, verificar si la posición se cerró manualmente
                            logger.warning("No se encontraron órdenes de cierre, verificando si la posición se cerró manualmente")
                            
                            try:
                                # Obtener trades recientes
                                account_trades = self.client.futures_account_trades(symbol=self.config['symbol'], limit=10)
                                # Ordenar por tiempo (más reciente primero)
                                account_trades.sort(key=lambda x: x['time'], reverse=True)
                                
                                # Buscar trades que coincidan con la dirección opuesta a nuestra posición
                                matching_trades = [t for t in account_trades if 
                                                  (trade['side'] == 'BUY' and t['side'] == 'SELL') or 
                                                  (trade['side'] == 'SELL' and t['side'] == 'BUY')]
                                
                                if matching_trades:
                                    # Usar el precio del trade más reciente
                                    exit_price = float(matching_trades[0]['price'])
                                    entry_price = trade['entry_price']
                                    quantity = trade['quantity']
                                    
                                    # Calcular PnL
                                    if trade['side'] == 'BUY':
                                        pnl = (exit_price - entry_price) * quantity
                                        pnl_percent = ((exit_price / entry_price) - 1) * 100 * self.config['leverage']
                                    else:
                                        pnl = (entry_price - exit_price) * quantity
                                        pnl_percent = ((entry_price / exit_price) - 1) * 100 * self.config['leverage']
                                    
                                    # Actualizar trade
                                    trade_history[i]['status'] = 'CLOSED'
                                    trade_history[i]['exit_price'] = exit_price
                                    trade_history[i]['pnl'] = pnl
                                    trade_history[i]['pnl_percent'] = pnl_percent
                                    trade_history[i]['exit_reason'] = 'MANUAL'
                                    
                                    # Calcular duración
                                    start_time = datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S')
                                    end_time = datetime.now()
                                    duration = (end_time - start_time).total_seconds() / 60  # en minutos
                                    trade_history[i]['duration'] = duration
                                    
                                    # Guardar actualización
                                    with open(trade_history_file, 'w') as f:
                                        json.dump(trade_history, f, indent=4)
                                    
                                    # Mostrar resumen de la operación
                                    result = "GANANCIA" if pnl > 0 else "PÉRDIDA"
                                    logger.warning(f"\n===== OPERACIÓN CERRADA MANUALMENTE =====")
                                    logger.warning(f"Símbolo: {trade['symbol']}")
                                    logger.warning(f"Dirección: {trade['side']}")
                                    logger.warning(f"Entrada: {entry_price}")
                                    logger.warning(f"Salida: {exit_price}")
                                    logger.warning(f"Cantidad: {quantity}")
                                    logger.warning(f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
                                    logger.warning(f"Duración: {duration:.1f} minutos")
                                    logger.warning(f"Resultado: {result}")
                                    logger.warning(f"=============================\n")
                                    
                                    # Notificar al usuario
                                    print(f"\n===== OPERACIÓN CERRADA MANUALMENTE: {result} =====")
                                    print(f"Símbolo: {trade['symbol']} | Dirección: {trade['side']}")
                                    print(f"Entrada: {entry_price} | Salida: {exit_price}")
                                    print(f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
                                    print(f"Duración: {duration:.1f} min")
                                    print("=======================================\n")
                            except Exception as e:
                                logger.error(f"Error verificando cierre manual: {e}")
                                
        except Exception as e:
            logger.error(f"Error al verificar posiciones cerradas: {e}")
    
    def execute_strategy(self):
        """Execute the trading strategy"""
        # Verificar si alguna posición se ha cerrado
        self.check_closed_positions()
        
        # Get current market data using the configured timeframe
        timeframe = self.config.get('timeframe', '1m')  # Use configured timeframe
        df = self.get_historical_data(interval=timeframe, limit=300)
        if df.empty:
            logger.error("No data received from Binance API")
            return
        
        # Calculate indicators
        analysis_df = self.calculate_indicators(df)
        
        # Get the latest row of data
        latest = analysis_df.iloc[-1]
        
        # Get current position
        position = self.get_position_info()
        position_side = position['side']
        
        # Check for trading signals
        long_signal = latest['long_signal']
        short_signal = latest['short_signal']
        
        # Debug information
        logger.info(f"Symbol: {self.config['symbol']} | Price: {latest['close']:.2f} | Position: {position_side} | Timeframe: {timeframe}")
        logger.info(f"Long Signal: {long_signal} | Short Signal: {short_signal}")
        logger.info(f"SMA200: {latest['sma200']:.4f} | MACD: {latest['macd']:.6f} | Signal: {latest['macd_signal']:.6f}")
        logger.info(f"ATR: {latest['atr']:.6f} | Volatility OK: {latest['volatility_ok']}")
        
        # Execute trades based on signals
        if position_side == 'NONE':
            # No current position, look for entry
            if long_signal:
                logger.info("🟢 LONG Entry Signal detected")
                quantity = self.calculate_position_size()
                
                # Open long position
                order = self.place_order("BUY", quantity)
                if order:
                    logger.info(f"Order response: {order}")  # Log the entire order response for debugging
                    
                    # Try to get the entry price from different possible fields in the order response
                    entry_price = None
                    if 'avgPrice' in order:
                        entry_price = float(order['avgPrice'])
                    elif 'price' in order and float(order['price']) > 0:
                        entry_price = float(order['price'])
                    elif 'fills' in order and len(order['fills']) > 0:
                        # Calculate average price from fills
                        total_qty = 0
                        total_price = 0
                        for fill in order['fills']:
                            fill_qty = float(fill['qty'])
                            fill_price = float(fill['price'])
                            total_qty += fill_qty
                            total_price += fill_qty * fill_price
                        if total_qty > 0:
                            entry_price = total_price / total_qty
                    
                    # If we still don't have a valid entry price, use the current market price
                    if not entry_price or entry_price <= 0:
                        ticker = self.client.futures_symbol_ticker(symbol=self.config['symbol'])
                        entry_price = float(ticker['price'])
                        logger.warning(f"Could not get entry price from order response. Using current market price: {entry_price}")
                    
                    logger.warning(f"Placing TP and SL orders for LONG position at entry price: {entry_price}")
                    
                    # Verify that the entry price is valid
                    if entry_price > 0:
                        # Place TP order with retry logic
                        tp_order = None
                        tp_attempts = 0
                        while not tp_order and tp_attempts < 3:
                            tp_order = self.place_take_profit_order("BUY", quantity, entry_price)
                            if not tp_order:
                                logger.error(f"Failed to place TP order for LONG position (attempt {tp_attempts + 1}/3).")
                                tp_attempts += 1
                                time.sleep(1)  # Wait a bit before retrying
                        
                        # Place SL order with retry logic
                        sl_order = None
                        sl_attempts = 0
                        while not sl_order and sl_attempts < 3:
                            sl_order = self.place_stop_loss_order("BUY", quantity, entry_price)
                            if not sl_order:
                                logger.error(f"Failed to place SL order for LONG position (attempt {sl_attempts + 1}/3).")
                                sl_attempts += 1
                                time.sleep(1)  # Wait a bit before retrying
                        
                        # Display prices in console
                        logger.info(f"Entry Price: {entry_price}, TP Price: {tp_order['stopPrice'] if tp_order else 'N/A'}, SL Price: {sl_order['stopPrice'] if sl_order else 'N/A'}, Current Price: {latest['close']:.2f}")
                        
                        # Registrar la operación
                        self.track_trade_history(order, tp_order, sl_order, "BUY", quantity, entry_price)
                        
                        # Log success or failure
                        if tp_order and sl_order:
                            logger.warning("Successfully placed both TP and SL orders for LONG position")
                        elif tp_order:
                            logger.warning("Successfully placed TP order but failed to place SL order for LONG position")
                        elif sl_order:
                            logger.warning("Successfully placed SL order but failed to place TP order for LONG position")
                        else:
                            logger.error("Failed to place both TP and SL orders for LONG position")
                    else:
                        logger.error("Entry price is invalid (0 or negative). Cannot place TP or SL orders.")
                else:
                    logger.error("Failed to place LONG order.")
            
            elif short_signal:
                logger.info("🟡 SHORT Entry Signal detected")
                quantity = self.calculate_position_size()
                
                # Open short position
                order = self.place_order("SELL", quantity)
                if order:
                    logger.info(f"Order response: {order}")  # Log the entire order response for debugging
                    
                    # Try to get the entry price from different possible fields in the order response
                    entry_price = None
                    if 'avgPrice' in order:
                        entry_price = float(order['avgPrice'])
                    elif 'price' in order and float(order['price']) > 0:
                        entry_price = float(order['price'])
                    elif 'fills' in order and len(order['fills']) > 0:
                        # Calculate average price from fills
                        total_qty = 0
                        total_price = 0
                        for fill in order['fills']:
                            fill_qty = float(fill['qty'])
                            fill_price = float(fill['price'])
                            total_qty += fill_qty
                            total_price += fill_qty * fill_price
                        if total_qty > 0:
                            entry_price = total_price / total_qty
                    
                    # If we still don't have a valid entry price, use the current market price
                    if not entry_price or entry_price <= 0:
                        ticker = self.client.futures_symbol_ticker(symbol=self.config['symbol'])
                        entry_price = float(ticker['price'])
                        logger.warning(f"Could not get entry price from order response. Using current market price: {entry_price}")
                    
                    logger.warning(f"Placing TP and SL orders for SHORT position at entry price: {entry_price}")
                    
                    # Verify that the entry price is valid
                    if entry_price > 0:
                        # Place TP order with retry logic
                        tp_order = None
                        tp_attempts = 0
                        while not tp_order and tp_attempts < 3:
                            tp_order = self.place_take_profit_order("SELL", quantity, entry_price)
                            if not tp_order:
                                logger.error(f"Failed to place TP order for SHORT position (attempt {tp_attempts + 1}/3).")
                                tp_attempts += 1
                                time.sleep(1)  # Wait a bit before retrying
                        
                        # Place SL order with retry logic
                        sl_order = None
                        sl_attempts = 0
                        while not sl_order and sl_attempts < 3:
                            sl_order = self.place_stop_loss_order("SELL", quantity, entry_price)
                            if not sl_order:
                                logger.error(f"Failed to place SL order for SHORT position (attempt {sl_attempts + 1}/3).")
                                sl_attempts += 1
                                time.sleep(1)  # Wait a bit before retrying
                        
                        # Display prices in console
                        logger.info(f"Entry Price: {entry_price}, TP Price: {tp_order['stopPrice'] if tp_order else 'N/A'}, SL Price: {sl_order['stopPrice'] if sl_order else 'N/A'}, Current Price: {latest['close']:.2f}")
                        
                        # Registrar la operación
                        self.track_trade_history(order, tp_order, sl_order, "SELL", quantity, entry_price)
                        
                        # Log success or failure
                        if tp_order and sl_order:
                            logger.warning("Successfully placed both TP and SL orders for SHORT position")
                        elif tp_order:
                            logger.warning("Successfully placed TP order but failed to place SL order for SHORT position")
                        elif sl_order:
                            logger.warning("Successfully placed SL order but failed to place TP order for SHORT position")
                        else:
                            logger.error("Failed to place both TP and SL orders for SHORT position")
                    else:
                        logger.error("Entry price is invalid (0 or negative). Cannot place TP or SL orders.")
                else:
                    logger.error("Failed to place SHORT order.")
        
        # Display position information if in a position
        if position_side != 'NONE':
            logger.info("\n=== CURRENT POSITION ===")
            logger.info(f"Symbol: {position['symbol']}")
            logger.info(f"Side: {position['side']}")
            logger.info(f"Size: {position['position_size']} (${abs(position['position_size'] * position['entry_price']):.2f})")
            logger.info(f"Entry Price: {position['entry_price']}")
            logger.info(f"Current Price: {position['mark_price']}")
            logger.info(f"PnL: ${position['unrealized_pnl']:.2f} ({position['pnl_percent']:.2f}%)")
            logger.info(f"Leverage: {position['leverage']}x")
            
            # Calcular y mostrar precios de TP y SL estimados
            atr_value = latest['atr']
            if position['side'] == 'LONG':
                est_tp_price = position['entry_price'] + (atr_value * self.config['atr_multiplier'])
                est_sl_price = position['entry_price'] - (atr_value * self.config['atr_multiplier'])
            else:  # SHORT
                est_tp_price = position['entry_price'] - (atr_value * self.config['atr_multiplier'])
                est_sl_price = position['entry_price'] + (atr_value * self.config['atr_multiplier'])
            
            logger.info(f"Est. TP Price: {est_tp_price:.4f} (${(est_tp_price - position['entry_price']) * position['position_size']:.2f})")
            logger.info(f"Est. SL Price: {est_sl_price:.4f} (${(est_sl_price - position['entry_price']) * position['position_size']:.2f})")
            logger.info("========================\n")
    
    def analyze_risk_reward(self):
        """Analiza la relación riesgo/recompensa de las operaciones históricas"""
        trade_history_file = 'trade_history.json'
        if os.path.exists(trade_history_file):
            try:
                with open(trade_history_file, 'r') as f:
                    trade_history = json.load(f)
                
                # Filtrar operaciones cerradas
                closed_trades = [t for t in trade_history if t['status'] == 'CLOSED']
                
                if not closed_trades:
                    logger.info("No hay operaciones cerradas para analizar")
                    return
                
                # Calcular estadísticas
                total_trades = len(closed_trades)
                winning_trades = [t for t in closed_trades if t['pnl'] > 0]
                losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
                
                win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
                
                # Calcular relación riesgo/recompensa promedio
                avg_win = sum([t['pnl'] for t in winning_trades]) / len(winning_trades) if winning_trades else 0
                avg_loss = sum([t['pnl'] for t in losing_trades]) / len(losing_trades) if losing_trades else 0
                
                avg_risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                # Calcular expectativa matemática
                expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
                
                # Mostrar resultados
                logger.info("\n===== ANÁLISIS DE RIESGO/RECOMPENSA =====")
                logger.info(f"Total de operaciones: {total_trades}")
                logger.info(f"Operaciones ganadoras: {len(winning_trades)} ({win_rate:.1f}%)")
                logger.info(f"Operaciones perdedoras: {len(losing_trades)} ({100-win_rate:.1f}%)")
                logger.info(f"Ganancia promedio: ${avg_win:.2f}")
                logger.info(f"Pérdida promedio: ${avg_loss:.2f}")
                logger.info(f"Relación riesgo/recompensa promedio: {avg_risk_reward:.2f}")
                logger.info(f"Expectativa matemática: ${expectancy:.2f}")
                logger.info("=========================================\n")
                
                # Mostrar en consola
                print("\n===== ANÁLISIS DE RIESGO/RECOMPENSA =====")
                print(f"Total de operaciones: {total_trades}")
                print(f"Operaciones ganadoras: {len(winning_trades)} ({win_rate:.1f}%)")
                print(f"Operaciones perdedoras: {len(losing_trades)} ({100-win_rate:.1f}%)")
                print(f"Ganancia promedio: ${avg_win:.2f}")
                print(f"Pérdida promedio: ${avg_loss:.2f}")
                print(f"Relación riesgo/recompensa promedio: {avg_risk_reward:.2f}")
                print(f"Expectativa matemática: ${expectancy:.2f}")
                print("=========================================\n")
                
            except Exception as e:
                logger.error(f"Error analizando riesgo/recompensa: {e}")
    
    def monitor_open_positions(self):
        """Monitorea las posiciones abiertas y muestra información detallada"""
        position = self.get_position_info()
        if position['side'] != 'NONE':
            # Obtener órdenes abiertas
            open_orders = self.client.futures_get_open_orders(symbol=self.config['symbol'])
            
            # Filtrar órdenes de TP y SL
            tp_orders = [o for o in open_orders if o['type'] == 'TAKE_PROFIT_MARKET']
            sl_orders = [o for o in open_orders if o['type'] == 'STOP_MARKET']
            
            # Mostrar información detallada
            print("\n===== POSICIÓN ABIERTA =====")
            print(f"Símbolo: {position['symbol']} | Dirección: {position['side']}")
            print(f"Tamaño: {position['position_size']} | Entrada: {position['entry_price']}")
            print(f"Precio actual: {position['mark_price']} | PnL: ${position['unrealized_pnl']:.2f} ({position['pnl_percent']:.2f}%)")
            
            if tp_orders:
                tp_price = float(tp_orders[0]['stopPrice'])
                tp_distance = abs(tp_price - position['mark_price'])
                tp_percent = (tp_distance / position['mark_price']) * 100
                print(f"TP: {tp_price} (Distancia: {tp_distance:.4f} / {tp_percent:.2f}%)")
            
            if sl_orders:
                sl_price = float(sl_orders[0]['stopPrice'])
                sl_distance = abs(sl_price - position['mark_price'])
                sl_percent = (sl_distance / position['mark_price']) * 100
                print(f"SL: {sl_price} (Distancia: {sl_distance:.4f} / {sl_percent:.2f}%)")
            
            # Calcular riesgo actual
            if tp_orders and sl_orders:
                tp_price = float(tp_orders[0]['stopPrice'])
                sl_price = float(sl_orders[0]['stopPrice'])
                
                if position['side'] == 'LONG':
                    potential_profit = (tp_price - position['mark_price']) * position['position_size']
                    potential_loss = (position['mark_price'] - sl_price) * position['position_size']
                else:
                    potential_profit = (position['mark_price'] - tp_price) * position['position_size']
                    potential_loss = (sl_price - position['mark_price']) * position['position_size']
                
                risk_reward_current = abs(potential_profit / potential_loss) if potential_loss > 0 else 0
                print(f"R/R actual: {risk_reward_current:.2f} | Ganancia potencial: ${potential_profit:.2f} | Pérdida potencial: ${potential_loss:.2f}")
            
            print("===========================\n")
    
    def verify_risk_reward_ratio(self):
        """Verifica y muestra la relación riesgo/recompensa de las órdenes abiertas"""
        position = self.get_position_info()
        if position['side'] != 'NONE':
            # Obtener órdenes abiertas
            open_orders = self.client.futures_get_open_orders(symbol=self.config['symbol'])
            
            # Filtrar órdenes de TP y SL
            tp_orders = [o for o in open_orders if o['type'] == 'TAKE_PROFIT_MARKET']
            sl_orders = [o for o in open_orders if o['type'] == 'STOP_MARKET']
            
            if tp_orders and sl_orders:
                tp_price = float(tp_orders[0]['stopPrice'])
                sl_price = float(sl_orders[0]['stopPrice'])
                entry_price = position['entry_price']
                
                if position['side'] == 'LONG':
                    risk = entry_price - sl_price
                    reward = tp_price - entry_price
                else:
                    risk = sl_price - entry_price
                    reward = entry_price - tp_price
                
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                logger.info(f"Relación riesgo/recompensa actual: 1:{risk_reward_ratio:.2f}")
                
                # Verificar si la relación es menor que 2.0
                if risk_reward_ratio < 2.0:
                    logger.warning(f"La relación riesgo/recompensa actual ({risk_reward_ratio:.2f}) es menor que 2.0")
                    logger.warning("Considera ajustar manualmente tus órdenes de TP y SL")
                    
                    # Calcular el precio de TP ideal para una relación 1:2
                    if position['side'] == 'LONG':
                        ideal_tp = entry_price + (2.0 * risk)
                    else:
                        ideal_tp = entry_price - (2.0 * risk)
                    
                    logger.info(f"Precio de TP ideal para relación 1:2: {ideal_tp:.6f}")
                
                return risk_reward_ratio
            
        return None
    
    def run(self):
        """Run the trading bot in a loop"""
        logger.info(f"Starting trading bot for {self.config['symbol']}")
        self.set_leverage()
        
        # Variables para estadísticas
        trades_total = 0
        trades_won = 0
        trades_lost = 0
        profit_total = 0
        last_analysis_time = time.time()
        last_rr_check_time = time.time()
        
        # Cargar historial de trades si existe
        trade_history_file = 'trade_history.json'
        if os.path.exists(trade_history_file):
            try:
                with open(trade_history_file, 'r') as f:
                    trade_history = json.load(f)
                
                # Calcular estadísticas
                for trade in trade_history:
                    if trade['status'] == 'CLOSED':
                        trades_total += 1
                        if trade['pnl'] > 0:
                            trades_won += 1
                            profit_total += trade['pnl']
                        else:
                            trades_lost += 1
                            profit_total += trade['pnl']
            except Exception as e:
                logger.error(f"Error cargando historial de trades: {e}")
        
        try:
            while True:
                # Clear terminal before each update
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Show header
                print(f"\n===== BINANCE FUTURES SCALPING BOT =====")
                print(f"Symbol: {self.config['symbol']} | Timeframe: {self.config.get('timeframe', '1m')}")
                print(f"Leverage: {self.config['leverage']}x | Position Size: {self.config['position_size_percent']}%")
                
                # Mostrar información sobre la relación riesgo/recompensa
                tp_multiplier = self.config.get('tp_atr_multiplier', self.config['atr_multiplier'] * 2)
                sl_multiplier = self.config['atr_multiplier']
                risk_reward_ratio = tp_multiplier / sl_multiplier
                
                print(f"Stop Loss: ATR x {sl_multiplier} | Take Profit: ATR x {tp_multiplier}")
                print(f"Relación Riesgo/Recompensa configurada: 1:{risk_reward_ratio:.1f}")
                print("=======================================")
                
                # Verificar relación riesgo/recompensa actual cada 5 minutos
                current_time = time.time()
                if current_time - last_rr_check_time > 300:  # 300 segundos = 5 minutos
                    rr_ratio = self.verify_risk_reward_ratio()
                    if rr_ratio:
                        print(f"Relación Riesgo/Recompensa actual: 1:{rr_ratio:.1f}")
                    last_rr_check_time = current_time
                
                # Mostrar estadísticas
                if trades_total > 0:
                    win_rate = (trades_won / trades_total) * 100
                    print(f"\n--- ESTADÍSTICAS DE TRADING ---")
                    print(f"Operaciones totales: {trades_total}")
                    print(f"Ganadas: {trades_won} | Perdidas: {trades_lost}")
                    print(f"Win Rate: {win_rate:.1f}%")
                    print(f"Beneficio total: ${profit_total:.2f}")
                    print("-------------------------------\n")
                
                # Execute strategy
                self.execute_strategy()
                
                # Realizar análisis de riesgo/recompensa cada hora
                if current_time - last_analysis_time > 3600:  # 3600 segundos = 1 hora
                    self.analyze_risk_reward()
                    last_analysis_time = current_time
                
                # Show last update time
                print(f"\nÚltima actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Esperando 30 segundos para la próxima actualización...\n")
                
                # Sleep for 30 seconds (avoids hitting rate limits, still checks frequently on 1m timeframe)
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise

if __name__ == "__main__":
    # Create default config file if it doesn't exist
    if not os.path.exists('config.json'):
        default_config = {
            'api_key': 'tu_api_key',
            'api_secret': 'tu_api_secret',
            'symbol': 'STXUSDT',
            'leverage': 10,
            'stop_loss_percent': 1,
            'take_profit_min': 2,
            'take_profit_max': 3,
            'position_size_percent': 100,
            'sma_period': 200,
            'use_heikin_ashi': True,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'use_volatility_filter': True,
            'atr_period': 14,
            'atr_multiplier': 1.0,
            'tp_atr_multiplier': 2.0,
            'timeframe': '1m'  # Añadir timeframe por defecto
        }
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
        print("Created default config.json file. Please edit with your API keys before running.")
        exit(0)
    
    # Check for API keys
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    if config['api_key'] == 'tu_api_key' and config['api_secret'] == 'tu_api_secret':
        print("Please edit config.json with your Binance API keys before running.")
        exit(0)
    
    # Start the bot
    bot = BinanceFuturesBot()
    bot.run()