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
            self.config.setdefault('leverage', 5)
            self.config.setdefault('stop_loss_percent', 2)
            self.config.setdefault('take_profit_min', 1)
            self.config.setdefault('take_profit_max', 2)
            self.config.setdefault('position_size_percent', 50)  # % of available balance
            self.config.setdefault('sma_period', 200)
            self.config.setdefault('use_heikin_ashi', True)
            self.config.setdefault('macd_fast', 12)
            self.config.setdefault('macd_slow', 26)
            self.config.setdefault('macd_signal', 9)
            self.config.setdefault('use_volatility_filter', True)
            self.config.setdefault('atr_period', 14)
            self.config.setdefault('atr_multiplier', 2.0)
            self.config.setdefault('timeframe', '1m')  # Añadir timeframe por defecto
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Create a default config
            self.config = {
                'api_key': 'tu_api_key',
                'api_secret': 'tu_api_secret',
                'symbol': 'STXUSDT',
                'leverage': 5,
                'stop_loss_percent': 2,
                'take_profit_min': 1,
                'take_profit_max': 2,
                'position_size_percent': 50,
                'sma_period': 200,
                'use_heikin_ashi': True,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'use_volatility_filter': True,
                'atr_period': 14,
                'atr_multiplier': 2.0,
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
            
            # Calculate the TP price
            tp_price = entry_price + (atr_value * self.config['atr_multiplier']) if side == "BUY" else entry_price - (atr_value * self.config['atr_multiplier'])
            
            # Get the price precision
            price_precision = self.get_price_precision(self.config['symbol'])
            
            # Round the TP price to the correct precision
            tp_price = round(tp_price, price_precision)
            
            # Ensure the tp_price is valid
            if tp_price <= 0:
                logger.error("Calculated TP price is invalid (0 or negative). Cannot place TP order.")
                return None
            
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
            
            logger.warning(f"Placed take profit order at {tp_price}")
            return order
        except Exception as e:
            logger.error(f"Error placing take profit order: {str(e)}")
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
            return None
    
    def execute_strategy(self):
        """Execute the trading strategy"""
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
        logger.info(f"SMA200: {latest['sma200']:.2f} | MACD: {latest['macd']:.6f} | Signal: {latest['macd_signal']:.6f}")
        
        # Execute trades based on signals
        if position_side == 'NONE':
            # No current position, look for entry
            if long_signal:
                logger.info("🟢 LONG Entry Signal detected")
                quantity = self.calculate_position_size()
                
                # Open long position
                order = self.place_order("BUY", quantity)
                if order:
                    # Captura el precio de entrada
                    entry_price = float(order['avgFillPrice']) if 'avgFillPrice' in order else float(latest['close'])
                    logger.warning(f"Placing TP and SL orders for LONG position at entry price: {entry_price}")
                    
                    # Verifica que el precio de entrada sea válido
                    if entry_price <= 0:
                        logger.error("Entry price is invalid (0 or negative). Cannot place TP or SL orders.")
                        return
                    
                    tp_order = self.place_take_profit_order("BUY", quantity, entry_price)
                    sl_order = self.place_stop_loss_order("BUY", quantity, entry_price)
                    
                    # Display prices in console
                    logger.info(f"Entry Price: {entry_price}, TP Price: {tp_order['stopPrice'] if tp_order else 'N/A'}, SL Price: {sl_order['stopPrice'] if sl_order else 'N/A'}, Current Price: {latest['close']:.2f}")
                    
                    # Verify that orders were placed correctly
                    if tp_order and sl_order:
                        logger.warning(f"Successfully placed TP and SL orders for LONG position")
                    else:
                        logger.error(f"Failed to place TP or SL orders for LONG position")
            
            elif short_signal:
                logger.info("🟡 SHORT Entry Signal detected")
                quantity = self.calculate_position_size()
                
                # Open short position
                order = self.place_order("SELL", quantity)
                if order:
                    # Place take profit and stop loss immediately after entry
                    entry_price = float(order['avgPrice']) if 'avgPrice' in order else float(latest['close'])
                    logger.warning(f"Placing TP and SL orders for SHORT position at entry price: {entry_price}")
                    tp_order = self.place_take_profit_order("SELL", quantity, entry_price)
                    sl_order = self.place_stop_loss_order("SELL", quantity, entry_price)
                    
                    # Display prices in console
                    logger.info(f"Entry Price: {entry_price}, TP Price: {tp_order['stopPrice'] if tp_order else 'N/A'}, SL Price: {sl_order['stopPrice'] if sl_order else 'N/A'}, Current Price: {latest['close']:.2f}")
                    
                    # Verify that orders were placed correctly
                    if tp_order and sl_order:
                        logger.warning(f"Successfully placed TP and SL orders for SHORT position")
                    else:
                        logger.error(f"Failed to place TP or SL orders for SHORT position")
        
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
            logger.info("========================\n")
    
    def run(self):
        """Run the trading bot in a loop"""
        logger.info(f"Starting trading bot for {self.config['symbol']}")
        self.set_leverage()
        
        try:
            while True:
                # Clear terminal before each update
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Show header
                print(f"\n===== BINANCE FUTURES SCALPING BOT =====")
                print(f"Symbol: {self.config['symbol']} | Timeframe: {self.config.get('timeframe', '1m')}")
                print(f"Leverage: {self.config['leverage']}x | Position Size: {self.config['position_size_percent']}%")
                print(f"Stop Loss: {self.config['stop_loss_percent']}% | Take Profit: {self.config['take_profit_min']}-{self.config['take_profit_max']}%")
                print("=======================================\n")
                
                # Execute strategy
                self.execute_strategy()
                
                # Show last update time
                print(f"\nLast update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Waiting 30 seconds for next update...\n")
                
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
            'leverage': 5,
            'stop_loss_percent': 2,
            'take_profit_min': 1,
            'take_profit_max': 2,
            'position_size_percent': 50,
            'sma_period': 200,
            'use_heikin_ashi': True,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'use_volatility_filter': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
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