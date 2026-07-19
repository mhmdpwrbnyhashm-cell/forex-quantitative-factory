"""
MetaTrader5 Data Collection Module for Institutional Quantitative Data Factory.

Handles automatic connection, symbol detection, historical data download,
and raw data storage with checkpoint and resume capabilities.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import json
from collections import defaultdict

from config import (
    START_DATE, END_DATE, BATCH_SIZE, MAX_DOWNLOAD_RETRIES,
    MT5_TIMEOUT, MT5_RECONNECT_ATTEMPTS, MT5_RECONNECT_DELAY,
    PRIMARY_TIMEFRAMES, RAW_DATA_DIR, TIMEFRAME_MAPPING, VERBOSE
)
from logger import pipeline_logger, performance_monitor, collector_logger
from utils import safe_save_parquet, ensure_directory, get_memory_usage_mb

# ============================================================================
# MT5 CONNECTION MANAGER
# ============================================================================

class MT5ConnectionManager:
    """Manages MetaTrader5 connection with auto-reconnect."""
    
    def __init__(self, timeout: int = MT5_TIMEOUT):
        """Initialize MT5 connection manager."""
        self.timeout = timeout
        self.is_connected = False
        self.connection_attempts = 0
        self.last_connection_time = None
    
    def connect(self) -> bool:
        """Establish MT5 connection with retries."""
        for attempt in range(MT5_RECONNECT_ATTEMPTS):
            try:
                if not mt5.initialize(timeout=self.timeout):
                    error = mt5.last_error()
                    collector_logger.warning(
                        f"MT5 init failed (attempt {attempt + 1}): {error}"
                    )
                    if attempt < MT5_RECONNECT_ATTEMPTS - 1:
                        time.sleep(MT5_RECONNECT_DELAY)
                    continue
                
                self.is_connected = True
                self.connection_attempts = attempt + 1
                self.last_connection_time = datetime.now()
                collector_logger.info(
                    f"MT5 connected successfully (attempt {attempt + 1})"
                )
                return True
            
            except Exception as e:
                collector_logger.error(f"MT5 connection error: {e}")
                if attempt < MT5_RECONNECT_ATTEMPTS - 1:
                    time.sleep(MT5_RECONNECT_DELAY)
        
        collector_logger.error("Failed to connect to MT5 after all attempts")
        return False
    
    def disconnect(self) -> bool:
        """Disconnect from MT5."""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                collector_logger.info("MT5 disconnected")
            return True
        except Exception as e:
            collector_logger.error(f"Error disconnecting MT5: {e}")
            return False
    
    def is_active(self) -> bool:
        """Check if connection is active."""
        if not self.is_connected:
            return False
        
        try:
            # Test connection by getting account info
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False
    
    def reconnect_if_needed(self) -> bool:
        """Reconnect if connection is lost."""
        if not self.is_active():
            collector_logger.warning("Connection lost, attempting reconnect...")
            self.disconnect()
            return self.connect()
        return True


# ============================================================================
# SYMBOL DETECTION & MANAGEMENT
# ============================================================================

class SymbolManager:
    """Manages Forex symbol detection and filtering."""
    
    def __init__(self, mt5_manager: MT5ConnectionManager):
        """Initialize symbol manager."""
        self.mt5_manager = mt5_manager
        self.symbols_cache = None
        self.tradeable_symbols = []
    
    def get_all_symbols(self, force_refresh: bool = False) -> List[str]:
        """Get all available symbols from broker."""
        if self.symbols_cache is not None and not force_refresh:
            return self.symbols_cache
        
        try:
            if not self.mt5_manager.is_active():
                collector_logger.error("MT5 not connected")
                return []
            
            symbols = mt5.symbol_total()
            if symbols is None or symbols <= 0:
                collector_logger.warning("No symbols available")
                return []
            
            all_symbols = []
            for i in range(symbols):
                symbol_info = mt5.symbol_info_by_index(i)
                if symbol_info is not None:
                    all_symbols.append(symbol_info.name)
            
            self.symbols_cache = all_symbols
            collector_logger.info(f"Found {len(all_symbols)} total symbols")
            return all_symbols
        
        except Exception as e:
            collector_logger.error(f"Error getting symbols: {e}")
            return []
    
    def filter_tradeable_symbols(
        self,
        all_symbols: List[str],
        keywords: List[str] = None
    ) -> List[str]:
        """Filter symbols that are tradeable."""
        if keywords is None:
            keywords = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD']
        
        tradeable = []
        
        for symbol in all_symbols:
            try:
                if not mt5.symbol_select(symbol, True):
                    continue
                
                # Check if symbol matches keywords
                if any(kw in symbol.upper() for kw in keywords):
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info is not None and symbol_info.visible:
                        tradeable.append(symbol)
            
            except Exception as e:
                collector_logger.debug(f"Error filtering symbol {symbol}: {e}")
        
        self.tradeable_symbols = tradeable
        collector_logger.info(f"Found {len(tradeable)} tradeable symbols")
        return tradeable
    
    def verify_symbol(self, symbol: str) -> bool:
        """Verify symbol is available and tradeable."""
        try:
            if not mt5.symbol_select(symbol, True):
                return False
            
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info is not None and symbol_info.visible
        
        except:
            return False


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    """Downloads historical OHLC data from MT5."""
    
    def __init__(self, mt5_manager: MT5ConnectionManager):
        """Initialize downloader."""
        self.mt5_manager = mt5_manager
        self.download_stats = defaultdict(int)
    
    def download_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = BATCH_SIZE
    ) -> Optional[pd.DataFrame]:
        """Download candles for symbol/timeframe."""
        
        if not self.mt5_manager.is_active():
            if not self.mt5_manager.reconnect_if_needed():
                return None
        
        try:
            mt5_timeframe = getattr(mt5, f'TIMEFRAME_{timeframe.replace("MN", "MN1")}')
        except:
            collector_logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        all_data = []
        current_date = start_date
        retry_count = 0
        
        while current_date < end_date:
            try:
                # Calculate batch end date
                batch_end = min(current_date + timedelta(days=365), end_date)
                
                # Download batch
                rates = mt5.copy_rates_range(
                    symbol,
                    mt5_timeframe,
                    current_date,
                    batch_end
                )
                
                if rates is None or len(rates) == 0:
                    collector_logger.warning(
                        f"No data: {symbol} {timeframe} {current_date}"
                    )
                    current_date = batch_end
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                all_data.append(df)
                retry_count = 0
                
                self.download_stats[f"{symbol}_{timeframe}"] += len(df)
                
                # Move to next batch
                current_date = batch_end
                
                if VERBOSE:
                    collector_logger.debug(
                        f"Downloaded {len(df)} candles: {symbol} {timeframe}"
                    )
            
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_DOWNLOAD_RETRIES:
                    collector_logger.error(
                        f"Max retries exceeded for {symbol} {timeframe}: {e}"
                    )
                    break
                
                collector_logger.warning(
                    f"Download error (retry {retry_count}): {e}"
                )
                time.sleep(2 ** retry_count)  # Exponential backoff
        
        if not all_data:
            return None
        
        result_df = pd.concat(all_data, ignore_index=True)
        result_df = result_df.drop_duplicates(subset=['time', 'symbol', 'timeframe'])
        result_df = result_df.sort_values('time').reset_index(drop=True)
        
        collector_logger.info(
            f"Downloaded {len(result_df)} candles: {symbol} {timeframe}"
        )
        
        return result_df
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        timeframes: List[str] = None,
        start_date: datetime = START_DATE,
        end_date: datetime = END_DATE
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Download data for multiple symbols and timeframes."""
        
        if timeframes is None:
            timeframes = [tf.value for tf in PRIMARY_TIMEFRAMES]
        
        all_data = {}
        total_symbols = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            all_data[symbol] = {}
            
            pipeline_logger.info(
                f"Processing symbol {idx + 1}/{total_symbols}: {symbol}"
            )
            
            for timeframe in timeframes:
                try:
                    df = self.download_candles(
                        symbol, timeframe, start_date, end_date
                    )
                    if df is not None and len(df) > 0:
                        all_data[symbol][timeframe] = df
                
                except Exception as e:
                    collector_logger.error(
                        f"Error downloading {symbol} {timeframe}: {e}"
                    )
        
        return all_data


# ============================================================================
# RAW DATA STORAGE
# ============================================================================

class RawDataStore:
    """Manages raw data storage and checkpoints."""
    
    def __init__(self, storage_dir: Path = RAW_DATA_DIR):
        """Initialize data store."""
        self.storage_dir = ensure_directory(storage_dir)
        self.checkpoint_file = self.storage_dir / "checkpoint.json"
    
    def save_raw_data(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame
    ) -> bool:
        """Save raw data to parquet."""
        try:
            timeframe_dir = ensure_directory(self.storage_dir / timeframe)
            file_path = timeframe_dir / f"{symbol}.parquet"
            
            return safe_save_parquet(df, file_path)
        
        except Exception as e:
            collector_logger.error(f"Error saving {symbol} {timeframe}: {e}")
            return False
    
    def load_raw_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load raw data from parquet."""
        try:
            file_path = self.storage_dir / timeframe / f"{symbol}.parquet"
            if not file_path.exists():
                return None
            
            return pd.read_parquet(file_path)
        
        except Exception as e:
            collector_logger.error(f"Error loading {symbol} {timeframe}: {e}")
            return None
    
    def get_checkpoint(self) -> Dict:
        """Load checkpoint data."""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            collector_logger.error(f"Error loading checkpoint: {e}")
        
        return {}
    
    def save_checkpoint(
        self,
        processed_symbols: List[str],
        current_symbol: str = None,
        timestamp: datetime = None
    ) -> bool:
        """Save checkpoint data."""
        try:
            checkpoint = {
                'timestamp': (timestamp or datetime.now()).isoformat(),
                'processed_symbols': processed_symbols,
                'current_symbol': current_symbol,
                'memory_usage_mb': get_memory_usage_mb()
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            return True
        
        except Exception as e:
            collector_logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def get_processed_symbols(self) -> List[str]:
        """Get list of already processed symbols."""
        checkpoint = self.get_checkpoint()
        return checkpoint.get('processed_symbols', [])
    
    def symbol_already_processed(self, symbol: str) -> bool:
        """Check if symbol was already processed."""
        timeframe_dir = self.storage_dir / "M1"  # Check one timeframe
        file_path = timeframe_dir / f"{symbol}.parquet"
        return file_path.exists()


# ============================================================================
# DATA COLLECTOR ORCHESTRATOR
# ============================================================================

class DataCollector:
    """Orchestrates the entire data collection process."""
    
    def __init__(self):
        """Initialize data collector."""
        self.mt5_manager = MT5ConnectionManager()
        self.symbol_manager = SymbolManager(self.mt5_manager)
        self.downloader = DataDownloader(self.mt5_manager)
        self.data_store = RawDataStore()
    
    def initialize(self) -> bool:
        """Initialize collector and connect to MT5."""
        pipeline_logger.log_stage_start("Data Collection Initialization")
        
        if not self.mt5_manager.connect():
            pipeline_logger.error("Failed to initialize MT5 connection")
            return False
        
        pipeline_logger.info("MT5 connection established")
        return True
    
    def finalize(self) -> None:
        """Clean up and disconnect."""
        self.mt5_manager.disconnect()
        pipeline_logger.info("Data collector finalized")
    
    def collect_data(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        skip_existing: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Collect data for symbols and timeframes."""
        
        performance_monitor.start_timer("data_collection")
        pipeline_logger.log_stage_start(
            "Data Collection",
            {'skip_existing': skip_existing, 'timeframes': timeframes}
        )
        
        # Get symbols
        if symbols is None:
            all_symbols = self.symbol_manager.get_all_symbols()
            symbols = self.symbol_manager.filter_tradeable_symbols(all_symbols)
        
        # Filter timeframes
        if timeframes is None:
            timeframes = [tf.value for tf in PRIMARY_TIMEFRAMES]
        
        # Download data
        processed_symbols = []
        if skip_existing:
            processed_symbols = self.data_store.get_processed_symbols()
        
        symbols_to_process = [s for s in symbols if s not in processed_symbols]
        
        pipeline_logger.info(
            f"Processing {len(symbols_to_process)} new symbols "
            f"(skipping {len(processed_symbols)} existing)"
        )
        
        all_data = self.downloader.download_multiple_symbols(
            symbols_to_process,
            timeframes,
            START_DATE,
            END_DATE
        )
        
        # Save data
        saved_count = 0
        for symbol, timeframe_data in all_data.items():
            for timeframe, df in timeframe_data.items():
                if self.data_store.save_raw_data(symbol, timeframe, df):
                    saved_count += 1
        
        # Update checkpoint
        all_processed = processed_symbols + list(all_data.keys())
        self.data_store.save_checkpoint(all_processed)
        
        # Log completion
        duration = performance_monitor.stop_timer("data_collection")
        pipeline_logger.log_stage_end(
            "Data Collection",
            duration,
            "SUCCESS",
            {
                'symbols_processed': len(all_data),
                'files_saved': saved_count,
                'total_processed': len(all_processed)
            }
        )
        
        return all_data


if __name__ == "__main__":
    collector = DataCollector()
    if collector.initialize():
        try:
            data = collector.collect_data(
                symbols=['EURUSD', 'GBPUSD'],
                skip_existing=True
            )
            print(f"\nCollected data for {len(data)} symbols")
        finally:
            collector.finalize()
