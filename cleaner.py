"""
Data Cleaning Module for Institutional Quantitative Data Factory.

Handles data validation, anomaly detection, missing candle interpolation,
duplicate removal, and timezone correction.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import json

from config import (
    CLEANED_DATA_DIR, MAX_SPREAD_BP, MAX_GAP_PERCENT, OUTLIER_THRESHOLD_STD,
    MISSING_DATA_TOLERANCE, DUPLICATE_TOLERANCE_MS, OHLC_RULES, PRIMARY_TIMEFRAMES
)
from logger import pipeline_logger, performance_monitor, cleaner_logger
from utils import (
    safe_save_parquet, safe_load_parquet, ensure_directory,
    validate_ohlc_data, detect_outliers, detect_missing_candles
)

# ============================================================================
# DATA CLEANER
# ============================================================================

class DataCleaner:
    """Cleans and repairs raw OHLC data."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_report = {
            'duplicates_removed': 0,
            'outliers_removed': 0,
            'missing_candles_interpolated': 0,
            'nans_filled': 0,
            'invalid_ohlc_rows': 0,
            'high_spreads': 0,
            'large_gaps': 0,
        }
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        tolerance_ms: int = DUPLICATE_TOLERANCE_MS
    ) -> pd.DataFrame:
        """Remove duplicate candles within tolerance."""
        if df.empty or 'time' not in df.columns:
            return df
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # Find duplicates (same time within tolerance)
        duplicates_mask = np.zeros(len(df), dtype=bool)
        
        for i in range(len(df) - 1):
            current_time = df.iloc[i]['time']
            next_time = df.iloc[i + 1]['time']
            
            if isinstance(current_time, str):
                current_time = pd.Timestamp(current_time)
            if isinstance(next_time, str):
                next_time = pd.Timestamp(next_time)
            
            time_diff_ms = (next_time - current_time).total_seconds() * 1000
            
            if time_diff_ms <= tolerance_ms:
                duplicates_mask[i + 1] = True
        
        removed_count = duplicates_mask.sum()
        df = df[~duplicates_mask].reset_index(drop=True)
        
        self.cleaning_report['duplicates_removed'] += removed_count
        
        if removed_count > 0:
            cleaner_logger.info(f"Removed {removed_count} duplicate candles")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        threshold_std: float = OUTLIER_THRESHOLD_STD,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """Remove statistical outliers from price data."""
        if df.empty:
            return df
        
        if columns is None:
            columns = ['close', 'high', 'low']
        
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        for col in columns:
            if col in df.columns:
                outliers = detect_outliers(df[col], threshold_std)
                outlier_mask |= outliers
        
        removed_count = outlier_mask.sum()
        df = df[~outlier_mask].reset_index(drop=True)
        
        self.cleaning_report['outliers_removed'] += removed_count
        
        if removed_count > 0:
            cleaner_logger.info(f"Removed {removed_count} outlier rows")
        
        return df
    
    def fix_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """Fix missing values in data."""
        if df.empty:
            return df
        
        nan_count_before = df.isnull().sum().sum()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        nan_count_after = df.isnull().sum().sum()
        fixed_count = nan_count_before - nan_count_after
        
        self.cleaning_report['nans_filled'] += fixed_count
        
        if fixed_count > 0:
            cleaner_logger.info(f"Fixed {fixed_count} missing values")
        
        return df
    
    def validate_and_fix_ohlc(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate and fix OHLC relationships."""
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
        
        original_len = len(df)
        
        # Fix High >= Low
        invalid_mask = df['high'] < df['low']
        if invalid_mask.any():
            df.loc[invalid_mask, ['high', 'low']] = df.loc[invalid_mask, ['low', 'high']].values
        
        # Ensure High >= Open, Close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure Low <= Open, Close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Remove rows where high == low (invalid candles)
        invalid_flat = df['high'] == df['low']
        if invalid_flat.any():
            self.cleaning_report['invalid_ohlc_rows'] += invalid_flat.sum()
            df = df[~invalid_flat].reset_index(drop=True)
        
        if len(df) < original_len:
            cleaner_logger.info(f"Removed {original_len - len(df)} invalid OHLC rows")
        
        return df
    
    def check_spreads(
        self,
        df: pd.DataFrame,
        max_spread_bp: int = MAX_SPREAD_BP
    ) -> pd.DataFrame:
        """Flag and optionally remove candles with excessive spreads."""
        if df.empty or 'spread' not in df.columns:
            return df
        
        max_spread_price = max_spread_bp / 10000  # Convert bp to price
        
        high_spread_mask = df['spread'] > max_spread_price
        high_spread_count = high_spread_mask.sum()
        
        self.cleaning_report['high_spreads'] += high_spread_count
        
        if high_spread_count > 0:
            cleaner_logger.warning(
                f"Found {high_spread_count} candles with excessive spreads "
                f"(max: {max_spread_bp} bp)"
            )
        
        return df
    
    def check_gaps(
        self,
        df: pd.DataFrame,
        max_gap_percent: float = MAX_GAP_PERCENT
    ) -> pd.DataFrame:
        """Detect and flag large gaps in price."""
        if df.empty or 'close' not in df.columns or 'open' not in df.columns:
            return df
        
        # Calculate gap from previous close to current open
        df['prev_close'] = df['close'].shift(1)
        gap_percent = np.abs((df['open'] - df['prev_close']) / df['prev_close'] * 100)
        
        large_gaps_mask = gap_percent > max_gap_percent
        large_gaps_count = large_gaps_mask.sum()
        
        self.cleaning_report['large_gaps'] += large_gaps_count
        
        if large_gaps_count > 0:
            cleaner_logger.warning(
                f"Found {large_gaps_count} candles with large gaps "
                f"(max: {max_gap_percent}%)"
            )
        
        df = df.drop('prev_close', axis=1)
        return df
    
    def interpolate_missing_candles(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int
    ) -> pd.DataFrame:
        """Interpolate missing candles in time series."""
        if df.empty or 'time' not in df.columns:
            return df
        
        df = df.sort_values('time').reset_index(drop=True)
        
        # Create complete time range
        start_time = df['time'].iloc[0]
        end_time = df['time'].iloc[-1]
        
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)
        
        expected_times = pd.date_range(
            start_time,
            end_time,
            freq=f'{timeframe_minutes}min'
        )
        
        missing_count = len(expected_times) - len(df)
        
        if missing_count > 0:
            # Set time as index and reindex to fill gaps
            df = df.set_index('time')
            df = df.reindex(expected_times)
            df.index.name = 'time'
            df = df.reset_index()
            
            # Interpolate prices
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # Fill remaining NaNs (at edges)
            for col in numeric_cols:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            self.cleaning_report['missing_candles_interpolated'] += missing_count
            cleaner_logger.info(f"Interpolated {missing_count} missing candles")
        
        return df
    
    def clean_dataset(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int,
        remove_outliers: bool = True,
        interpolate_missing: bool = True
    ) -> pd.DataFrame:
        """Apply complete cleaning pipeline."""
        
        if df.empty:
            cleaner_logger.warning("Empty dataset provided for cleaning")
            return df
        
        cleaner_logger.info(f"Cleaning dataset with {len(df)} rows")
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Fix OHLC relationships
        df = self.validate_and_fix_ohlc(df)
        
        # Step 3: Fix missing values
        df = self.fix_missing_values(df, method='interpolate')
        
        # Step 4: Remove outliers (optional)
        if remove_outliers:
            df = self.remove_outliers(df)
        
        # Step 5: Check spreads and gaps
        df = self.check_spreads(df)
        df = self.check_gaps(df)
        
        # Step 6: Interpolate missing candles (optional)
        if interpolate_missing and 'time' in df.columns:
            df = self.interpolate_missing_candles(df, timeframe_minutes)
        
        cleaner_logger.info(f"Cleaning complete. Final dataset: {len(df)} rows")
        
        return df
    
    def get_report(self) -> Dict:
        """Get cleaning report."""
        return self.cleaning_report.copy()


# ============================================================================
# BATCH DATA CLEANER
# ============================================================================

class BatchDataCleaner:
    """Cleans multiple datasets with progress tracking."""
    
    def __init__(self, raw_dir: Path, output_dir: Path = CLEANED_DATA_DIR):
        """Initialize batch cleaner."""
        self.raw_dir = raw_dir
        self.output_dir = ensure_directory(output_dir)
        self.cleaner = DataCleaner()
        self.overall_stats = {}
    
    def get_timeframe_minutes(self, timeframe: str) -> int:
        """Get minutes for timeframe."""
        mapping = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN': 43200
        }
        return mapping.get(timeframe, 1)
    
    def clean_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        raw_file: Path
    ) -> bool:
        """Clean single symbol/timeframe file."""
        try:
            # Load raw data
            df = safe_load_parquet(raw_file)
            if df is None or df.empty:
                cleaner_logger.warning(f"Empty data: {symbol} {timeframe}")
                return False
            
            # Clean data
            timeframe_minutes = self.get_timeframe_minutes(timeframe)
            df_cleaned = self.cleaner.clean_dataset(
                df,
                timeframe_minutes,
                remove_outliers=True,
                interpolate_missing=True
            )
            
            # Save cleaned data
            output_file = self.output_dir / timeframe / f"{symbol}.parquet"
            success = safe_save_parquet(df_cleaned, output_file)
            
            if success:
                pipeline_logger.info(
                    f"Cleaned {symbol} {timeframe}: "
                    f"{len(df)} -> {len(df_cleaned)} rows"
                )
            
            return success
        
        except Exception as e:
            cleaner_logger.error(f"Error cleaning {symbol} {timeframe}: {e}")
            return False
    
    def clean_all_data(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ) -> Dict[str, int]:
        """Clean all raw data files."""
        
        if timeframes is None:
            timeframes = [tf.value for tf in PRIMARY_TIMEFRAMES]
        
        stats = {'processed': 0, 'successful': 0, 'failed': 0}
        
        for timeframe in timeframes:
            timeframe_dir = self.raw_dir / timeframe
            
            if not timeframe_dir.exists():
                continue
            
            # Get parquet files
            parquet_files = list(timeframe_dir.glob('*.parquet'))
            
            for file in parquet_files:
                symbol = file.stem
                
                # Skip if symbols specified and not in list
                if symbols and symbol not in symbols:
                    continue
                
                stats['processed'] += 1
                
                if self.clean_symbol_timeframe(symbol, timeframe, file):
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
        
        self.overall_stats = stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get cleaning statistics."""
        return {
            'overall': self.overall_stats,
            'detailed': self.cleaner.get_report()
        }


if __name__ == "__main__":
    # Test data cleaner
    np.random.seed(42)
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'tick_volume': np.random.randint(100, 1000, 100),
        'spread': np.random.uniform(0.0001, 0.0002, 100),
    })
    
    # Fix high/low
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    cleaner = DataCleaner()
    cleaned = cleaner.clean_dataset(test_data, timeframe_minutes=60)
    
    print(f"\nCleaning Report:")
    for key, value in cleaner.get_report().items():
        if value > 0:
            print(f"  {key}: {value}")
