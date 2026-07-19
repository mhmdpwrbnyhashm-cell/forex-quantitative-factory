"""
Utility Functions Module for Institutional Quantitative Data Factory.

Provides helper functions, data validation, memory management, and
common operations used across the pipeline.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime, timedelta
import pytz
from functools import wraps
import time

from config import (
    MAX_MEMORY_PERCENT, NUMPY_CHUNK_SIZE, PARQUET_COMPRESSION,
    RANDOM_SEED, DEFAULT_ENCODING
)
from logger import pipeline_logger, performance_monitor

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    memory = psutil.virtual_memory()
    return memory.available / (1024 ** 2)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def check_memory_available(required_mb: float) -> bool:
    """Check if required memory is available."""
    available = get_available_memory_mb()
    max_allowed = psutil.virtual_memory().total / (1024 ** 2) * (MAX_MEMORY_PERCENT / 100)
    return available > required_mb and get_memory_usage_mb() < max_allowed


def memory_efficient(func):
    """Decorator to monitor memory usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_memory = get_memory_usage_mb()
        result = func(*args, **kwargs)
        end_memory = get_memory_usage_mb()
        delta = end_memory - start_memory
        
        if delta > 10:  # Log if > 10MB increase
            pipeline_logger.info(
                f"{func.__name__} memory delta: {delta:.2f}MB (used: {end_memory:.2f}MB)"
            )
        
        return result
    return wrapper


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """Validate DataFrame structure and content."""
    if df is None or df.empty:
        return False, "DataFrame is None or empty"
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        return False, f"NaN values found in columns: {null_cols.to_dict()}"
    
    return True, "Valid"


def validate_ohlc_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate OHLC (Open, High, Low, Close) relationships."""
    issues = []
    
    if 'high' not in df.columns or 'low' not in df.columns:
        return False, ["Missing OHLC columns"]
    
    # High >= Low
    invalid = df[df['high'] < df['low']]
    if len(invalid) > 0:
        issues.append(f"{len(invalid)} rows where high < low")
    
    # High >= Open
    if 'open' in df.columns:
        invalid = df[df['high'] < df['open']]
        if len(invalid) > 0:
            issues.append(f"{len(invalid)} rows where high < open")
    
    # High >= Close
    if 'close' in df.columns:
        invalid = df[df['high'] < df['close']]
        if len(invalid) > 0:
            issues.append(f"{len(invalid)} rows where high < close")
    
    # Low <= Open
    if 'open' in df.columns:
        invalid = df[df['low'] > df['open']]
        if len(invalid) > 0:
            issues.append(f"{len(invalid)} rows where low > open")
    
    # Low <= Close
    if 'close' in df.columns:
        invalid = df[df['low'] > df['close']]
        if len(invalid) > 0:
            issues.append(f"{len(invalid)} rows where low > close")
    
    return len(issues) == 0, issues


def detect_outliers(data: pd.Series, threshold_std: float = 5.0) -> pd.Series:
    """Detect outliers using standard deviation method."""
    mean = data.mean()
    std = data.std()
    return np.abs((data - mean) / std) > threshold_std


def detect_missing_candles(
    timestamps: pd.Series,
    timeframe_minutes: int
) -> List[datetime]:
    """Detect missing candles in time series."""
    missing = []
    
    for i in range(len(timestamps) - 1):
        current = timestamps.iloc[i]
        next_ts = timestamps.iloc[i + 1]
        expected_diff = timedelta(minutes=timeframe_minutes)
        actual_diff = next_ts - current
        
        if actual_diff > expected_diff:
            # Calculate number of missing candles
            gap_candles = int(actual_diff / expected_diff) - 1
            for j in range(1, gap_candles + 1):
                missing.append(current + expected_diff * j)
    
    return missing


# ============================================================================
# DATA CONVERSION & TRANSFORMATION
# ============================================================================

def convert_timestamp_to_datetime(ts: Union[int, float], unit: str = 'ns') -> datetime:
    """Convert timestamp to datetime."""
    return pd.Timestamp(ts, unit=unit).to_pydatetime()


def convert_datetime_to_timestamp(dt: datetime, unit: str = 'ns') -> int:
    """Convert datetime to timestamp."""
    return int(pd.Timestamp(dt).value)


def normalize_price(price: float, decimal_places: int = 5) -> float:
    """Normalize price to decimal places."""
    return round(price, decimal_places)


def price_to_pips(price_change: float, pip_decimal_places: int = 4) -> float:
    """Convert price change to pips."""
    return price_change * (10 ** pip_decimal_places)


def pips_to_price(pips: float, pip_decimal_places: int = 4) -> float:
    """Convert pips to price change."""
    return pips / (10 ** pip_decimal_places)


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series."""
    return np.log(prices / prices.shift(1))


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from price series."""
    return prices.pct_change()


# ============================================================================
# STATISTICAL OPERATIONS
# ============================================================================

def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """Calculate rolling volatility from returns."""
    return returns.rolling(window).std()


def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 252
) -> pd.Series:
    """Calculate rolling correlation between two series."""
    return series1.rolling(window).corr(series2)


def calculate_zscore(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling z-score."""
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / std


def calculate_percentile_rank(
    data: pd.Series,
    window: int = 252
) -> pd.Series:
    """Calculate rolling percentile rank."""
    def percentile(x):
        return (x == np.max(x)).argmax() / len(x) * 100 if len(x) > 0 else 0
    
    return data.rolling(window).apply(percentile, raw=False)


def calculate_entropy(data: pd.Series, bins: int = 10) -> float:
    """Calculate Shannon entropy of data distribution."""
    counts, _ = np.histogram(data.dropna(), bins=bins)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_hurst_exponent(data: pd.Series, lags: range = range(10, 100)) -> float:
    """Calculate Hurst exponent."""
    tau = []
    for lag in lags:
        tau.append(
            np.sqrt(
                np.mean(
                    np.abs(
                        np.diff(data, lag)
                    ) ** 2
                )
            )
        )
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2


# ============================================================================
# CHUNKED PROCESSING
# ============================================================================

def process_in_chunks(
    data: pd.DataFrame,
    processor_func,
    chunk_size: int = NUMPY_CHUNK_SIZE,
    **kwargs
) -> pd.DataFrame:
    """Process large DataFrame in chunks to optimize memory."""
    chunks = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size].copy()
        processed_chunk = processor_func(chunk, **kwargs)
        chunks.append(processed_chunk)
        
        memory_mb = get_memory_usage_mb()
        if memory_mb > psutil.virtual_memory().total / (1024 ** 2) * (MAX_MEMORY_PERCENT / 100):
            pipeline_logger.warning(f"Memory usage high: {memory_mb:.2f}MB")
    
    return pd.concat(chunks, ignore_index=True)


def batch_load_parquet_files(
    directory: Path,
    pattern: str = "*.parquet",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load multiple parquet files from directory."""
    files = list(directory.glob(pattern))
    
    if not files:
        pipeline_logger.warning(f"No parquet files found in {directory}")
        return pd.DataFrame()
    
    dfs = []
    for file in sorted(files):
        try:
            df = pd.read_parquet(file, columns=columns)
            dfs.append(df)
        except Exception as e:
            pipeline_logger.error(f"Error loading {file}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 ** 2)


def safe_save_parquet(
    df: pd.DataFrame,
    file_path: Path,
    compression: str = PARQUET_COMPRESSION,
    index: bool = False
) -> bool:
    """Safely save DataFrame to parquet with error handling."""
    try:
        ensure_directory(file_path.parent)
        df.to_parquet(
            file_path,
            compression=compression,
            index=index,
            engine='pyarrow'
        )
        size_mb = get_file_size_mb(file_path)
        pipeline_logger.info(f"Saved {file_path.name} ({size_mb:.2f}MB)")
        return True
    except Exception as e:
        pipeline_logger.error(f"Failed to save {file_path}: {e}")
        return False


def safe_load_parquet(
    file_path: Path,
    columns: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Safely load parquet file with error handling."""
    try:
        df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
        pipeline_logger.info(f"Loaded {file_path.name} ({len(df)} rows)")
        return df
    except Exception as e:
        pipeline_logger.error(f"Failed to load {file_path}: {e}")
        return None


# ============================================================================
# TIME UTILITIES
# ============================================================================

def get_market_day_start(date: datetime, tz: pytz.timezone = pytz.UTC) -> datetime:
    """Get market day start (Sunday 21:00 UTC for Forex)."""
    # Forex week starts Sunday 21:00 UTC
    if date.weekday() == 6:  # Sunday
        return date.replace(hour=21, minute=0, second=0, microsecond=0, tzinfo=tz)
    else:
        # Go back to previous Sunday
        days_back = (date.weekday() + 1) % 7
        prev_sunday = date - timedelta(days=days_back)
        return prev_sunday.replace(hour=21, minute=0, second=0, microsecond=0, tzinfo=tz)


def get_market_day_end(date: datetime, tz: pytz.timezone = pytz.UTC) -> datetime:
    """Get market day end (Friday 21:00 UTC for Forex)."""
    # Forex week ends Friday 21:00 UTC
    if date.weekday() == 4:  # Friday
        return date.replace(hour=21, minute=0, second=0, microsecond=0, tzinfo=tz)
    else:
        # Go forward to next Friday
        days_forward = (4 - date.weekday()) % 7
        if days_forward == 0:
            days_forward = 7
        next_friday = date + timedelta(days=days_forward)
        return next_friday.replace(hour=21, minute=0, second=0, microsecond=0, tzinfo=tz)


def is_trading_session(dt: datetime, session: str) -> bool:
    """Check if datetime falls within trading session."""
    session_times = {
        'sydney': (21, 6),
        'tokyo': (23, 8),
        'london': (8, 17),
        'newyork': (13, 22),
    }
    
    if session.lower() not in session_times:
        return False
    
    start, end = session_times[session.lower()]
    hour = dt.hour
    
    if start < end:
        return start <= hour < end
    else:  # Wraps around midnight
        return hour >= start or hour < end


def get_trading_days(
    start_date: datetime,
    end_date: datetime,
    exclude_weekends: bool = True
) -> List[datetime]:
    """Get list of trading days between dates."""
    trading_days = []
    current = start_date
    
    while current <= end_date:
        if exclude_weekends:
            if current.weekday() < 5:  # Monday-Friday
                trading_days.append(current)
        else:
            trading_days.append(current)
        
        current += timedelta(days=1)
    
    return trading_days


# ============================================================================
# FORMATTING & DISPLAY
# ============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display."""
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage for display."""
    return f"{value * 100:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_dataframe_display(
    df: pd.DataFrame,
    max_rows: int = 10,
    max_columns: int = 10
) -> str:
    """Format DataFrame for nice display."""
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    return str(df)


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    pd.np.random.seed(seed)


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total_mb': psutil.virtual_memory().total / (1024 ** 2),
        'memory_available_mb': get_available_memory_mb(),
        'memory_used_mb': get_memory_usage_mb(),
        'disk_free_gb': psutil.disk_usage('/').free / (1024 ** 3),
    }


if __name__ == "__main__":
    print("✓ Utils module loaded")
    print(f"✓ Available memory: {get_available_memory_mb():.2f}MB")
    print(f"✓ Current memory usage: {get_memory_usage_mb():.2f}MB")
    info = get_system_info()
    print(f"✓ System info: {info}")
