"""
Production Configuration Module for Institutional Quantitative Data Factory.

This module defines all configuration parameters, paths, constants, and
broker settings for the entire quantitative data pipeline.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Set, Optional
import pytz

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
VALIDATED_DATA_DIR = DATA_DIR / "validated"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
OUTPUT_DIR = DATA_DIR / "output"
METADATA_DIR = OUTPUT_DIR / "metadata"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories automatically
for directory in [
    DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, VALIDATED_DATA_DIR,
    FEATURES_DIR, LABELS_DIR, OUTPUT_DIR, METADATA_DIR, REPORTS_DIR,
    LOGS_DIR, CACHE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / f"factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LOG_SIZE_MB = 100
BACKUP_COUNT = 5

# ============================================================================
# METATRADER5 CONNECTION
# ============================================================================

MT5_TIMEOUT = 60000  # milliseconds
MT5_RECONNECT_ATTEMPTS = 3
MT5_RECONNECT_DELAY = 5  # seconds

# Common broker suffixes
MT5_BROKER_SUFFIXES = {
    "ICMarkets": ".i",
    "Pepperstone": ".p",
    "IG": ".ig",
    "RoboForex": ".r",
    "AMarkets": ".a",
    "Darwinex": ".d",
}

# ============================================================================
# FOREX UNIVERSE
# ============================================================================

# Major Forex pairs (non-exhaustive, auto-detection will find all symbols)
MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
]

# Exotic pairs
EXOTIC_PAIRS = [
    "USDSEK", "USDNOK", "USDDKK", "USDPLN", "USDCZK", "USDHUF",
    "SGDUSD", "HKDUSD", "ZARUSD", "ZARJPY",
]

# Indices and commodities
CROSS_SYMBOLS = [
    "XAUUSD",  # Gold
    "XAGUSD",  # Silver
    "USOIL",   # Crude Oil
    "BRENT",   # Brent Oil
    "NATGAS",  # Natural Gas
]

# Crypto (if available)
CRYPTO_SYMBOLS = [
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
]

# Equity indices (if available)
INDEX_SYMBOLS = [
    "SP500", "NASDAQ", "DAX", "FTSE", "HSI", "N225", "ASX200",
]

# Correlation matrix symbols (for feature generation)
CORRELATION_SYMBOLS = {
    "EURUSD": 1.0,
    "GBPUSD": 1.0,
    "USDJPY": 1.0,
    "USDCHF": 1.0,
    "AUDUSD": 1.0,
    "NZDUSD": 1.0,
    "USDCAD": 1.0,
    "XAUUSD": 1.0,
    "DXY": 1.0,
    "USOIL": 1.0,
    "SP500": 1.0,
    "NASDAQ": 1.0,
    "BTCUSD": 1.0,
    "ETHUSD": 1.0,
}

# ============================================================================
# TIMEFRAMES
# ============================================================================

class Timeframe(Enum):
    """Enumeration of all supported MetaTrader5 timeframes."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN = "MN"

# MetaTrader5 timeframe constants (mt5.TIMEFRAME_*)
TIMEFRAME_MAPPING = {
    "M1": 1,      # 1 minute
    "M5": 5,      # 5 minutes
    "M15": 15,    # 15 minutes
    "M30": 30,    # 30 minutes
    "H1": 60,     # 1 hour
    "H4": 240,    # 4 hours
    "D1": 1440,   # 1 day
    "W1": 10080,  # 1 week
    "MN": 43200,  # 1 month
}

# Candles per timeframe in one day (for alignment)
CANDLES_PER_DAY = {
    "M1": 1440,
    "M5": 288,
    "M15": 96,
    "M30": 48,
    "H1": 24,
    "H4": 6,
    "D1": 1,
    "W1": 0.2,  # ~5 candles per month
    "MN": 0.033,  # ~12 candles per year
}

# Time in minutes per timeframe
MINUTES_PER_TIMEFRAME = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN": 43200,
}

# Primary timeframes for features (most commonly used)
PRIMARY_TIMEFRAMES = [
    Timeframe.M1, Timeframe.M5, Timeframe.M15,
    Timeframe.M30, Timeframe.H1, Timeframe.H4,
    Timeframe.D1,
]

# Higher timeframes for multi-timeframe context
HIGHER_TIMEFRAMES = [Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1]

# ============================================================================
# DATA COLLECTION
# ============================================================================

# Historical data range
START_DATE = datetime(2015, 1, 1, tzinfo=pytz.UTC)  # 9 years of history
END_DATE = datetime.now(pytz.UTC)

# Chunk size for batch downloads (bars per request)
BATCH_SIZE = 10000

# Maximum retries for failed downloads
MAX_DOWNLOAD_RETRIES = 3

# Connection check interval (seconds)
CONNECTION_CHECK_INTERVAL = 300

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Technical indicator periods (standard institutional settings)
INDICATOR_PERIODS = {
    # Moving Averages
    "EMA_PERIODS": [5, 10, 20, 50, 100, 200],
    "SMA_PERIODS": [5, 10, 20, 50, 100, 200],
    "WMA_PERIODS": [5, 10, 20, 50, 100],
    "HULL_MA_PERIOD": 9,
    "KAMA_PERIOD": 10,
    "VWAP_RESET": "D1",  # Reset VWAP daily
    
    # Momentum
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "ROC_PERIOD": 12,
    "MOMENTUM_PERIOD": 10,
    "CCI_PERIOD": 20,
    "WILLIAMS_R_PERIOD": 14,
    "TSI_R": 25,
    "TSI_S": 13,
    "STOCHASTIC_K": 14,
    "STOCHASTIC_D": 3,
    "AWESOME_PERIOD_FAST": 5,
    "AWESOME_PERIOD_SLOW": 34,
    
    # Volatility
    "ATR_PERIOD": 14,
    "ROLLING_STD_PERIOD": 20,
    "HISTORICAL_VOL_PERIOD": 20,
    "VOLATILITY_PERCENTILE_PERIOD": 252,
    
    # Volume
    "OBV_PERIOD": 14,
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOLUME_ZSCORE_PERIOD": 20,
    
    # SuperTrend
    "SUPERTREND_PERIOD": 10,
    "SUPERTREND_MULTIPLIER": 3.0,
    
    # ADX/DMI
    "ADX_PERIOD": 14,
    
    # Ichimoku
    "ICHIMOKU_TENKAN": 9,
    "ICHIMOKU_KIJUN": 26,
    "ICHIMOKU_SENKOU_B": 52,
    "ICHIMOKU_CHIKOU": 26,
    
    # Parabolic SAR
    "SAR_STEP": 0.02,
    "SAR_MAX_STEP": 0.2,
    
    # Linear Regression
    "REGRESSION_PERIOD": 50,
    
    # Swing Detection
    "SWING_PERIOD": 5,  # Min 5 bars for swing high/low
    
    # Fair Value Gap
    "FVG_LOOKBACK": 3,
    
    # Rolling Correlations
    "CORRELATION_PERIOD": 252,
}

# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

# Maximum allowed spread in basis points (0.0001 = 1 pip)
MAX_SPREAD_BP = 500  # 50 pips for most pairs

# Maximum allowed gap (% change between close and next open)
MAX_GAP_PERCENT = 5.0

# Outlier detection (standard deviations)
OUTLIER_THRESHOLD_STD = 5.0

# Missing data tolerance (% of missing bars in a session allowed)
MISSING_DATA_TOLERANCE = 5.0

# Duplicate check tolerance (milliseconds)
DUPLICATE_TOLERANCE_MS = 100

# ============================================================================
# LABEL ENGINE CONFIGURATION
# ============================================================================

# Future returns lookback periods (in candles)
FUTURE_RETURN_PERIODS = [5, 10, 20, 50]

# Maximum Favorable Excursion (MFE) / Maximum Adverse Excursion (MAE)
MFE_MAE_LOOKBACK = 20  # Candles to look forward

# Triple Barrier Method
TRIPLE_BARRIER = {
    "UPSIDE_BARRIER_STD": 2.0,     # ATR multiples
    "DOWNSIDE_BARRIER_STD": 2.0,
    "TIMEFRAME_BARS": 20,
}

# Meta Labeling
META_LABEL_THRESHOLD = 0.55  # Confidence threshold

# Holding time max (candles)
MAX_HOLDING_TIME = 100

# ============================================================================
# QUALITY ASSURANCE
# ============================================================================

# Minimum data points required per symbol
MIN_CANDLES_REQUIRED = 10000

# Data quality score threshold (0-100)
MIN_QUALITY_SCORE = 80

# Leakage detection thresholds
LEAKAGE_CORRELATION_THRESHOLD = 0.95
LEAKAGE_LOOKAHEAD_BARS = 5

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# Parquet compression
PARQUET_COMPRESSION = "snappy"
PARQUET_ROW_GROUP_SIZE = 100000

# Chunk size for large exports (rows per file)
EXPORT_CHUNK_SIZE = 1000000

# Encoding
DEFAULT_ENCODING = "utf-8"

# ============================================================================
# PERFORMANCE & OPTIMIZATION
# ============================================================================

# NumPy/Pandas chunk processing
NUMPY_CHUNK_SIZE = 100000

# Multiprocessing
MAX_WORKERS = min(os.cpu_count() or 4, 8)

# Memory settings
MAX_MEMORY_PERCENT = 80  # Use max 80% of available RAM

# Cache settings
USE_CACHE = True
CACHE_EXPIRATION_HOURS = 24

# ============================================================================
# MARKET SESSIONS (UTC times)
# ============================================================================

MARKET_SESSIONS = {
    "Sydney": {"start": "21:00", "end": "06:00", "timezone": "Australia/Sydney"},
    "Tokyo": {"start": "23:00", "end": "08:00", "timezone": "Asia/Tokyo"},
    "London": {"start": "08:00", "end": "17:00", "timezone": "Europe/London"},
    "New York": {"start": "13:00", "end": "22:00", "timezone": "America/New_York"},
}

# Session overlaps in UTC
SESSION_OVERLAPS = {
    "Sydney-Tokyo": ("23:00", "06:00"),
    "Tokyo-London": ("08:00", "08:00"),
    "London-New York": ("13:00", "17:00"),
    "New York-Sydney": ("22:00", "21:00"),
}

# ============================================================================
# HOLIDAYS & SPECIAL DAYS
# ============================================================================

# Major holidays when Forex trades with reduced liquidity
FOREX_HOLIDAYS = {
    (1, 1): "New Year",
    (1, 2): "New Year Holiday",
    (12, 25): "Christmas",
    (12, 26): "Boxing Day",
}

# Early closes (half-day trading)
EARLY_CLOSE_DAYS = {
    (12, 24): "Christmas Eve",
    (12, 31): "New Year's Eve",
}

# ============================================================================
# FEATURE DICTIONARY
# ============================================================================

FEATURE_GROUPS = {
    "RAW": [
        "open", "high", "low", "close", "tick_volume", "real_volume", "spread"
    ],
    "PRICE": [
        "median_price", "typical_price", "weighted_price", "log_price",
        "return", "log_return", "body_size", "upper_wick", "lower_wick",
        "body_ratio", "range", "gap", "atr_ratio", "bullish", "bearish", "neutral"
    ],
    "CANDLE_PATTERNS": [
        "doji", "hammer", "shooting_star", "engulfing", "harami",
        "morning_star", "evening_star", "inside_bar", "outside_bar"
    ],
    "TREND": [
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "wma_5", "wma_10", "wma_20", "wma_50", "wma_100",
        "hull_ma", "kama", "vwap", "supertrend", "adx", "di_plus", "di_minus",
        "psar", "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b", "ichimoku_chikou", "regression_trend",
        "trend_strength", "trend_slope"
    ],
    "MOMENTUM": [
        "rsi", "macd", "macd_signal", "macd_histogram", "roc", "momentum",
        "cci", "williams_r", "tsi", "stoch_k", "stoch_d",
        "awesome_oscillator", "ultimate_oscillator"
    ],
    "VOLATILITY": [
        "atr", "rolling_std", "rolling_variance", "historical_volatility",
        "parkinson_volatility", "yang_zhang_volatility", "garman_klass_volatility",
        "volatility_percentile", "volatility_regime"
    ],
    "VOLUME": [
        "obv", "accumulation_distribution", "mfi", "cmf", "volume_delta",
        "volume_zscore"
    ],
    "LIQUIDITY": [
        "spread", "spread_ma", "spread_median", "spread_std", "liquidity_score",
        "estimated_slippage"
    ],
    "MARKET_STRUCTURE": [
        "higher_high", "higher_low", "lower_high", "lower_low",
        "swing_high", "swing_low", "bos", "choc", "order_block",
        "breaker_block", "mitigation_block", "fair_value_gap",
        "liquidity_sweep", "equal_high", "equal_low", "premium_zone",
        "discount_zone"
    ],
    "DISTANCE": [
        "dist_to_ema_20", "dist_to_ema_50", "dist_to_vwap",
        "dist_to_daily_high", "dist_to_daily_low",
        "dist_to_weekly_high", "dist_to_weekly_low",
        "dist_to_monthly_high", "dist_to_monthly_low"
    ],
    "SESSION": [
        "hour", "minute", "weekday", "week", "month", "quarter",
        "is_london_session", "is_newyork_session", "is_tokyo_session",
        "is_sydney_session", "is_session_overlap", "is_market_open",
        "is_market_close", "is_weekend", "is_holiday"
    ],
    "CORRELATIONS": {
        symbol: f"correlation_{symbol}" for symbol in CORRELATION_SYMBOLS.keys()
    },
    "STATISTICS": [
        "rolling_mean", "rolling_median", "rolling_variance", "rolling_std",
        "entropy", "autocorrelation", "rolling_rank", "rolling_percentile",
        "zscore", "skewness", "kurtosis", "hurst_exponent", "fractal_dimension"
    ],
    "RISK": [
        "rolling_drawdown", "max_drawdown", "ulcer_index", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "risk_score"
    ],
}

# ============================================================================
# DATA TYPES
# ============================================================================

# Parquet data types for efficient storage
PARQUET_DTYPES = {
    "timestamp": "int64",  # nanoseconds since epoch
    "timeframe": "string",
    "symbol": "string",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "tick_volume": "int32",
    "real_volume": "float32",
    "spread": "float32",
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

# OHLC relationships
OHLC_RULES = {
    "high_gte_low": True,
    "high_gte_open": True,
    "high_gte_close": True,
    "low_lte_open": True,
    "low_lte_close": True,
}

# ============================================================================
# REPORTING
# ============================================================================

# Report sections to generate
REPORT_SECTIONS = [
    "summary",
    "data_quality",
    "missing_data",
    "outliers",
    "statistics",
    "leakage_analysis",
    "feature_importance",
    "label_distribution",
]

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

# Verbose output
VERBOSE = True

# Random seed for reproducibility
RANDOM_SEED = 42

# Version tracking
FACTORY_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"

# Checkpoint frequency (every N bars processed)
CHECKPOINT_FREQUENCY = 100000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_timeframe_minutes(timeframe: str) -> int:
    """Get minutes for a given timeframe string."""
    return MINUTES_PER_TIMEFRAME.get(timeframe, 1)


def get_timeframe_candles_per_day(timeframe: str) -> int:
    """Get approximate candles per day for a timeframe."""
    return int(CANDLES_PER_DAY.get(timeframe, 1))


def validate_timeframe(timeframe: str) -> bool:
    """Validate if timeframe is supported."""
    return timeframe in TIMEFRAME_MAPPING


def get_all_symbols_config() -> Set[str]:
    """Get all configured symbols across all categories."""
    symbols = set()
    for symbol_list in [
        MAJOR_PAIRS, EXOTIC_PAIRS, CROSS_SYMBOLS, CRYPTO_SYMBOLS, INDEX_SYMBOLS
    ]:
        symbols.update(symbol_list)
    return symbols


def get_data_path(entity: str, timeframe: str = "", symbol: str = "") -> Path:
    """Construct a standardized data path."""
    if entity == "raw":
        base = RAW_DATA_DIR
    elif entity == "cleaned":
        base = CLEANED_DATA_DIR
    elif entity == "validated":
        base = VALIDATED_DATA_DIR
    elif entity == "features":
        base = FEATURES_DIR
    elif entity == "labels":
        base = LABELS_DIR
    else:
        base = OUTPUT_DIR
    
    if timeframe and symbol:
        return base / timeframe / f"{symbol}.parquet"
    elif symbol:
        return base / f"{symbol}.parquet"
    else:
        return base


# ============================================================================
# SYSTEM VERIFICATION
# ============================================================================

if __name__ == "__main__":
    print("✓ Configuration module loaded successfully")
    print(f"✓ Project root: {PROJECT_ROOT}")
    print(f"✓ Data directory: {DATA_DIR}")
    print(f"✓ Logs directory: {LOGS_DIR}")
    print(f"✓ Supported timeframes: {len(PRIMARY_TIMEFRAMES)}")
    print(f"✓ Indicator periods configured: {len(INDICATOR_PERIODS)}")
    print(f"✓ Feature groups: {len(FEATURE_GROUPS)}")
    print(f"✓ Max workers: {MAX_WORKERS}")
    print(f"✓ Parquet compression: {PARQUET_COMPRESSION}")
    print("✓ All directories created and ready")
