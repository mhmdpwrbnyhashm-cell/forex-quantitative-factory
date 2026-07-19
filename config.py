"""Production Configuration Module for Institutional Quantitative Data Factory.

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
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / f"factory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_LOG_SIZE_MB = 100
BACKUP_COUNT = 5

# ============================================================================
# METATRADER5 CONNECTION
# ============================================================================

MT5_TIMEOUT = 60000
MT5_RECONNECT_ATTEMPTS = 3
MT5_RECONNECT_DELAY = 5

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

MAJOR_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
]

EXOTIC_PAIRS = [
    "USDSEK", "USDNOK", "USDDKK", "USDPLN", "USDCZK", "USDHUF",
    "SGDUSD", "HKDUSD", "ZARUSD", "ZARJPY",
]

CROSS_SYMBOLS = [
    "XAUUSD", "XAGUSD", "USOIL", "BRENT", "NATGAS",
]

CRYPTO_SYMBOLS = [
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
]

INDEX_SYMBOLS = [
    "SP500", "NASDAQ", "DAX", "FTSE", "HSI", "N225", "ASX200",
]

CORRELATION_SYMBOLS = {
    "EURUSD": 1.0, "GBPUSD": 1.0, "USDJPY": 1.0, "USDCHF": 1.0,
    "AUDUSD": 1.0, "NZDUSD": 1.0, "USDCAD": 1.0, "XAUUSD": 1.0,
    "DXY": 1.0, "USOIL": 1.0, "SP500": 1.0, "NASDAQ": 1.0,
    "BTCUSD": 1.0, "ETHUSD": 1.0,
}

# ============================================================================
# TIMEFRAMES
# ============================================================================

class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN = "MN"

TIMEFRAME_MAPPING = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60,
    "H4": 240, "D1": 1440, "W1": 10080, "MN": 43200,
}

CANDLES_PER_DAY = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24, "H4": 6, "D1": 1, "W1": 0.2, "MN": 0.033,
}

MINUTES_PER_TIMEFRAME = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60,
    "H4": 240, "D1": 1440, "W1": 10080, "MN": 43200,
}

PRIMARY_TIMEFRAMES = [
    Timeframe.M1, Timeframe.M5, Timeframe.M15,
    Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1,
]

HIGHER_TIMEFRAMES = [Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1]

# ============================================================================
# DATA COLLECTION
# ============================================================================

START_DATE = datetime(2015, 1, 1, tzinfo=pytz.UTC)
END_DATE = datetime.now(pytz.UTC)
BATCH_SIZE = 10000
MAX_DOWNLOAD_RETRIES = 3
CONNECTION_CHECK_INTERVAL = 300

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

INDICATOR_PERIODS = {
    "EMA_PERIODS": [5, 10, 20, 50, 100, 200],
    "SMA_PERIODS": [5, 10, 20, 50, 100, 200],
    "WMA_PERIODS": [5, 10, 20, 50, 100],
    "HULL_MA_PERIOD": 9,
    "KAMA_PERIOD": 10,
    "VWAP_RESET": "D1",
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
    "ATR_PERIOD": 14,
    "ROLLING_STD_PERIOD": 20,
    "HISTORICAL_VOL_PERIOD": 20,
    "VOLATILITY_PERCENTILE_PERIOD": 252,
    "OBV_PERIOD": 14,
    "CMF_PERIOD": 20,
    "MFI_PERIOD": 14,
    "VOLUME_ZSCORE_PERIOD": 20,
    "SUPERTREND_PERIOD": 10,
    "SUPERTREND_MULTIPLIER": 3.0,
    "ADX_PERIOD": 14,
    "ICHIMOKU_TENKAN": 9,
    "ICHIMOKU_KIJUN": 26,
    "ICHIMOKU_SENKOU_B": 52,
    "ICHIMOKU_CHIKOU": 26,
    "SAR_STEP": 0.02,
    "SAR_MAX_STEP": 0.2,
    "REGRESSION_PERIOD": 50,
    "SWING_PERIOD": 5,
    "FVG_LOOKBACK": 3,
    "CORRELATION_PERIOD": 252,
}

# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

MAX_SPREAD_BP = 500
MAX_GAP_PERCENT = 5.0
OUTLIER_THRESHOLD_STD = 5.0
MISSING_DATA_TOLERANCE = 5.0
DUPLICATE_TOLERANCE_MS = 100

# ============================================================================
# LABEL ENGINE CONFIGURATION
# ============================================================================

FUTURE_RETURN_PERIODS = [5, 10, 20, 50]
MFE_MAE_LOOKBACK = 20

TRIPLE_BARRIER = {
    "UPSIDE_BARRIER_STD": 2.0,
    "DOWNSIDE_BARRIER_STD": 2.0,
    "TIMEFRAME_BARS": 20,
}

META_LABEL_THRESHOLD = 0.55
MAX_HOLDING_TIME = 100

# ============================================================================
# QUALITY ASSURANCE
# ============================================================================

MIN_CANDLES_REQUIRED = 10000
MIN_QUALITY_SCORE = 80
LEAKAGE_CORRELATION_THRESHOLD = 0.95
LEAKAGE_LOOKAHEAD_BARS = 5

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

PARQUET_COMPRESSION = "snappy"
PARQUET_ROW_GROUP_SIZE = 100000
EXPORT_CHUNK_SIZE = 1000000
DEFAULT_ENCODING = "utf-8"

# ============================================================================
# PERFORMANCE & OPTIMIZATION
# ============================================================================

NUMPY_CHUNK_SIZE = 100000
MAX_WORKERS = min(os.cpu_count() or 4, 8)
MAX_MEMORY_PERCENT = 80
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

SESSION_OVERLAPS = {
    "Sydney-Tokyo": ("23:00", "06:00"),
    "Tokyo-London": ("08:00", "08:00"),
    "London-New York": ("13:00", "17:00"),
    "New York-Sydney": ("22:00", "21:00"),
}

# ============================================================================
# HOLIDAYS & SPECIAL DAYS
# ============================================================================

FOREX_HOLIDAYS = {
    (1, 1): "New Year",
    (1, 2): "New Year Holiday",
    (12, 25): "Christmas",
    (12, 26): "Boxing Day",
}

EARLY_CLOSE_DAYS = {
    (12, 24): "Christmas Eve",
    (12, 31): "New Year's Eve",
}

# ============================================================================
# SYSTEM SETTINGS
# ============================================================================

VERBOSE = True
RANDOM_SEED = 42
FACTORY_VERSION = "1.0.0"
DATASET_VERSION = "1.0.0"
CHECKPOINT_FREQUENCY = 100000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_timeframe_minutes(timeframe: str) -> int:
    return MINUTES_PER_TIMEFRAME.get(timeframe, 1)

def get_timeframe_candles_per_day(timeframe: str) -> int:
    return int(CANDLES_PER_DAY.get(timeframe, 1))

def validate_timeframe(timeframe: str) -> bool:
    return timeframe in TIMEFRAME_MAPPING

def get_all_symbols_config() -> Set[str]:
    symbols = set()
    for symbol_list in [MAJOR_PAIRS, EXOTIC_PAIRS, CROSS_SYMBOLS, CRYPTO_SYMBOLS, INDEX_SYMBOLS]:
        symbols.update(symbol_list)
    return symbols

def get_data_path(entity: str, timeframe: str = "", symbol: str = "") -> Path:
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

if __name__ == "__main__":
    print("✓ Configuration module loaded successfully")
    print(f"✓ Project root: {PROJECT_ROOT}")
    print(f"✓ Data directory: {DATA_DIR}")
    print(f"✓ Logs directory: {LOGS_DIR}")
