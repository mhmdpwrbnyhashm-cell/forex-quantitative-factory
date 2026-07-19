"""
Production Logging Module for Institutional Quantitative Data Factory.

Provides comprehensive logging, monitoring, and error tracking across all
data pipeline stages with automatic rotation, formatting, and alerts.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import traceback
import json

from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE,
    MAX_LOG_SIZE_MB, BACKUP_COUNT, LOGS_DIR, VERBOSE
)

# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color output for console logging."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[Path]
        File path for file handler. If None, uses default LOG_FILE
    structured : bool
        Use structured JSON formatting
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_path = log_file or LOG_FILE
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(file_path),
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    if structured:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class PipelineLogger:
    """High-level logger for data pipeline stages."""
    
    def __init__(self, name: str = "QuantDataFactory"):
        """Initialize pipeline logger."""
        self.logger = setup_logger(name)
        self.metrics: Dict[str, Any] = {}
        self.errors: list = []
        self.warnings: list = []
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        if VERBOSE:
            if kwargs:
                extra = logging.LogRecordFactory()
                extra.extra_data = kwargs
                self.logger.info(message, extra=extra)
            else:
                self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message and track it."""
        self.warnings.append({'message': message, 'timestamp': datetime.now()})
        if kwargs:
            extra = logging.LogRecordFactory()
            extra.extra_data = kwargs
            self.logger.warning(message, extra=extra)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message and track it."""
        self.errors.append({
            'message': message,
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc() if exc_info else None
        })
        if kwargs:
            extra = logging.LogRecordFactory()
            extra.extra_data = kwargs
            self.logger.error(message, extra=extra, exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        if kwargs:
            extra = logging.LogRecordFactory()
            extra.extra_data = kwargs
            self.logger.debug(message, extra=extra)
        else:
            self.logger.debug(message)
    
    def record_metric(self, metric_name: str, value: Any) -> None:
        """Record a performance metric."""
        self.metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Metric recorded: {metric_name}={value}")
    
    def log_stage_start(self, stage_name: str, details: Dict[str, Any] = None) -> None:
        """Log pipeline stage start."""
        msg = f"\n{'='*80}\nSTART: {stage_name}\n{'='*80}"
        self.logger.info(msg)
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_stage_end(
        self,
        stage_name: str,
        duration_seconds: float,
        status: str = "SUCCESS",
        details: Dict[str, Any] = None
    ) -> None:
        """Log pipeline stage completion."""
        msg = f"\n{'='*80}\nEND: {stage_name} | Status: {status} | Duration: {duration_seconds:.2f}s\n{'='*80}"
        
        if status == "SUCCESS":
            self.logger.info(msg)
        elif status == "WARNING":
            self.logger.warning(msg)
        else:
            self.logger.error(msg)
        
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_data_summary(
        self,
        symbol: str,
        timeframe: str,
        candle_count: int,
        missing_count: int = 0,
        duplicate_count: int = 0
    ) -> None:
        """Log data summary for symbol/timeframe."""
        msg = (
            f"Data Summary | Symbol: {symbol} | Timeframe: {timeframe} | "
            f"Candles: {candle_count} | Missing: {missing_count} | Duplicates: {duplicate_count}"
        )
        self.logger.info(msg)
    
    def log_feature_generation(
        self,
        symbol: str,
        feature_count: int,
        duration_seconds: float
    ) -> None:
        """Log feature generation progress."""
        rate = feature_count / duration_seconds if duration_seconds > 0 else 0
        msg = (
            f"Features Generated | Symbol: {symbol} | "
            f"Features: {feature_count} | Duration: {duration_seconds:.2f}s | "
            f"Rate: {rate:.0f} features/sec"
        )
        self.logger.info(msg)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all logged errors."""
        return {
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors[-10:],  # Last 10 errors
            'warnings': self.warnings[-10:]  # Last 10 warnings
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics."""
        return self.metrics
    
    def export_logs(self, output_path: Path) -> None:
        """Export log summary to file."""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'errors': self.get_error_summary(),
            'metrics': self.get_metrics_summary(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: PipelineLogger):
        """Initialize performance monitor."""
        self.logger = logger
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[f"{name}_start"] = datetime.now().timestamp()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds."""
        start_key = f"{name}_start"
        if start_key not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = datetime.now().timestamp() - self.timers[start_key]
        self.logger.record_metric(f"duration_{name}", elapsed)
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> int:
        """Increment a named counter."""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        return self.counters[name]
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def log_memory_usage(self, usage_mb: float) -> None:
        """Log current memory usage."""
        self.logger.record_metric("memory_usage_mb", usage_mb)
    
    def log_throughput(self, items: int, duration_seconds: float) -> None:
        """Log throughput (items per second)."""
        throughput = items / duration_seconds if duration_seconds > 0 else 0
        self.logger.record_metric("throughput_items_per_sec", throughput)


# ============================================================================
# GLOBAL LOGGER INSTANCES
# ============================================================================

# Main pipeline logger
main_logger = setup_logger("QuantDataFactory")
pipeline_logger = PipelineLogger("QuantDataFactory")
performance_monitor = PerformanceMonitor(pipeline_logger)

# Module-specific loggers
collector_logger = setup_logger("QuantDataFactory.Collector")
cleaner_logger = setup_logger("QuantDataFactory.Cleaner")
validator_logger = setup_logger("QuantDataFactory.Validator")
feature_logger = setup_logger("QuantDataFactory.Features")
label_logger = setup_logger("QuantDataFactory.Labels")
export_logger = setup_logger("QuantDataFactory.Export")


if __name__ == "__main__":
    # Test logging setup
    test_logger = PipelineLogger("TestLogger")
    test_logger.log_stage_start("Testing Logger", {'version': '1.0.0'})
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning")
    test_logger.debug("Debug info")
    test_logger.record_metric("test_metric", 42)
    test_logger.log_stage_end("Testing Logger", 0.5, "SUCCESS")
    
    # Test performance monitor
    monitor = PerformanceMonitor(test_logger)
    monitor.start_timer("test_operation")
    monitor.increment_counter("processed_items", 100)
    elapsed = monitor.stop_timer("test_operation")
    test_logger.info(f"Test operation completed in {elapsed:.4f}s")
