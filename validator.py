"""
Data Validation Module for Institutional Quantitative Data Factory.

Performs final data quality checks, completeness validation, and generates
quality reports before feature engineering.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from config import (
    VALIDATED_DATA_DIR, MIN_CANDLES_REQUIRED, MIN_QUALITY_SCORE,
    PRIMARY_TIMEFRAMES, MINUTES_PER_TIMEFRAME
)
from logger import pipeline_logger, performance_monitor
from utils import safe_save_parquet, safe_load_parquet, ensure_directory

# ============================================================================
# DATA VALIDATOR
# ============================================================================

class DataValidator:
    """Validates cleaned data quality and completeness."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_results = {}
    
    def check_completeness(
        self,
        df: pd.DataFrame,
        expected_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """Check data completeness."""
        issues = []
        
        if df is None or df.empty:
            return False, ["DataFrame is empty"]
        
        # Check columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isnull().sum()
            issues.append(f"NaN values in: {dict(nan_counts)}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            issues.append(f"Infinite values in: {inf_cols}")
        
        return len(issues) == 0, issues
    
    def check_minimum_data(
        self,
        df: pd.DataFrame,
        min_rows: int = MIN_CANDLES_REQUIRED
    ) -> Tuple[bool, str]:
        """Check if minimum data requirements are met."""
        if len(df) < min_rows:
            msg = f"Insufficient data: {len(df)} < {min_rows} required"
            return False, msg
        
        return True, f"Sufficient data: {len(df)} rows"
    
    def check_data_continuity(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int,
        tolerance_percent: float = 10.0
    ) -> Tuple[float, str]:
        """Check continuity of time series."""
        if df.empty or 'time' not in df.columns:
            return 0.0, "Cannot check continuity"
        
        df = df.sort_values('time')
        times = pd.to_datetime(df['time'])
        
        # Calculate expected candle count
        time_span = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60  # minutes
        expected_candles = time_span / timeframe_minutes
        
        actual_candles = len(df)
        continuity_percent = (actual_candles / expected_candles * 100) if expected_candles > 0 else 0
        
        is_continuous = continuity_percent >= (100 - tolerance_percent)
        
        msg = f"Continuity: {continuity_percent:.1f}% "
        msg += "(OK)" if is_continuous else "(WARNING)"
        
        return continuity_percent, msg
    
    def check_price_sanity(
        self,
        df: pd.DataFrame,
        max_price_change_percent: float = 10.0
    ) -> Tuple[bool, str]:
        """Check price sanity."""
        if df.empty or 'close' not in df.columns:
            return True, "Cannot check prices"
        
        # Check consecutive price changes
        returns = df['close'].pct_change()
        extreme_returns = np.abs(returns) > (max_price_change_percent / 100)
        
        if extreme_returns.any():
            count = extreme_returns.sum()
            msg = f"Found {count} extreme price changes (>{max_price_change_percent}%)"
            return False, msg
        
        return True, "Price sanity check passed"
    
    def check_volume_sanity(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Check volume sanity."""
        if df.empty or 'tick_volume' not in df.columns:
            return True, "No volume data"
        
        volume = df['tick_volume']
        
        # Check for all zeros
        if (volume == 0).all():
            return False, "All volumes are zero"
        
        # Check for unrealistic volume spikes
        mean_vol = volume.mean()
        std_vol = volume.std()
        if std_vol > 0:
            z_scores = np.abs((volume - mean_vol) / std_vol)
            extreme_vols = (z_scores > 5).sum()
            
            if extreme_vols > len(volume) * 0.05:  # More than 5% outliers
                return False, f"Excessive volume outliers: {extreme_vols}"
        
        return True, "Volume sanity check passed"
    
    def check_spread_stats(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate spread statistics."""
        if df.empty or 'spread' not in df.columns:
            return {}
        
        spread = df['spread']
        
        return {
            'mean': float(spread.mean()),
            'median': float(spread.median()),
            'std': float(spread.std()),
            'min': float(spread.min()),
            'max': float(spread.max()),
            'q95': float(spread.quantile(0.95)),
        }
    
    def calculate_quality_score(
        self,
        df: pd.DataFrame,
        timeframe_minutes: int,
        symbol: str = ""
    ) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        if df is None or df.empty:
            return 0.0
        
        # Size penalty
        if len(df) < MIN_CANDLES_REQUIRED:
            size_penalty = (1 - len(df) / MIN_CANDLES_REQUIRED) * 20
            score -= min(size_penalty, 20)
        
        # Completeness penalty
        nan_percent = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        score -= min(nan_percent * 0.5, 15)
        
        # Continuity penalty
        continuity, _ = self.check_data_continuity(df, timeframe_minutes)
        if continuity < 90:
            score -= (100 - continuity) * 0.2
        
        # Price sanity
        sane, _ = self.check_price_sanity(df)
        if not sane:
            score -= 15
        
        # Volume sanity
        vol_sane, _ = self.check_volume_sanity(df)
        if not vol_sane:
            score -= 10
        
        return max(score, 0.0)
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        expected_columns: List[str] = None
    ) -> Dict[str, any]:
        """Run complete validation suite."""
        
        timeframe_minutes = MINUTES_PER_TIMEFRAME.get(timeframe, 1)
        
        if expected_columns is None:
            expected_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'row_count': len(df) if df is not None else 0,
        }
        
        if df is None or df.empty:
            results['quality_score'] = 0.0
            results['status'] = 'INVALID'
            return results
        
        # Run checks
        completeness_ok, completeness_issues = self.check_completeness(df, expected_columns)
        results['completeness_check'] = {
            'ok': completeness_ok,
            'issues': completeness_issues
        }
        
        minimum_ok, minimum_msg = self.check_minimum_data(df)
        results['minimum_data_check'] = {
            'ok': minimum_ok,
            'message': minimum_msg
        }
        
        continuity, continuity_msg = self.check_data_continuity(df, timeframe_minutes)
        results['continuity_check'] = {
            'score': continuity,
            'message': continuity_msg
        }
        
        price_sane, price_msg = self.check_price_sanity(df)
        results['price_sanity'] = {
            'ok': price_sane,
            'message': price_msg
        }
        
        volume_sane, volume_msg = self.check_volume_sanity(df)
        results['volume_sanity'] = {
            'ok': volume_sane,
            'message': volume_msg
        }
        
        spread_stats = self.check_spread_stats(df)
        results['spread_stats'] = spread_stats
        
        # Overall quality score
        quality_score = self.calculate_quality_score(df, timeframe_minutes, symbol)
        results['quality_score'] = quality_score
        
        # Overall status
        if quality_score >= MIN_QUALITY_SCORE:
            results['status'] = 'VALID'
        else:
            results['status'] = 'INVALID'
        
        return results


# ============================================================================
# BATCH VALIDATOR
# ============================================================================

class BatchValidator:
    """Validates multiple datasets and generates reports."""
    
    def __init__(self, cleaned_dir: Path, output_dir: Path = VALIDATED_DATA_DIR):
        """Initialize batch validator."""
        self.cleaned_dir = cleaned_dir
        self.output_dir = ensure_directory(output_dir)
        self.validator = DataValidator()
        self.all_results = []
    
    def validate_symbol_timeframe(
        self,
        symbol: str,
        timeframe: str,
        cleaned_file: Path
    ) -> Dict[str, any]:
        """Validate single symbol/timeframe."""
        
        df = safe_load_parquet(cleaned_file)
        
        results = self.validator.validate_dataset(df, symbol, timeframe)
        
        # Save if valid
        if results['status'] == 'VALID':
            output_file = self.output_dir / timeframe / f"{symbol}.parquet"
            if df is not None:
                safe_save_parquet(df, output_file)
        
        return results
    
    def validate_all_data(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ) -> List[Dict]:
        """Validate all cleaned data files."""
        
        if timeframes is None:
            timeframes = [tf.value for tf in PRIMARY_TIMEFRAMES]
        
        self.all_results = []
        
        for timeframe in timeframes:
            timeframe_dir = self.cleaned_dir / timeframe
            
            if not timeframe_dir.exists():
                continue
            
            parquet_files = list(timeframe_dir.glob('*.parquet'))
            
            for file in parquet_files:
                symbol = file.stem
                
                if symbols and symbol not in symbols:
                    continue
                
                results = self.validate_symbol_timeframe(symbol, timeframe, file)
                self.all_results.append(results)
        
        return self.all_results
    
    def generate_validation_report(
        self,
        output_path: Optional[Path] = None
    ) -> Dict[str, any]:
        """Generate comprehensive validation report."""
        
        if not self.all_results:
            return {}
        
        results_df = pd.DataFrame(self.all_results)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_files': len(self.all_results),
            'valid_files': len(results_df[results_df['status'] == 'VALID']),
            'invalid_files': len(results_df[results_df['status'] == 'INVALID']),
            'average_quality_score': float(results_df['quality_score'].mean()),
            'min_quality_score': float(results_df['quality_score'].min()),
            'max_quality_score': float(results_df['quality_score'].max()),
            'average_row_count': float(results_df['row_count'].mean()),
            'by_status': results_df['status'].value_counts().to_dict(),
            'detailed_results': self.all_results
        }
        
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                pipeline_logger.info(f"Validation report saved to {output_path}")
            except Exception as e:
                pipeline_logger.error(f"Error saving report: {e}")
        
        return report
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Get summary statistics from validation."""
        
        if not self.all_results:
            return {}
        
        results_df = pd.DataFrame(self.all_results)
        
        valid_df = results_df[results_df['status'] == 'VALID']
        
        return {
            'total_datasets': len(results_df),
            'valid_datasets': len(valid_df),
            'invalid_datasets': len(results_df) - len(valid_df),
            'validation_pass_rate': len(valid_df) / len(results_df) * 100,
            'quality_score_stats': {
                'mean': float(results_df['quality_score'].mean()),
                'median': float(results_df['quality_score'].median()),
                'std': float(results_df['quality_score'].std()),
                'min': float(results_df['quality_score'].min()),
                'max': float(results_df['quality_score'].max()),
            },
            'row_count_stats': {
                'mean': float(results_df['row_count'].mean()),
                'median': float(results_df['row_count'].median()),
                'min': int(results_df['row_count'].min()),
                'max': int(results_df['row_count'].max()),
            }
        }


if __name__ == "__main__":
    # Test validator
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=10000, freq='1H')
    test_data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(10000).cumsum() + 100,
        'high': np.random.randn(10000).cumsum() + 101,
        'low': np.random.randn(10000).cumsum() + 99,
        'close': np.random.randn(10000).cumsum() + 100,
        'tick_volume': np.random.randint(100, 1000, 10000),
        'spread': np.random.uniform(0.0001, 0.0002, 10000),
    })
    
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    validator = DataValidator()
    results = validator.validate_dataset(test_data, 'EURUSD', 'H1')
    
    print(f"\nValidation Results:")
    print(f"Status: {results['status']}")
    print(f"Quality Score: {results['quality_score']:.1f}")
    print(f"Row Count: {results['row_count']}")
