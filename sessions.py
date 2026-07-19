"""
Trading Session Detection Module for Institutional Quantitative Data Factory.

Provides trading session identification, session overlap detection, and
market-hour feature generation for Forex data.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple
import pytz

from config import MARKET_SESSIONS, SESSION_OVERLAPS, FOREX_HOLIDAYS, EARLY_CLOSE_DAYS
from logger import pipeline_logger

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manages trading session detection and features."""
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions = MARKET_SESSIONS
        self.overlaps = SESSION_OVERLAPS
        self.holidays = FOREX_HOLIDAYS
        self.early_closes = EARLY_CLOSE_DAYS
    
    def get_session_name(self, dt: datetime) -> str:
        """Get session name for datetime (UTC)."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        # Define session times in UTC
        if 21 <= hour or hour < 6:
            return 'Sydney'
        elif 23 <= hour or hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 17:
            return 'London'
        elif 13 <= hour < 22:
            return 'NewYork'
        else:
            return 'Unknown'
    
    def is_london_session(self, dt: datetime) -> bool:
        """Check if datetime falls in London session."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        return 8 <= hour < 17
    
    def is_newyork_session(self, dt: datetime) -> bool:
        """Check if datetime falls in New York session."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        return 13 <= hour < 22
    
    def is_tokyo_session(self, dt: datetime) -> bool:
        """Check if datetime falls in Tokyo session."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        return 23 <= hour or hour < 8
    
    def is_sydney_session(self, dt: datetime) -> bool:
        """Check if datetime falls in Sydney session."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        return 21 <= hour or hour < 6
    
    def is_session_overlap(self, dt: datetime) -> bool:
        """Check if datetime falls in overlapping sessions."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        hour = dt_utc.hour
        
        # Sydney-Tokyo overlap: 23:00-06:00 UTC
        # Tokyo-London overlap: 23:00-08:00 UTC (overlaps with Sydney)
        # London-NY overlap: 13:00-17:00 UTC
        # NY-Sydney overlap: 22:00-21:00 UTC (next day, wraps)
        
        overlapping_hours = [
            (23, 24),  # Sydney-Tokyo
            (13, 17),  # London-NY
        ]
        
        for start, end in overlapping_hours:
            if start <= hour < end:
                return True
        
        # Handle wrap-around
        if hour < 6:  # Early morning Sydney-Tokyo
            return True
        
        return False
    
    def is_market_open(self, dt: datetime) -> bool:
        """Check if Forex market is open (Sunday 21:00 UTC - Friday 21:00 UTC)."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        # Market is closed Saturday-Sunday (except Sunday 21:00 UTC start)
        weekday = dt.weekday()  # Monday=0, Sunday=6
        
        if weekday == 5:  # Saturday
            return False
        
        if weekday == 6:  # Sunday
            dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
            return dt_utc.hour >= 21
        
        return True
    
    def is_market_close(self, dt: datetime) -> bool:
        """Check if it's market close time (Friday 21:00 UTC)."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        dt_utc = dt.tz_convert('UTC') if dt.tz else dt.replace(tzinfo=pytz.UTC)
        
        return dt_utc.weekday() == 4 and dt_utc.hour >= 21  # Friday >= 21:00
    
    def is_weekend(self, dt: datetime) -> bool:
        """Check if datetime is weekend."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        weekday = dt.weekday()
        return weekday >= 5
    
    def is_holiday(self, dt: datetime) -> bool:
        """Check if datetime is Forex holiday."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        date_tuple = (dt.month, dt.day)
        return date_tuple in self.holidays
    
    def is_early_close(self, dt: datetime) -> bool:
        """Check if datetime is early close day."""
        if not isinstance(dt, pd.Timestamp):
            dt = pd.Timestamp(dt)
        
        date_tuple = (dt.month, dt.day)
        return date_tuple in self.early_closes


# ============================================================================
# SESSION FEATURE GENERATOR
# ============================================================================

class SessionFeatureGenerator:
    """Generates session-based features from timestamps."""
    
    def __init__(self):
        """Initialize session feature generator."""
        self.session_manager = SessionManager()
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session-based features to DataFrame."""
        
        if df.empty or 'time' not in df.columns:
            pipeline_logger.warning("Cannot add session features: no time column")
            return df
        
        df = df.copy()
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Ensure UTC timezone
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            df['time'] = df['time'].dt.tz_convert('UTC')
        
        # Time components
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['weekday'] = df['time'].dt.weekday
        df['week'] = df['time'].dt.isocalendar().week
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        
        # Session indicators
        df['is_london_session'] = df['time'].apply(self.session_manager.is_london_session).astype(int)
        df['is_newyork_session'] = df['time'].apply(self.session_manager.is_newyork_session).astype(int)
        df['is_tokyo_session'] = df['time'].apply(self.session_manager.is_tokyo_session).astype(int)
        df['is_sydney_session'] = df['time'].apply(self.session_manager.is_sydney_session).astype(int)
        df['is_session_overlap'] = df['time'].apply(self.session_manager.is_session_overlap).astype(int)
        
        # Market status
        df['is_market_open'] = df['time'].apply(self.session_manager.is_market_open).astype(int)
        df['is_market_close'] = df['time'].apply(self.session_manager.is_market_close).astype(int)
        df['is_weekend'] = df['time'].apply(self.session_manager.is_weekend).astype(int)
        df['is_holiday'] = df['time'].apply(self.session_manager.is_holiday).astype(int)
        
        return df
    
    def get_session_name_vectorized(self, timestamps: pd.Series) -> pd.Series:
        """Vectorized session name detection."""
        dt_utc = timestamps.dt.tz_convert('UTC') if timestamps.dt.tz else timestamps.dt.tz_localize('UTC')
        hours = dt_utc.dt.hour
        
        sessions = pd.Series(index=timestamps.index, dtype='object')
        sessions[(hours >= 21) | (hours < 6)] = 'Sydney'
        sessions[(hours >= 23) | (hours < 8)] = 'Tokyo'
        sessions[(hours >= 8) & (hours < 17)] = 'London'
        sessions[(hours >= 13) & (hours < 22)] = 'NewYork'
        sessions[sessions.isna()] = 'Unknown'
        
        return sessions


# ============================================================================
# SESSION STATISTICS
# ============================================================================

class SessionStatistics:
    """Calculate statistics by trading session."""
    
    def __init__(self):
        """Initialize session statistics calculator."""
        self.session_manager = SessionManager()
        self.feature_generator = SessionFeatureGenerator()
    
    def get_session_stats(
        self,
        df: pd.DataFrame,
        session_name: str
    ) -> Dict[str, float]:
        """Calculate statistics for a specific session."""
        
        if 'time' not in df.columns:
            return {}
        
        # Get session data
        if session_name == 'London':
            session_df = df[df['is_london_session'] == 1]
        elif session_name == 'NewYork':
            session_df = df[df['is_newyork_session'] == 1]
        elif session_name == 'Tokyo':
            session_df = df[df['is_tokyo_session'] == 1]
        elif session_name == 'Sydney':
            session_df = df[df['is_sydney_session'] == 1]
        else:
            return {}
        
        if session_df.empty:
            return {}
        
        # Calculate statistics
        stats = {
            'candle_count': len(session_df),
            'avg_volume': float(session_df['tick_volume'].mean()) if 'tick_volume' in session_df.columns else 0,
            'avg_spread': float(session_df['spread'].mean()) if 'spread' in session_df.columns else 0,
            'avg_range': float((session_df['high'] - session_df['low']).mean()) if all(col in session_df.columns for col in ['high', 'low']) else 0,
            'volatility': float(session_df['close'].pct_change().std()) if 'close' in session_df.columns else 0,
        }
        
        return stats
    
    def compare_sessions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compare statistics across all sessions."""
        
        if 'is_london_session' not in df.columns:
            df = self.feature_generator.add_session_features(df)
        
        comparison = {}
        for session in ['Sydney', 'Tokyo', 'London', 'NewYork']:
            comparison[session] = self.get_session_stats(df, session)
        
        return comparison
    
    def get_peak_hours(self, df: pd.DataFrame, session_name: str) -> List[int]:
        """Get peak trading hours for a session."""
        
        if 'hour' not in df.columns or 'is_market_open' not in df.columns:
            df = self.feature_generator.add_session_features(df)
        
        # Get session data
        if session_name == 'London':
            session_df = df[df['is_london_session'] == 1]
        elif session_name == 'NewYork':
            session_df = df[df['is_newyork_session'] == 1]
        elif session_name == 'Tokyo':
            session_df = df[df['is_tokyo_session'] == 1]
        elif session_name == 'Sydney':
            session_df = df[df['is_sydney_session'] == 1]
        else:
            return []
        
        if session_df.empty or 'tick_volume' not in session_df.columns:
            return []
        
        # Group by hour and calculate average volume
        hourly_volume = session_df.groupby('hour')['tick_volume'].mean()
        
        # Get top 3 hours
        peak_hours = hourly_volume.nlargest(3).index.tolist()
        
        return peak_hours


if __name__ == "__main__":
    # Test session manager
    manager = SessionManager()
    
    # Test timestamps
    test_times = [
        pd.Timestamp('2023-01-02 05:00:00', tz='UTC'),  # Sydney
        pd.Timestamp('2023-01-02 12:00:00', tz='UTC'),  # Tokyo-London overlap
        pd.Timestamp('2023-01-02 15:00:00', tz='UTC'),  # London
        pd.Timestamp('2023-01-02 18:00:00', tz='UTC'),  # London-NY overlap
    ]
    
    print("\nSession Detection:")
    for ts in test_times:
        session = manager.get_session_name(ts)
        print(f"  {ts}: {session}")
    
    # Test feature generation
    dates = pd.date_range('2023-01-02', periods=24, freq='1H', tz='UTC')
    test_df = pd.DataFrame({
        'time': dates,
        'close': np.random.randn(24).cumsum() + 100,
        'high': np.random.randn(24).cumsum() + 101,
        'low': np.random.randn(24).cumsum() + 99,
        'tick_volume': np.random.randint(100, 1000, 24),
        'spread': np.random.uniform(0.0001, 0.0002, 24),
    })
    
    generator = SessionFeatureGenerator()
    test_df_features = generator.add_session_features(test_df)
    
    print("\nSession Features Added:")
    print(f"  Columns: {[col for col in test_df_features.columns if 'session' in col or 'market' in col or 'hour' in col]}")
