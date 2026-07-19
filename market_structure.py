"""
Market Structure Analysis Module for Institutional Quantitative Data Factory.

Detects swing highs/lows, order blocks, fair value gaps, and other key market
structure elements for advanced price action analysis.

Author: Lead Quantitative Architect
Version: 1.0.0
Python: 3.12+
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from logger import pipeline_logger

# ============================================================================
# MARKET STRUCTURE ANALYZER
# ============================================================================

class MarketStructureAnalyzer:
    """Analyzes market structure patterns from price data."""
    
    def __init__(self, min_swing_period: int = 5):
        """Initialize market structure analyzer."""
        self.min_swing_period = min_swing_period
    
    def find_swing_highs(
        self,
        df: pd.DataFrame,
        period: int = None,
        use_high_column: bool = True
    ) -> np.ndarray:
        """Find swing highs in price data."""
        if period is None:
            period = self.min_swing_period
        
        if df.empty or len(df) < period * 2 + 1:
            return np.zeros(len(df), dtype=bool)
        
        price_col = 'high' if use_high_column and 'high' in df.columns else 'close'
        prices = df[price_col].values
        
        swing_highs = np.zeros(len(df), dtype=bool)
        
        for i in range(period, len(df) - period):
            if prices[i] == prices[i:i+period+1].max() and prices[i] >= prices[i-period:i].max():
                swing_highs[i] = True
        
        return swing_highs
    
    def find_swing_lows(
        self,
        df: pd.DataFrame,
        period: int = None,
        use_low_column: bool = True
    ) -> np.ndarray:
        """Find swing lows in price data."""
        if period is None:
            period = self.min_swing_period
        
        if df.empty or len(df) < period * 2 + 1:
            return np.zeros(len(df), dtype=bool)
        
        price_col = 'low' if use_low_column and 'low' in df.columns else 'close'
        prices = df[price_col].values
        
        swing_lows = np.zeros(len(df), dtype=bool)
        
        for i in range(period, len(df) - period):
            if prices[i] == prices[i:i+period+1].min() and prices[i] <= prices[i-period:i].min():
                swing_lows[i] = True
        
        return swing_lows
    
    def find_higher_highs(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> np.ndarray:
        """Identify higher highs (uptrend confirmation)."""
        if df.empty or 'high' not in df.columns or len(df) < lookback:
            return np.zeros(len(df), dtype=bool)
        
        highs = df['high'].values
        higher_highs = np.zeros(len(df), dtype=bool)
        
        for i in range(lookback, len(df)):
            current_high = highs[i]
            previous_high = highs[i-lookback:i].max()
            if current_high > previous_high:
                higher_highs[i] = True
        
        return higher_highs
    
    def find_higher_lows(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> np.ndarray:
        """Identify higher lows (uptrend confirmation)."""
        if df.empty or 'low' not in df.columns or len(df) < lookback:
            return np.zeros(len(df), dtype=bool)
        
        lows = df['low'].values
        higher_lows = np.zeros(len(df), dtype=bool)
        
        for i in range(lookback, len(df)):
            current_low = lows[i]
            previous_low = lows[i-lookback:i].min()
            if current_low > previous_low:
                higher_lows[i] = True
        
        return higher_lows
    
    def find_lower_highs(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> np.ndarray:
        """Identify lower highs (downtrend confirmation)."""
        if df.empty or 'high' not in df.columns or len(df) < lookback:
            return np.zeros(len(df), dtype=bool)
        
        highs = df['high'].values
        lower_highs = np.zeros(len(df), dtype=bool)
        
        for i in range(lookback, len(df)):
            current_high = highs[i]
            previous_high = highs[i-lookback:i].max()
            if current_high < previous_high:
                lower_highs[i] = True
        
        return lower_highs
    
    def find_lower_lows(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> np.ndarray:
        """Identify lower lows (downtrend confirmation)."""
        if df.empty or 'low' not in df.columns or len(df) < lookback:
            return np.zeros(len(df), dtype=bool)
        
        lows = df['low'].values
        lower_lows = np.zeros(len(df), dtype=bool)
        
        for i in range(lookback, len(df)):
            current_low = lows[i]
            previous_low = lows[i-lookback:i].min()
            if current_low < previous_low:
                lower_lows[i] = True
        
        return lower_lows
    
    def find_order_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 10
    ) -> Dict[str, List[Dict]]:
        """Find bullish and bearish order blocks."""
        if df.empty or len(df) < lookback:
            return {'bullish': [], 'bearish': []}
        
        order_blocks = {'bullish': [], 'bearish': []}
        
        # Find swing points
        swing_lows = self.find_swing_lows(df, period=5)
        swing_highs = self.find_swing_highs(df, period=5)
        
        # Bullish order blocks (consolidation above swing low)
        for i in range(len(df) - lookback):
            if swing_lows[i] and i + lookback < len(df):
                consolidation = df.iloc[i:i+lookback]
                if consolidation['close'].mean() > consolidation['low'].min():
                    order_blocks['bullish'].append({
                        'start_idx': i,
                        'end_idx': i + lookback,
                        'high': consolidation['high'].max(),
                        'low': consolidation['low'].min(),
                        'type': 'bullish'
                    })
        
        # Bearish order blocks (consolidation below swing high)
        for i in range(len(df) - lookback):
            if swing_highs[i] and i + lookback < len(df):
                consolidation = df.iloc[i:i+lookback]
                if consolidation['close'].mean() < consolidation['high'].max():
                    order_blocks['bearish'].append({
                        'start_idx': i,
                        'end_idx': i + lookback,
                        'high': consolidation['high'].max(),
                        'low': consolidation['low'].min(),
                        'type': 'bearish'
                    })
        
        return order_blocks
    
    def find_fair_value_gaps(
        self,
        df: pd.DataFrame,
        lookback: int = 3
    ) -> List[Dict]:
        """Find fair value gaps (FVG) - price gaps that may be filled."""
        if df.empty or len(df) < lookback + 1:
            return []
        
        fvgs = []
        
        for i in range(lookback, len(df) - 1):
            # Check for bullish FVG (gap up)
            if df['low'].iloc[i] > df['high'].iloc[i-1]:
                fvg = {
                    'idx': i,
                    'type': 'bullish',
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-1],
                    'size': df['low'].iloc[i] - df['high'].iloc[i-1],
                }
                fvgs.append(fvg)
            
            # Check for bearish FVG (gap down)
            elif df['high'].iloc[i] < df['low'].iloc[i-1]:
                fvg = {
                    'idx': i,
                    'type': 'bearish',
                    'top': df['low'].iloc[i-1],
                    'bottom': df['high'].iloc[i],
                    'size': df['low'].iloc[i-1] - df['high'].iloc[i],
                }
                fvgs.append(fvg)
        
        return fvgs
    
    def find_breaker_blocks(
        self,
        df: pd.DataFrame,
        lookback: int = 10
    ) -> Dict[str, List[Dict]]:
        """Find breaker blocks (market structure break with retest)."""
        if df.empty or len(df) < lookback * 2:
            return {'bullish': [], 'bearish': []}
        
        breaker_blocks = {'bullish': [], 'bearish': []}
        
        swing_lows = self.find_swing_lows(df, period=5)
        swing_highs = self.find_swing_highs(df, period=5)
        
        # Bullish breaker block (break of lower low, retest as support)
        for i in range(lookback, len(df)):
            if swing_lows[i] and i + lookback < len(df):
                future_data = df.iloc[i:i+lookback]
                if future_data['low'].min() < df['low'].iloc[i]:
                    breaker_blocks['bullish'].append({
                        'idx': i,
                        'block_low': df['low'].iloc[i],
                        'break_low': future_data['low'].min(),
                    })
        
        # Bearish breaker block (break of higher high, retest as resistance)
        for i in range(lookback, len(df)):
            if swing_highs[i] and i + lookback < len(df):
                future_data = df.iloc[i:i+lookback]
                if future_data['high'].max() > df['high'].iloc[i]:
                    breaker_blocks['bearish'].append({
                        'idx': i,
                        'block_high': df['high'].iloc[i],
                        'break_high': future_data['high'].max(),
                    })
        
        return breaker_blocks
    
    def find_premium_discount_zones(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> Dict[str, Tuple[float, float]]:
        """Identify premium and discount zones based on recent price levels."""
        if df.empty or len(df) < lookback or 'close' not in df.columns:
            return {'premium': (0, 0), 'discount': (0, 0)}
        
        recent_data = df.tail(lookback)
        avg_price = recent_data['close'].mean()
        price_std = recent_data['close'].std()
        
        premium_zone = (avg_price + price_std, avg_price + price_std * 2)
        discount_zone = (avg_price - price_std * 2, avg_price - price_std)
        
        return {
            'premium': premium_zone,
            'discount': discount_zone,
            'avg_price': avg_price
        }
    
    def add_market_structure_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add all market structure features to DataFrame."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Swing points
        df['swing_high'] = self.find_swing_highs(df).astype(int)
        df['swing_low'] = self.find_swing_lows(df).astype(int)
        
        # Trend confirmation
        df['higher_high'] = self.find_higher_highs(df).astype(int)
        df['higher_low'] = self.find_higher_lows(df).astype(int)
        df['lower_high'] = self.find_lower_highs(df).astype(int)
        df['lower_low'] = self.find_lower_lows(df).astype(int)
        
        # Fair value gaps
        fvgs = self.find_fair_value_gaps(df)
        df['fair_value_gap'] = 0
        for fvg in fvgs:
            df.loc[fvg['idx'], 'fair_value_gap'] = 1 if fvg['type'] == 'bullish' else -1
        
        # Premium/Discount zones
        zones = self.find_premium_discount_zones(df)
        if 'close' in df.columns and zones.get('avg_price'):
            avg = zones['avg_price']
            df['premium_zone'] = (df['close'] > zones['premium'][0]).astype(int)
            df['discount_zone'] = (df['close'] < zones['discount'][1]).astype(int)
        
        return df


# ============================================================================
# LIQUIDITY & SWEEP DETECTION
# ============================================================================

class LiquidityAnalyzer:
    """Detects liquidity sweeps and levels."""
    
    def find_liquidity_sweeps(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        threshold_percent: float = 0.1
    ) -> List[Dict]:
        """Find liquidity sweeps (breaches of recent support/resistance)."""
        if df.empty or 'high' not in df.columns or 'low' not in df.columns:
            return []
        
        sweeps = []
        
        for i in range(lookback, len(df)):
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()
            
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Sweep above resistance
            if current_high > recent_high * (1 + threshold_percent/100):
                sweeps.append({
                    'idx': i,
                    'type': 'above_resistance',
                    'level': recent_high,
                    'breach': current_high
                })
            
            # Sweep below support
            if current_low < recent_low * (1 - threshold_percent/100):
                sweeps.append({
                    'idx': i,
                    'type': 'below_support',
                    'level': recent_low,
                    'breach': current_low
                })
        
        return sweeps
    
    def find_equal_levels(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        tolerance_pips: float = 5.0
    ) -> Dict[str, List[float]]:
        """Find equal highs and equal lows."""
        if df.empty or 'high' not in df.columns or 'low' not in df.columns:
            return {'equal_highs': [], 'equal_lows': []}
        
        recent_data = df.tail(lookback)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        tolerance = tolerance_pips / 10000
        
        equal_highs = []
        equal_lows = []
        
        for i, high in enumerate(highs):
            for j in range(i + 1, len(highs)):
                if abs(highs[j] - high) < tolerance:
                    if high not in equal_highs:
                        equal_highs.append(high)
        
        for i, low in enumerate(lows):
            for j in range(i + 1, len(lows)):
                if abs(lows[j] - low) < tolerance:
                    if low not in equal_lows:
                        equal_lows.append(low)
        
        return {
            'equal_highs': equal_highs,
            'equal_lows': equal_lows
        }


if __name__ == "__main__":
    # Test market structure analyzer
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    prices = np.random.randn(100).cumsum() + 100
    
    test_df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.uniform(-0.1, 0.1, 100),
        'high': prices + np.random.uniform(0, 0.5, 100),
        'low': prices - np.random.uniform(0, 0.5, 100),
        'close': prices,
        'tick_volume': np.random.randint(100, 1000, 100),
    })
    
    analyzer = MarketStructureAnalyzer()
    
    swing_highs = analyzer.find_swing_highs(test_df)
    swing_lows = analyzer.find_swing_lows(test_df)
    
    print(f"\nMarket Structure Analysis:")
    print(f"  Swing Highs: {swing_highs.sum()}")
    print(f"  Swing Lows: {swing_lows.sum()}")
    
    fvgs = analyzer.find_fair_value_gaps(test_df)
    print(f"  Fair Value Gaps: {len(fvgs)}")
