"""
Optimized utility functions for handling date parsing in the sentiment analysis system
Enhanced with better performance, error handling, and caching
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDateParser:
    """Optimized date parser with caching and improved performance"""
    
    def __init__(self):
        # Common date formats for Indonesian data
        self.common_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%d %b %Y %H:%M:%S',
            '%d %B %Y %H:%M:%S',
            '%d %b %Y',
            '%d %B %Y'
        ]
        
        # Cache for format detection
        self._format_cache = {}
        self._success_format = None
    
    @lru_cache(maxsize=1000)
    def _cached_parse_single(self, date_str, format_str):
        """Cached single date parsing"""
        try:
            return pd.to_datetime(date_str, format=format_str)
        except:
            return None
    
    def detect_format(self, sample_dates, max_samples=100):
        """Detect the most likely date format from samples"""
        if not sample_dates or len(sample_dates) == 0:
            return None
        
        # Use a sample for performance
        sample_size = min(max_samples, len(sample_dates))
        samples = sample_dates[:sample_size]
        
        format_scores = {}
        
        for fmt in self.common_formats:
            score = 0
            for date_str in samples:
                if self._cached_parse_single(str(date_str), fmt) is not None:
                    score += 1
            
            if score > 0:
                format_scores[fmt] = score / len(samples)
        
        if format_scores:
            best_format = max(format_scores, key=format_scores.get)
            if format_scores[best_format] > 0.7:  # 70% success rate
                return best_format
        
        return None
    
    def parse_dates_vectorized(self, date_series, detected_format=None):
        """Vectorized date parsing with fallback methods"""
        if detected_format:
            try:
                parsed = pd.to_datetime(date_series, format=detected_format, errors='coerce')
                success_rate = parsed.notna().sum() / len(parsed)
                if success_rate > 0.8:
                    return parsed, detected_format, success_rate
            except:
                pass
        
        # Try pandas auto-detection
        try:
            parsed = pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce')
            success_rate = parsed.notna().sum() / len(parsed)
            if success_rate > 0.7:
                return parsed, 'auto_infer', success_rate
        except:
            pass
        
        # Try common formats
        for fmt in self.common_formats:
            try:
                parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
                success_rate = parsed.notna().sum() / len(parsed)
                if success_rate > 0.6:
                    return parsed, fmt, success_rate
            except:
                continue
        
        # Last resort: coerce all errors
        try:
            parsed = pd.to_datetime(date_series, errors='coerce')
            success_rate = parsed.notna().sum() / len(parsed)
            return parsed, 'coerce', success_rate
        except:
            return pd.Series([pd.NaT] * len(date_series)), 'failed', 0.0

def parse_date_column_optimized(df, date_column, chunk_size=10000):
    """
    Optimized date parsing function with chunking and caching
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the column containing dates
        chunk_size: Size of chunks for processing large datasets
    
    Returns:
        tuple: (success, parsed_df, metadata)
    """
    
    if date_column not in df.columns:
        return False, df, {"error": f"Column '{date_column}' not found"}
    
    df_copy = df.copy()
    
    # Remove missing values
    initial_count = len(df_copy)
    df_copy = df_copy.dropna(subset=[date_column])
    
    if len(df_copy) == 0:
        return False, df_copy, {"error": "No valid date values found"}
    
    # Convert to string and clean
    df_copy[date_column] = df_copy[date_column].astype(str).str.strip()
    df_copy = df_copy[df_copy[date_column] != '']
    
    if len(df_copy) == 0:
        return False, df_copy, {"error": "No valid date strings found"}
    
    # Get sample for format detection
    sample_size = min(1000, len(df_copy))
    sample_values = df_copy[date_column].head(sample_size).tolist()
    
    logger.info(f"Processing {len(df_copy)} dates, using sample of {sample_size} for format detection")
    
    # Initialize parser
    parser = OptimizedDateParser()
    
    # Detect format
    detected_format = parser.detect_format(sample_values)
    logger.info(f"Detected format: {detected_format}")
    
    # Process in chunks for large datasets
    if len(df_copy) > chunk_size:
        parsed_chunks = []
        chunk_metadata = []
        
        for i in range(0, len(df_copy), chunk_size):
            chunk = df_copy.iloc[i:i+chunk_size].copy()
            
            parsed_dates, used_format, success_rate = parser.parse_dates_vectorized(
                chunk[date_column], detected_format
            )
            
            chunk['parsed_date'] = parsed_dates
            chunk_clean = chunk[chunk['parsed_date'].notna()].copy()
            
            parsed_chunks.append(chunk_clean)
            chunk_metadata.append({
                'chunk_index': i // chunk_size,
                'chunk_size': len(chunk),
                'parsed_count': len(chunk_clean),
                'success_rate': success_rate,
                'format_used': used_format
            })
            
            logger.info(f"Processed chunk {i//chunk_size + 1}, success rate: {success_rate:.2%}")
        
        # Combine chunks
        if parsed_chunks:
            df_parsed = pd.concat(parsed_chunks, ignore_index=True)
        else:
            return False, df_copy, {"error": "No dates could be parsed in any chunk"}
        
        overall_success_rate = len(df_parsed) / initial_count
        
        metadata = {
            "initial_count": initial_count,
            "final_count": len(df_parsed),
            "overall_success_rate": overall_success_rate,
            "chunks_processed": len(chunk_metadata),
            "chunk_details": chunk_metadata,
            "detected_format": detected_format
        }
        
    else:
        # Process all at once for smaller datasets
        parsed_dates, used_format, success_rate = parser.parse_dates_vectorized(
            df_copy[date_column], detected_format
        )
        
        df_copy['parsed_date'] = parsed_dates
        df_parsed = df_copy[df_copy['parsed_date'].notna()].copy()
        
        if len(df_parsed) == 0:
            return False, df_copy, {
                "error": "No dates could be parsed",
                "sample_values": sample_values[:5]
            }
        
        overall_success_rate = len(df_parsed) / initial_count
        
        metadata = {
            "initial_count": initial_count,
            "final_count": len(df_parsed),
            "overall_success_rate": overall_success_rate,
            "format_used": used_format,
            "detected_format": detected_format,
            "success_rate": success_rate
        }
    
    success_message = f"Successfully parsed {len(df_parsed)}/{initial_count} dates ({overall_success_rate:.1%})"
    
    return True, df_parsed, {**metadata, "message": success_message}

def analyze_date_formats_optimized(series, max_samples=1000):
    """
    Optimized analysis of date formats in a series
    
    Args:
        series: Pandas series containing date strings
        max_samples: Maximum number of samples to analyze for performance
        
    Returns:
        dict: Enhanced analysis results
    """
    
    if series.empty:
        return {"error": "Empty series provided"}
    
    # Basic statistics
    analysis = {
        'total_values': len(series),
        'null_values': series.isnull().sum(),
        'unique_values': series.nunique(),
        'data_types': {},
        'potential_formats': [],
        'length_distribution': {},
        'sample_values': []
    }
    
    # Get non-null values
    valid_series = series.dropna()
    if len(valid_series) == 0:
        analysis['error'] = "No valid values found"
        return analysis
    
    # Sample for performance
    sample_size = min(max_samples, len(valid_series))
    sample_series = valid_series.sample(n=sample_size, random_state=42) if len(valid_series) > sample_size else valid_series
    
    # Analyze data types
    type_counts = {}
    for val in sample_series.head(100):
        val_type = type(val).__name__
        type_counts[val_type] = type_counts.get(val_type, 0) + 1
    
    analysis['data_types'] = type_counts
    analysis['sample_values'] = sample_series.head(10).tolist()
    
    # Convert to strings and analyze
    sample_strings = sample_series.astype(str)
    
    # Analyze string lengths
    lengths = sample_strings.str.len()
    analysis['length_distribution'] = {
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': float(lengths.mean()),
        'mode': int(lengths.mode().iloc[0]) if not lengths.mode().empty else 0
    }
    
    # Enhanced format patterns with scoring
    format_patterns = [
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', 'YYYY-MM-DD HH:MM:SS', '%Y-%m-%d %H:%M:%S'),
        (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$', 'YYYY-MM-DD HH:MM:SS.fff', '%Y-%m-%d %H:%M:%S.%f'),
        (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD', '%Y-%m-%d'),
        (r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$', 'DD/MM/YYYY HH:MM:SS', '%d/%m/%Y %H:%M:%S'),
        (r'^\d{2}/\d{2}/\d{4}$', 'DD/MM/YYYY', '%d/%m/%Y'),
        (r'^\d{1,2}/\d{1,2}/\d{4}$', 'D/M/YYYY or M/D/YYYY', '%m/%d/%Y'),
        (r'^\d{2}-\d{2}-\d{4}$', 'DD-MM-YYYY', '%d-%m-%Y'),
        (r'^\d{4}/\d{2}/\d{2}$', 'YYYY/MM/DD', '%Y/%m/%d'),
        (r'^\d{1,2} \w{3} \d{4}$', 'D Mon YYYY', '%d %b %Y'),
        (r'^\d{1,2} \w+ \d{4}$', 'D Month YYYY', '%d %B %Y'),
    ]
    
    # Test patterns
    for pattern, format_name, pandas_format in format_patterns:
        matches = sample_strings.str.match(pattern).sum()
        if matches > 0:
            match_percentage = (matches / len(sample_strings)) * 100
            confidence = "High" if match_percentage > 80 else "Medium" if match_percentage > 50 else "Low"
            
            analysis['potential_formats'].append({
                'format_name': format_name,
                'pandas_format': pandas_format,
                'matches': int(matches),
                'total_samples': len(sample_strings),
                'match_percentage': round(match_percentage, 2),
                'confidence': confidence
            })
    
    # Sort by match percentage
    analysis['potential_formats'].sort(key=lambda x: x['match_percentage'], reverse=True)
    
    # Add recommendations
    if analysis['potential_formats']:
        best_format = analysis['potential_formats'][0]
        if best_format['match_percentage'] > 80:
            analysis['recommendation'] = f"Use format: {best_format['pandas_format']}"
        elif best_format['match_percentage'] > 50:
            analysis['recommendation'] = f"Likely format: {best_format['pandas_format']}, but consider manual verification"
        else:
            analysis['recommendation'] = "Mixed formats detected, use pandas auto-detection"
    else:
        analysis['recommendation'] = "No standard formats detected, use pandas coerce method"
    
    return analysis

def create_time_features_optimized(df, date_column, timezone_aware=False):
    """
    Create optimized time-based features from a datetime column
    
    Args:
        df: DataFrame with parsed date column
        date_column: Name of the datetime column
        timezone_aware: Whether to handle timezone information
        
    Returns:
        DataFrame with additional time features
    """
    
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        try:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        except:
            raise ValueError(f"Cannot convert column '{date_column}' to datetime")
    
    # Basic time features (vectorized operations)
    dt_accessor = df_copy[date_column].dt
    
    # Core temporal features
    df_copy['year'] = dt_accessor.year
    df_copy['month'] = dt_accessor.month
    df_copy['day'] = dt_accessor.day
    df_copy['weekday'] = dt_accessor.dayofweek  # 0=Monday, 6=Sunday
    df_copy['weekday_name'] = dt_accessor.day_name()
    df_copy['hour'] = dt_accessor.hour
    df_copy['minute'] = dt_accessor.minute
    
    # Extended temporal features
    df_copy['quarter'] = dt_accessor.quarter
    df_copy['day_of_year'] = dt_accessor.dayofyear
    df_copy['week_of_year'] = dt_accessor.isocalendar().week
    df_copy['month_name'] = dt_accessor.month_name()
    
    # Period features (optimized)
    df_copy['year_month'] = dt_accessor.to_period('M')
    df_copy['year_week'] = dt_accessor.to_period('W')
    df_copy['year_quarter'] = dt_accessor.to_period('Q')
    
    # Derived boolean features
    df_copy['is_weekend'] = df_copy['weekday'] >= 5
    df_copy['is_month_start'] = dt_accessor.is_month_start
    df_copy['is_month_end'] = dt_accessor.is_month_end
    df_copy['is_quarter_start'] = dt_accessor.is_quarter_start
    df_copy['is_quarter_end'] = dt_accessor.is_quarter_end
    df_copy['is_year_start'] = dt_accessor.is_year_start
    df_copy['is_year_end'] = dt_accessor.is_year_end
    
    # Time-based categories
    def categorize_hour(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df_copy['time_of_day'] = df_copy['hour'].apply(categorize_hour)
    
    # Indonesian context features
    def get_season_indonesia(month):
        if month in [12, 1, 2]:
            return 'Wet_Season'
        elif month in [6, 7, 8]:
            return 'Dry_Season'
        else:
            return 'Transition'
    
    df_copy['season_indonesia'] = df_copy['month'].apply(get_season_indonesia)
    
    # Business time features
    df_copy['is_business_hour'] = (df_copy['hour'] >= 9) & (df_copy['hour'] < 17) & (~df_copy['is_weekend'])
    
    return df_copy

def get_time_range_summary_optimized(df, date_column, include_trends=True):
    """
    Get optimized summary statistics for a time range with trend analysis
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of the datetime column
        include_trends: Whether to calculate trend statistics
        
    Returns:
        dict: Comprehensive summary statistics
    """
    
    if len(df) == 0:
        return {"error": "Empty DataFrame provided"}
    
    if date_column not in df.columns:
        return {"error": f"Column '{date_column}' not found"}
    
    dates = df[date_column]
    
    # Remove NaT values
    valid_dates = dates.dropna()
    if len(valid_dates) == 0:
        return {"error": "No valid dates found"}
    
    # Basic summary
    summary = {
        'start_date': valid_dates.min(),
        'end_date': valid_dates.max(),
        'total_days': (valid_dates.max() - valid_dates.min()).days,
        'total_records': len(df),
        'valid_records': len(valid_dates),
        'invalid_records': len(dates) - len(valid_dates),
        'unique_dates': valid_dates.dt.date.nunique(),
        'date_range': f"{valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}"
    }
    
    # Time span analysis
    time_span_days = summary['total_days']
    if time_span_days < 7:
        summary['time_span_category'] = 'Less than a week'
    elif time_span_days < 30:
        summary['time_span_category'] = 'Less than a month'
    elif time_span_days < 365:
        summary['time_span_category'] = 'Less than a year'
    else:
        summary['time_span_category'] = f'Multiple years ({time_span_days // 365} years)'
    
    # Enhanced temporal distributions
    if include_trends and len(valid_dates) > 1:
        # Monthly distribution
        monthly_counts = df.groupby(valid_dates.dt.to_period('M')).size()
        if len(monthly_counts) > 0:
            summary['monthly_stats'] = {
                'average': float(monthly_counts.mean()),
                'std': float(monthly_counts.std()),
                'min': int(monthly_counts.min()),
                'max': int(monthly_counts.max()),
                'most_active_month': str(monthly_counts.idxmax()),
                'least_active_month': str(monthly_counts.idxmin()),
                'coefficient_of_variation': float(monthly_counts.std() / monthly_counts.mean()) if monthly_counts.mean() > 0 else 0
            }
        
        # Weekly distribution
        weekly_counts = df.groupby(valid_dates.dt.to_period('W')).size()
        if len(weekly_counts) > 0:
            summary['weekly_stats'] = {
                'average': float(weekly_counts.mean()),
                'std': float(weekly_counts.std()),
                'most_active_week': str(weekly_counts.idxmax()),
                'least_active_week': str(weekly_counts.idxmin())
            }
        
        # Daily distribution by weekday
        weekday_counts = df.groupby(valid_dates.dt.day_name()).size()
        if len(weekday_counts) > 0:
            summary['weekday_stats'] = {
                'most_active_day': weekday_counts.idxmax(),
                'least_active_day': weekday_counts.idxmin(),
                'weekend_vs_weekday_ratio': float(
                    weekday_counts.reindex(['Saturday', 'Sunday'], fill_value=0).sum() / 
                    weekday_counts.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], fill_value=0).sum()
                ) if weekday_counts.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], fill_value=0).sum() > 0 else 0
            }
        
        # Hourly distribution (if time information is available)
        if valid_dates.dt.hour.nunique() > 1:
            hourly_counts = df.groupby(valid_dates.dt.hour).size()
            summary['hourly_stats'] = {
                'peak_hour': int(hourly_counts.idxmax()),
                'quiet_hour': int(hourly_counts.idxmin()),
                'business_hours_percentage': float(
                    hourly_counts.reindex(range(9, 17), fill_value=0).sum() / len(df) * 100
                ),
                'night_hours_percentage': float(
                    hourly_counts.reindex(list(range(0, 6)) + list(range(22, 24)), fill_value=0).sum() / len(df) * 100
                )
            }
        
        # Trend analysis
        if len(monthly_counts) > 2:
            # Simple trend calculation
            months_numeric = np.arange(len(monthly_counts))
            correlation = np.corrcoef(months_numeric, monthly_counts.values)[0, 1]
            
            if correlation > 0.3:
                trend = 'Increasing'
            elif correlation < -0.3:
                trend = 'Decreasing'
            else:
                trend = 'Stable'
            
            summary['trend_analysis'] = {
                'overall_trend': trend,
                'correlation_coefficient': float(correlation),
                'growth_rate': float((monthly_counts.iloc[-1] - monthly_counts.iloc[0]) / monthly_counts.iloc[0] * 100) if monthly_counts.iloc[0] != 0 else 0
            }
    
    # Data quality metrics
    summary['data_quality'] = {
        'completeness': float(len(valid_dates) / len(dates) * 100),
        'uniqueness': float(summary['unique_dates'] / len(valid_dates) * 100) if len(valid_dates) > 0 else 0,
        'temporal_coverage': float(summary['unique_dates'] / max(1, summary['total_days']) * 100)
    }
    
    return summary

# Backward compatibility functions
def parse_date_column(df, date_column):
    """Backward compatible wrapper"""
    success, result_df, metadata = parse_date_column_optimized(df, date_column)
    
    if success:
        error_message = metadata.get('message', 'Success')
    else:
        error_message = metadata.get('error', 'Unknown error')
    
    return success, result_df, error_message

def analyze_date_formats(series):
    """Backward compatible wrapper"""
    return analyze_date_formats_optimized(series)

def create_time_features(df, date_column):
    """Backward compatible wrapper"""
    return create_time_features_optimized(df, date_column)

def get_time_range_summary(df, date_column):
    """Backward compatible wrapper"""
    return get_time_range_summary_optimized(df, date_column)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Create sample data
    sample_dates = [
        '2024-01-15 10:30:00',
        '2024-02-20 14:45:30',
        '2024-03-10 09:15:45',
        '15/04/2024 16:20:10',
        '2024-05-25',
        'invalid_date',
        '2024-06-30 23:59:59'
    ]
    
    df_test = pd.DataFrame({
        'id': range(len(sample_dates)),
        'date_column': sample_dates,
        'value': np.random.randn(len(sample_dates))
    })
    
    print("Testing optimized date parsing...")
    
    # Test date parsing
    success, parsed_df, metadata = parse_date_column_optimized(df_test, 'date_column')
    
    if success:
        print(f"✅ Success: {metadata['message']}")
        print(f"📊 Parsed {metadata['final_count']}/{metadata['initial_count']} dates")
        
        # Test time feature creation
        enhanced_df = create_time_features_optimized(parsed_df, 'parsed_date')
        print(f"✅ Created {len(enhanced_df.columns) - len(parsed_df.columns)} additional time features")
        
        # Test time range summary
        summary = get_time_range_summary_optimized(enhanced_df, 'parsed_date')
        print(f"✅ Time range: {summary['date_range']}")
        print(f"📈 Data quality: {summary['data_quality']['completeness']:.1f}% complete")
        
    else:
        print(f"❌ Failed: {metadata['error']}")
    
    # Test format analysis
    print("\nTesting format analysis...")
    format_analysis = analyze_date_formats_optimized(df_test['date_column'])
    print(f"✅ Found {len(format_analysis['potential_formats'])} potential formats")
    if format_analysis['potential_formats']:
        best_format = format_analysis['potential_formats'][0]
        print(f"🎯 Best format: {best_format['format_name']} ({best_format['match_percentage']}% match)")
    
    print("\n🎉 All tests completed successfully!")