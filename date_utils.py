"""
Utility functions for handling date parsing in the sentiment analysis system
"""

import pandas as pd
import re
from datetime import datetime

def parse_date_column(df, date_column):
    """
    Robust date parsing function that tries multiple methods
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the column containing dates
    
    Returns:
        tuple: (success, parsed_df, error_message)
    """
    
    df_copy = df.copy()
    
    # Remove missing values
    initial_count = len(df_copy)
    df_copy = df_copy.dropna(subset=[date_column])
    
    if len(df_copy) == 0:
        return False, df_copy, "No valid date values found"
    
    # Get sample values for debugging
    sample_values = df_copy[date_column].head(5).tolist()
    
    # Define parsing methods to try
    parsing_methods = [
        # Method 1: Standard pandas parsing
        lambda x: pd.to_datetime(x),
        
        # Method 2: Infer format automatically
        lambda x: pd.to_datetime(x, infer_datetime_format=True),
        
        # Method 3: Common formats
        lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'),
        lambda x: pd.to_datetime(x, format='%Y-%m-%d'),
        lambda x: pd.to_datetime(x, format='%d/%m/%Y'),
        lambda x: pd.to_datetime(x, format='%d-%m-%Y'),
        lambda x: pd.to_datetime(x, format='%m/%d/%Y'),
        lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'),
        
        # Method 4: Coerce errors (last resort)
        lambda x: pd.to_datetime(x, errors='coerce')
    ]
    
    # Try each parsing method
    for i, method in enumerate(parsing_methods):
        try:
            df_copy['parsed_date'] = method(df_copy[date_column])
            
            # Remove NaT (Not a Time) values
            valid_dates = df_copy['parsed_date'].notna()
            df_parsed = df_copy[valid_dates].copy()
            
            if len(df_parsed) > 0:
                success_rate = len(df_parsed) / initial_count * 100
                return True, df_parsed, f"Successfully parsed {len(df_parsed)}/{initial_count} dates ({success_rate:.1f}%) using method {i+1}"
                
        except Exception as e:
            continue
    
    # If all methods failed
    return False, df_copy, f"Failed to parse dates. Sample values: {sample_values}"

def analyze_date_formats(series):
    """
    Analyze the formats present in a date series
    
    Args:
        series: Pandas series containing date strings
        
    Returns:
        dict: Analysis results
    """
    
    analysis = {
        'total_values': len(series),
        'null_values': series.isnull().sum(),
        'unique_values': series.nunique(),
        'sample_values': series.dropna().head(10).tolist(),
        'data_types': {},
        'potential_formats': []
    }
    
    # Analyze data types
    for val in series.dropna().head(50):
        val_type = type(val).__name__
        analysis['data_types'][val_type] = analysis['data_types'].get(val_type, 0) + 1
    
    # Detect potential formats
    sample_strings = [str(val) for val in series.dropna().head(20)]
    
    format_patterns = [
        (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'YYYY-MM-DD HH:MM:SS'),
        (r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD'),
        (r'\d{2}/\d{2}/\d{4}', 'DD/MM/YYYY or MM/DD/YYYY'),
        (r'\d{2}-\d{2}-\d{4}', 'DD-MM-YYYY'),
        (r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}', 'D/M/YYYY H:MM:SS'),
    ]
    
    for pattern, format_name in format_patterns:
        matches = sum(1 for s in sample_strings if re.match(pattern, str(s)))
        if matches > 0:
            analysis['potential_formats'].append({
                'format': format_name,
                'matches': matches,
                'sample_count': len(sample_strings)
            })
    
    return analysis

def create_time_features(df, date_column):
    """
    Create additional time-based features from a datetime column
    
    Args:
        df: DataFrame with parsed date column
        date_column: Name of the datetime column
        
    Returns:
        DataFrame with additional time features
    """
    
    df_copy = df.copy()
    
    # Basic time features
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month
    df_copy['day'] = df_copy[date_column].dt.day
    df_copy['weekday'] = df_copy[date_column].dt.day_name()
    df_copy['hour'] = df_copy[date_column].dt.hour
    
    # Period features
    df_copy['year_month'] = df_copy[date_column].dt.to_period('M')
    df_copy['year_week'] = df_copy[date_column].dt.to_period('W')
    df_copy['quarter'] = df_copy[date_column].dt.quarter
    
    # Derived features
    df_copy['is_weekend'] = df_copy[date_column].dt.weekday >= 5
    df_copy['month_name'] = df_copy[date_column].dt.month_name()
    
    return df_copy

def get_time_range_summary(df, date_column):
    """
    Get summary statistics for a time range
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of the datetime column
        
    Returns:
        dict: Summary statistics
    """
    
    if len(df) == 0:
        return {}
    
    dates = df[date_column]
    
    summary = {
        'start_date': dates.min(),
        'end_date': dates.max(),
        'total_days': (dates.max() - dates.min()).days,
        'total_records': len(df),
        'unique_dates': dates.dt.date.nunique(),
        'date_range': f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    }
    
    # Monthly distribution
    monthly_counts = df.groupby(dates.dt.to_period('M')).size()
    summary['monthly_average'] = monthly_counts.mean()
    summary['monthly_std'] = monthly_counts.std()
    summary['most_active_month'] = monthly_counts.idxmax()
    summary['least_active_month'] = monthly_counts.idxmin()
    
    return summary