# Utility functions for the trading bot

import pandas as pd
import numpy as np

def example_util_function():
    print("This is a utility function.")

def load_historical_data(file_path, date_col=None, required_cols=None, 
                         ohlcv_cols=None, dropna=True):
    """
    Loads historical market data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        date_col (str, optional): Name of the date/timestamp column. 
                                  If provided, it will be parsed and set as index.
        required_cols (list, optional): A list of column names that must be present.
                                        Defaults to ['Open', 'High', 'Low', 'Close', 'Volume'].
        ohlcv_cols (dict, optional): A dictionary to map custom column names to standard 
                                     OHLCV names. E.g., {'o': 'Open', 'c': 'Close'}.
        dropna (bool): If True, rows with any NaN values will be dropped. Otherwise, ffill.

    Returns:
        pandas.DataFrame: DataFrame with historical data, or None if loading fails.
    """
    if required_cols is None:
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Rename columns if a mapping is provided
    if ohlcv_cols:
        df.rename(columns=ohlcv_cols, inplace=True)

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None

    # Convert date column and set as index
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        except Exception as e:
            print(f"Error processing date column '{date_col}': {e}")
            # Continue without setting index if date processing fails, or return None based on strictness
    elif date_col:
        print(f"Warning: Specified date column '{date_col}' not found in CSV.")

    # Ensure numeric types for OHLCV columns (and others if specified)
    for col in required_cols:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {e}")
    
    # Handle missing values
    if df.isnull().any().any(): # Check if there are any NaNs at all
        if dropna:
            df.dropna(inplace=True)
        else:
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True) # Backfill any remaining NaNs at the beginning
    
    # Sort by index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)

    print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df


def add_sma_feature(df, window=20, price_col='Close'):
    """
    Adds a Simple Moving Average (SMA) feature to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with market data.
        window (int): The window size for the SMA.
        price_col (str): The name of the column to calculate SMA on (e.g., 'Close').

    Returns:
        pd.DataFrame: DataFrame with the new SMA column added.
                      Returns the original DataFrame if price_col is not found.
    """
    if price_col not in df.columns:
        print(f"Error: Price column '{price_col}' not found for SMA calculation.")
        return df
    
    sma_col_name = f'SMA_{window}'
    df[sma_col_name] = df[price_col].rolling(window=window, min_periods=1).mean()
    print(f"Added '{sma_col_name}' feature.")
    return df


def add_rsi_feature(df, window=14, price_col='Close'):
    """
    Adds a Relative Strength Index (RSI) feature to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with market data.
        window (int): The window size for the RSI (typically 14).
        price_col (str): The name of the column to calculate RSI on (e.g., 'Close').

    Returns:
        pd.DataFrame: DataFrame with the new RSI column added.
                      Returns the original DataFrame if price_col is not found.
    """
    if price_col not in df.columns:
        print(f"Error: Price column '{price_col}' not found for RSI calculation.")
        return df

    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi_col_name = f'RSI_{window}'
    df[rsi_col_name] = 100 - (100 / (1 + rs))
    
    # Handle cases where loss is zero (prevents inf RSI)
    df[rsi_col_name].replace([np.inf, -np.inf], 100, inplace=True) # If gain > 0 and loss = 0, RSI is 100
    df[rsi_col_name].fillna(50, inplace=True) # Fill initial NaNs (e.g., first row if diff() makes it NaN)
                                            # or if gain and loss are both 0, RSI is undefined (NaN from rs=0/0), use 50.

    # print(f"Added '{rsi_col_name}' feature.") # Less verbose
    return df


if __name__ == '__main__':
    # Create a dummy CSV for testing
    dummy_csv_path = "dummy_market_data_utils.csv"
    num_rows = 100
    data = pd.DataFrame({
        'Timestamp': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(num_rows), 'D'),
        'Open': np.random.rand(num_rows) * 100 + 100,
        'High': np.random.rand(num_rows) * 10 + 105, # Ensure High > Open/Close
        'Low': np.random.rand(num_rows) * -10 + 95, # Ensure Low < Open/Close
        'Close': np.random.rand(num_rows) * 100 + 100,
        'Volume': np.random.randint(1000, 5000, size=num_rows),
        'OtherFeature': np.random.rand(num_rows)
    })
    # Add some NaNs for testing missing value handling
    for col in ['Open', 'Close', 'Volume']:
        idx_to_nan = np.random.choice(data.index, size=num_rows // 20, replace=False)
        data.loc[idx_to_nan, col] = np.nan
    
    data['High'] = data[['Open', 'Close']].max(axis=1).fillna(110) + np.random.rand(num_rows) * 2
    data['Low'] = data[['Open', 'Close']].min(axis=1).fillna(90) - np.random.rand(num_rows) * 2

    data.to_csv(dummy_csv_path, index=False)
    print(f"Created dummy CSV: {dummy_csv_path}")

    # Test loading the dummy data
    print("\n--- Test 1: Basic Load with Date Column ---")
    df_loaded = load_historical_data(dummy_csv_path, date_col='Timestamp')
    if df_loaded is not None:
        print("Original df head:")
        print(df_loaded.head())
        
        print("\n--- Test: Adding SMA_20 Feature ---")
        df_with_sma = add_sma_feature(df_loaded.copy(), window=20, price_col='Close')
        print(df_with_sma.head(25)) # Print more rows to see SMA values populate
        print(df_with_sma.info())
        # Note: SMA will have NaNs for initial period < window size if min_periods is not 1.
        # Our add_sma_feature uses min_periods=1, so it populates from the start.

    print("\n--- Test 2: Custom OHLCV Names & No Drop NA ---")
    # Create another dummy with different column names
    dummy_custom_names_path = "dummy_custom_names.csv"
    data_custom = data.copy()
    data_custom.rename(columns={'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C', 'Volume':'V'}, inplace=True)
    data_custom.to_csv(dummy_custom_names_path, index=False)
    df_custom_loaded = load_historical_data(
        dummy_custom_names_path, 
        date_col='Timestamp',
        ohlcv_cols={'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close', 'V':'Volume'},
        required_cols=['Open', 'High', 'Low', 'Close', 'Volume', 'OtherFeature'],
        dropna=False
    )
    if df_custom_loaded is not None:
        print(df_custom_loaded.head())
        print(df_custom_loaded.info())
        print(f"NaNs after ffill/bfill: {df_custom_loaded.isnull().sum().sum()}")

    print("\n--- Test 3: File Not Found ---")
    load_historical_data("non_existent_file.csv")

    print("\n--- Test 4: Missing Required Column ---")
    dummy_missing_col_path = "dummy_missing_col.csv"
    data_missing = data.copy().drop(columns=['Volume'])
    data_missing.to_csv(dummy_missing_col_path, index=False)
    load_historical_data(dummy_missing_col_path, date_col='Timestamp')

    # Clean up dummy files
    # import os
    # os.remove(dummy_csv_path)
    # os.remove(dummy_custom_names_path)
    # os.remove(dummy_missing_col_path)
    # print("\nCleaned up dummy CSV files.")

    print("\n--- Test 1: Load Data & Add Features ---")
    df_loaded = load_historical_data(dummy_csv_path, date_col='Timestamp', dropna=False) # Use dropna=False for better feature calculation demo
    if df_loaded is not None:
        print("Original df head (after load_historical_data ffill/bfill):")
        print(df_loaded.head())
        print(df_loaded.info())
        
        df_featured = df_loaded.copy()
        df_featured = add_sma_feature(df_featured, window=20)
        df_featured = add_rsi_feature(df_featured, window=14)
        
        print("\n--- DataFrame with SMA and RSI Features (first 25 rows) ---")
        print(df_featured.head(25))
        print(df_featured.info())
        print(f"NaNs remaining after feature engineering: {df_featured.isnull().sum().sum()}")
        
        # Final check: drop any NaNs that might have been introduced by feature calculations if not handled perfectly by min_periods or ffill in functions
        df_featured.dropna(inplace=True) 
        print(f"NaNs after final dropna: {df_featured.isnull().sum().sum()}")
        print("Shape after final dropna:", df_featured.shape) 