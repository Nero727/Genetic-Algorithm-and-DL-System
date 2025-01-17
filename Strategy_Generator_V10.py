import psycopg2
import pandas as pd
import numpy as np
import random
from psycopg2 import sql
from datetime import datetime, timedelta
import logging
from tqdm import tqdm  
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import numpy.ma as ma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)


cryptocurrencies = ['BTCUSDT']
timeframes = ['1d']
use_filtered_data = False


# Database connection parameters
db_connection_params = {
    "dbname": "strategies",
    "user": "admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

insert_query = """
    INSERT INTO binance_strategies_filtered_complete (
        name, sharpe_ratio, final_capital, percentage_profit, optimized_params, crypto_currency, timeframe
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# Initialize counters globally
total_strategies = 0
profitable_strategies_count = 0


def fetch_btc_data_from_db(symbol, interval):

    query = f"""
    SELECT 
        original_timestamp AS datetime,
        open,
        high,
        low,
        close,
        volume
    FROM binance_data
    WHERE symbol = %s AND timeframe = %s
    ORDER BY original_timestamp;
    """
    try:
        with psycopg2.connect(**db_connection_params) as conn:
            data = pd.read_sql_query(query, conn, params=(symbol, interval))
            data['datetime'] = pd.to_datetime(data['datetime'], unit='ms')
            data.set_index('datetime', inplace=True)
            return data
    except Exception as e:
        logger.error(f"Error fetching BTC data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
def fetch_reduced_btc_data_from_db(symbol, interval):

    query = f"""
    SELECT 
        original_timestamp AS datetime,
        open,
        high,
        low,
        close,
        volume
    FROM binance_data_vrm_60_7
    WHERE symbol = %s AND timeframe = %s
    ORDER BY original_timestamp;
    """
    try:
        with psycopg2.connect(**db_connection_params) as conn:
            data = pd.read_sql_query(query, conn, params=(symbol, interval))
            data['datetime'] = pd.to_datetime(data['datetime'], unit='ms')
            data.set_index('datetime', inplace=True)
            return data
    except Exception as e:
        logger.error(f"Error fetching reduced BTC data from database: {e}")
        return pd.DataFrame()
    
def overlay_with_reduced_dates(full_data_with_indicators, reduced_data):

    logger.info("Overlaying reduced dataset dates on full dataset indicators...")

    # Ensure the reduced_data index is a subset of the full_data_with_indicators index
    valid_indices = reduced_data.index.intersection(full_data_with_indicators.index)
    
    if len(valid_indices) == 0:
        logger.error("No valid indices overlap between full and reduced datasets.")
        raise ValueError("Reduced dataset has no overlapping timestamps with the full dataset.")

    # Filter the full dataset to only include valid indices
    filtered_data = full_data_with_indicators.loc[valid_indices]

    logger.info(f"Filtered dataset contains {len(filtered_data)} rows, with NaN values for gaps if any.")
    return filtered_data







    
def validate_params(params):
    """
    Validate and adjust parameters to ensure logical and valid values.
    """
    # Ensure RSI period is within a reasonable range
    params['rsi_period'] = max(2, int(round(params['rsi_period'])))  # Minimum RSI period is 2
    params['rsi_period'] = min(params['rsi_period'], 50)  # Cap maximum RSI period to avoid excessive lookbacks

    # ADX threshold must be non-negative
    params['adx_threshold'] = max(0, params['adx_threshold'])

    # Stop loss and take profit must be positive
    params['stop_loss'] = abs(params['stop_loss'])
    params['take_profit'] = abs(params['take_profit'])

    # Ensure RSI thresholds are logical
    params['rsi_oversold'] = max(0, min(params['rsi_oversold'], 100))
    params['rsi_overbought'] = max(0, min(params['rsi_overbought'], 100))

    # Ensure BBW threshold is non-negative
    params['bbw_threshold'] = max(0, params['bbw_threshold'])

    # Validate SMA and EMA periods
    params['sma_period'] = max(2, int(round(params.get('sma_period', 20))))  # Default SMA period is 20
    params['ema_period'] = max(2, int(round(params.get('ema_period', 9))))   # Default EMA period is 9

    # Ensure volatility period is valid
    params['volatility_period'] = max(2, int(round(params.get('volatility_period', 20))))  # Default volatility period is 20

    # LRC slope threshold: no specific constraints (domain-dependent)
    params['lrc_slope_threshold'] = params['lrc_slope_threshold']

    return params



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)


# Global set to track tested strategies
tested_profitable_strategies = set()


def plot_with_gaps(ax, x, y, label, **kwargs):
    """
    Helper function to plot data with gaps (disconnected lines for NaN values).
    """
    y_masked = ma.masked_invalid(y)  # Mask invalid (NaN) values
    ax.plot(x, y_masked, label=label, **kwargs)


def plot_indicators(data, timeframe, output_dir="plots", label="filtered"):
    """
    Plots the price along with selected indicators such as RSI, LRC Slope, BBW, and ADX.
    Saves plots as images and ensures gaps are visible in the plot.

    Parameters:
    - data: pandas DataFrame containing 'close', 'rsi', 'lrc_slope', 'bbw', 'adx'.
    - timeframe: str, the chosen timeframe (e.g., '5m', '1h', '1d').
    - output_dir: str, the directory where plots will be saved.
    - label: str, a label to distinguish the dataset (e.g., "complete" or "filtered").

    Returns:
    - None. Displays and saves the plots.
    """
    logger.info(f"Preparing to plot indicators for {label} data.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Map timeframe to frequency
    freq_map = {'1d': 'D', '1h': 'H', '5m': '5T'}
    freq = freq_map.get(timeframe, 'D')  # Default to daily if timeframe is unsupported

    # Create a complete time range based on the dataset
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)

    # Reindex the data to include the full time range
    data = data.reindex(full_index)

    # Log NaN details for debugging
    logger.info(f"Reindexed {label} data to include full time range.")
    logger.info(f"{label.capitalize()} data head after reindexing:\n%s", data.head())
    logger.info(f"NaNs in reindexed {label} data:\n%s", data.isna().sum())

    # Filter data based on timeframe-specific requirements
    if timeframe in ['1h', '5m']:
        # Plot only the first month of data
        logger.info(f"Restricting to the first month for {timeframe} timeframe ({label} data).")
        first_month = data.index.min() + pd.DateOffset(months=1)
        data_to_plot = data[data.index <= first_month]
    else:
        data_to_plot = data  # Plot the full range for `1d` or any other timeframe

    # Create the plot with subplots
    fig, axs = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"Price and Indicators Visualization ({label.capitalize()} Data)", fontsize=16)

    # Plot 1: Close Price
    axs[0].plot(data_to_plot.index, data_to_plot['close'], label='Close Price', linewidth=1.5)
    axs[0].set_title("Close Price")
    axs[0].legend()

    # Plot 2: LRC Slope
    if 'lrc_slope' in data_to_plot.columns:
        axs[1].plot(data_to_plot.index, data_to_plot['lrc_slope'], label='LRC Slope', linewidth=1.5, color='blue')
        axs[1].set_title("Linear Regression Channel (LRC) Slope")
        axs[1].legend()

    # Plot 3: RSI
    if 'rsi' in data_to_plot.columns:
        axs[2].plot(data_to_plot.index, data_to_plot['rsi'], label='RSI', color='green', linewidth=1.5)
        axs[2].axhline(30, color='red', linestyle='--', linewidth=0.75, label='Oversold (30)')
        axs[2].axhline(70, color='red', linestyle='--', linewidth=0.75, label='Overbought (70)')
        axs[2].set_title("Relative Strength Index (RSI)")
        axs[2].legend()

    # Plot 4: BBW
    if 'bbw' in data_to_plot.columns:
        axs[3].plot(data_to_plot.index, data_to_plot['bbw'], label='BBW', color='purple', linewidth=1.5)
        axs[3].set_title("Bollinger Bands Width (BBW)")
        axs[3].legend()

    # Plot 5: ADX
    if 'adx' in data_to_plot.columns:
        axs[4].plot(data_to_plot.index, data_to_plot['adx'], label='ADX', color='orange', linewidth=1.5)
        axs[4].set_title("Average Directional Index (ADX)")
        axs[4].legend()

    # Final adjustments
    for ax in axs:
        ax.grid(True)
        ax.set_xlabel("Time")
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, f"indicators_{timeframe}_{label}.png")
    plt.savefig(save_path, dpi=300)
    logger.info(f"Saved plot for {label} data to {save_path}")
    plt.show()



# Example usage
# Assuming `data` is a pandas DataFrame containing your price and indicators
# data = fetch_btc_data_from_db('BTCUSDT', '5m')  # Replace with your actual data fetching function
# plot_indicators(data)

def fitness_function(population, data, crypto_currency, timeframe, one_minute_data, gen):
    """
    Evaluate the fitness of a population of strategies in a batch using CPU-based calculations.
    Save profitable strategies to the database immediately when found.
    """
    logger.info("\n--- Batch Fitness Function Evaluation ---")
    population_size = len(population)
    logger.info(f"Evaluating {population_size} strategies...")

    # Initialize results
    fitness_scores = np.full(population_size, np.inf, dtype=np.float32)  # Default to infinity
    final_capitals = np.zeros(population_size, dtype=np.float32)
    profitable_count = 0  # Count of profitable strategies

    # Set to track evaluated strategies
    evaluated_strategies = set()

    try:
        # Ensure `data` has a `DatetimeIndex`
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("'data' must have a DatetimeIndex.")
            raise ValueError("'data' must have a DatetimeIndex.")

        # Ensure `one_minute_data` has a `DatetimeIndex`
        if not isinstance(one_minute_data.index, pd.DatetimeIndex):
            logger.warning("'one_minute_data' does not have a DatetimeIndex. Converting...")
            if 'datetime' in one_minute_data.columns:
                one_minute_data['datetime'] = pd.to_datetime(one_minute_data['datetime'], errors='coerce')
                if one_minute_data['datetime'].isnull().any():
                    logger.warning("Null values found in 'datetime' after conversion. Removing invalid rows.")
                    one_minute_data = one_minute_data.dropna(subset=['datetime'])
                one_minute_data.set_index('datetime', inplace=True)
            else:
                logger.error("'one_minute_data' must include a 'datetime' column.")
                raise ValueError("'one_minute_data' must include a 'datetime' column.")

        logger.info("Input validation successful. Proceeding with fitness evaluation.")

        # Pre-calculate indicators
        logger.info("Calculating adjusted data for population evaluation...")
        adjusted_data = calculate_indicators(data.copy(), timeframe, gpu=False)

        # Validate adjusted data structure
        if not isinstance(adjusted_data, pd.DataFrame):
            logger.error("Adjusted data is invalid or not a DataFrame. Skipping evaluation.")
            return fitness_scores

        # Validate all required columns are present in adjusted_data
        required_columns = ['close', 'high', 'low', 'rsi', 'adx', 'bbw', 'lrc_slope', 'volatility']
        missing_columns = [col for col in required_columns if col not in adjusted_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns in adjusted data: {missing_columns}. Aborting evaluation.")
            return fitness_scores

        logger.info("Adjusted data validation successful. Proceeding with batch strategy evaluation.")

        # Batch backtest for all strategies
        for i, X in enumerate(population):
            params = validate_params({
                'rsi_period': int(round(X[0], 0)),
                'adx_threshold': round(X[1], 4),
                'stop_loss': round(X[2], 4),
                'take_profit': round(X[3], 4),
                'rsi_oversold': round(X[4], 2),
                'rsi_overbought': round(X[5], 2),
                'bbw_threshold': round(X[6], 4),
                'lrc_slope_threshold': round(X[7], 4),
                'verbose': False
            })

            # Create a signature for the strategy to detect duplicates
            strategy_signature = (
                params['rsi_period'], params['adx_threshold'], params['stop_loss'], params['take_profit'],
                params['rsi_oversold'], params['rsi_overbought'], params['bbw_threshold'], params['lrc_slope_threshold'],
                timeframe, crypto_currency
            )

            # Check for duplicate strategies before testing
            if strategy_signature in evaluated_strategies:
                logger.debug(f"Skipping duplicate strategy {strategy_signature}.")
                continue

            # Add to evaluated strategies
            evaluated_strategies.add(strategy_signature)

            # Backtest the strategy
            try:
                logger.debug(f"Backtesting strategy {i + 1}/{population_size} with parameters: {params}")

                # Pass adjusted data (with DatetimeIndex) and raw prefetch data to backtest_strategy
                final_capital, positions = backtest_strategy(
                    adjusted_data,
                    params,
                    prefetch_data=one_minute_data
                )

                final_capitals[i] = final_capital

                # Calculate metrics
                percentage_profit = (final_capital - 100000) / 100000 * 100
                returns = np.diff(final_capitals) / np.where(final_capitals[:-1] != 0, final_capitals[:-1], np.nan)
                returns = np.nan_to_num(returns, nan=0.0)  # Replace NaNs with 0

                sharpe_ratio = (
                    np.mean(returns) / (np.std(returns) + 1e-6)
                    if len(returns) > 0 and np.std(returns) > 0
                    else 0
                )
                fitness_scores[i] = -sharpe_ratio if sharpe_ratio > 0 else -percentage_profit

                # Count profitable strategies
                if percentage_profit > 0.5:
                    profitable_count += 1

                logger.info(
                    f"Generation {gen + 1}: Strategy {i + 1}/{population_size} evaluated. "
                    f"Profit: {percentage_profit:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}, Fitness: {fitness_scores[i]:.2f}"
                )

                # Save profitable strategies to database
                if percentage_profit > 0.5:
                    logger.info(f"Saving profitable strategy with profit {percentage_profit:.2f}%")
                    save_profitable_strategy(
                        params, percentage_profit, final_capital, sharpe_ratio, crypto_currency, timeframe
                    )

            except Exception as e:
                logger.error(f"Error during backtesting for strategy {i + 1}: {e}")

        logger.info(f"Number of profitable strategies: {profitable_count}/{population_size}")

    except Exception as e:
        logger.error(f"Critical error during batch fitness evaluation: {e}")

    logger.info("Batch evaluation complete.")
    return fitness_scores




def save_profitable_strategy(params, percentage_profit, final_capital, sharpe_ratio, crypto_currency, timeframe):
    """
    Save a profitable strategy to the database, ensuring no duplicates are saved.
    """
    try:
        with psycopg2.connect(**db_connection_params) as conn:
            with conn.cursor() as cur:
                # Check if the strategy already exists in the database
                cur.execute("""
                    SELECT COUNT(*) FROM binance_strategies_filtered_complete
                    WHERE rsi_period = %s AND adx_threshold = %s AND stop_loss = %s AND take_profit = %s
                      AND rsi_oversold = %s AND rsi_overbought = %s AND bbw_threshold = %s AND lrc_slope_threshold = %s
                      AND timeframe = %s AND crypto_currency = %s;
                """, (
                    int(params['rsi_period']),
                    float(params['adx_threshold']),
                    float(params['stop_loss']),
                    float(params['take_profit']),
                    float(params['rsi_oversold']),
                    float(params['rsi_overbought']),
                    float(params['bbw_threshold']),
                    float(params['lrc_slope_threshold']),
                    str(timeframe),
                    str(crypto_currency)
                ))
                count = cur.fetchone()[0]

                if count == 0:
                    # If no duplicate found, insert the new strategy
                    cur.execute("""
                        INSERT INTO binance_strategies_filtered_complete (
                            rsi_period, adx_threshold, stop_loss, take_profit,
                            rsi_oversold, rsi_overbought, bbw_threshold, lrc_slope_threshold,
                            timeframe, crypto_currency, percentage_profit, final_capital, sharpe_ratio, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
                    """, (
                        int(params['rsi_period']),
                        float(params['adx_threshold']),
                        float(params['stop_loss']),
                        float(params['take_profit']),
                        float(params['rsi_oversold']),
                        float(params['rsi_overbought']),
                        float(params['bbw_threshold']),
                        float(params['lrc_slope_threshold']),
                        str(timeframe),
                        str(crypto_currency),
                        float(percentage_profit),
                        float(final_capital),
                        float(sharpe_ratio)
                    ))
                    conn.commit()
                    logger.info("Strategy saved successfully.")
                else:
                    logger.info("Duplicate strategy found. Skipping save operation.")
    except Exception as e:
        logger.error(f"Failed to save strategy to `binance_strategies_filtered_complete`: {e}")





def initialize_database(db_connection_params, table_name="binance_strategies_filtered_complete"):
    """
    Ensure the database and the required `binance_strategies` table exist. Create them if necessary.
    """
    base_db_connection_params = db_connection_params.copy()
    base_db_connection_params['dbname'] = 'postgres'  # Connect to the default 'postgres' database

    try:
        # Step 1: Connect to the base database to check/create the target database
        with psycopg2.connect(**base_db_connection_params) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if the target database exists
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_connection_params['dbname'],))
                if not cur.fetchone():
                    # Create the target database
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_connection_params['dbname'])))
                    print(f"Database '{db_connection_params['dbname']}' created successfully.")
                else:
                    print(f"Database '{db_connection_params['dbname']}' already exists.")
    except Exception as e:
        print(f"Error initializing database: {e}")

    # Step 2: Connect to the target database to check/create the `binance_strategies` table
    try:
        with psycopg2.connect(**db_connection_params) as conn:
            with conn.cursor() as cur:
                # Create the `binance_strategies` table if it doesn't exist
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        rsi_period INT,
                        adx_threshold FLOAT,
                        stop_loss FLOAT,
                        take_profit FLOAT,
                        rsi_oversold FLOAT,
                        rsi_overbought FLOAT,
                        bbw_threshold FLOAT,
                        lrc_slope_threshold FLOAT,
                        timeframe TEXT,
                        crypto_currency TEXT,
                        percentage_profit FLOAT CHECK (percentage_profit > 0.5),
                        final_capital FLOAT,
                        sharpe_ratio FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print(f"Table '{table_name}' is ready.")
    except Exception as e:
        print(f"Error creating table '{table_name}': {e}")

import numpy as np

def calculate_indicators(data, timeframe='timeframes', gpu=False):
    """
    Calculate indicators dynamically based on the selected timeframe.
    Use the old periods for indicator calculations.
    """
    logger.info(f"Calculating indicators for timeframe: {timeframe}, GPU: {gpu}")

    def get_periods_for_timeframe(timeframe):
        if timeframe == '5m':
            return {
                'rsi_period': 96,  # 8 hours
                'adx_period': 96,  # 8 hours
                'lrc_period': 192,  # 16 hours
                'bbw_period': 96,  # 8 hours
                'volatility_period': 96,  # 8 hours
                'sma_short': 12,   # Short SMA period (1 hour)
                'sma_long': 96     # Long SMA period (8 hours)
            }
        elif timeframe == '1h':
            return {
                'rsi_period': 24,  # 1 day
                'adx_period': 24,  # 1 day
                'lrc_period': 48,  # 2 days
                'bbw_period': 24,  # 1 day
                'volatility_period': 24,  # 1 day
                'sma_short': 8,    # Short SMA period (8 hours)
                'sma_long': 24     # Long SMA period (1 day)
            }
        elif timeframe == '1d':
            return {
                'rsi_period': 14,
                'adx_period': 14,
                'lrc_period': 14,
                'bbw_period': 20,
                'volatility_period': 14,
                'sma_short': 5,    # Short SMA period (5 days)
                'sma_long': 20     # Long SMA period (20 days)
            }
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    # Required columns for calculations
    required_columns = {'close', 'open', 'high', 'low', 'volume'}
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing. Data structure issue.")

    periods = get_periods_for_timeframe(timeframe)

    try:
        # Convert data columns to NumPy arrays
        close = np.array(data['close'].values, dtype=np.float32)
        high = np.array(data['high'].values, dtype=np.float32)
        low = np.array(data['low'].values, dtype=np.float32)

        logger.info(f"Starting CPU calculations. Data length: {len(data)}")

        # RSI Calculation
        logger.debug("Calculating RSI...")
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = -np.where(delta < 0, delta, 0)
        avg_gain = pd.Series(gain).ewm(span=periods['rsi_period'], adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=periods['rsi_period'], adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        data['rsi'] = rsi
        logger.debug("RSI calculated successfully.")

        # ADX Calculation
        logger.debug("Calculating ADX...")
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        pdm = np.maximum(high - np.roll(high, 1), 0)
        ndm = np.maximum(np.roll(low, 1) - low, 0)
        atr = pd.Series(tr).ewm(span=periods['adx_period'], adjust=False).mean().values
        pdi = (pd.Series(pdm).ewm(span=periods['adx_period'], adjust=False).mean().values / atr) * 100
        ndi = (pd.Series(ndm).ewm(span=periods['adx_period'], adjust=False).mean().values / atr) * 100
        dx = np.abs(pdi - ndi) / (pdi + ndi + 1e-6) * 100
        adx = pd.Series(dx).ewm(span=periods['adx_period'], adjust=False).mean().values
        data['adx'] = adx
        logger.debug("ADX calculated successfully.")

        # Bollinger Bands (BBW)
        logger.debug("Calculating Bollinger Bands Width (BBW)...")
        bb_ema = pd.Series(close).ewm(span=periods['bbw_period'], adjust=False).mean().values
        std_dev = pd.Series(close - bb_ema).rolling(window=periods['bbw_period']).std().values
        bbw = (2 * std_dev) / bb_ema
        bbw = np.nan_to_num(bbw)
        bbw[:periods['bbw_period']] = bbw[periods['bbw_period']]  # Use the first valid BBW
        data['bbw'] = bbw
        logger.debug("BBW calculated successfully.")

        # Volatility Calculation
        logger.debug("Calculating Volatility...")
        returns = np.diff(close, prepend=close[0]) / close
        volatility = pd.Series(returns ** 2).rolling(window=periods['volatility_period']).mean().apply(np.sqrt).values
        data['volatility'] = volatility
        logger.debug("Volatility calculated successfully.")

        # Linear Regression Channel (LRC) Slope
        logger.debug("Calculating LRC Slope...")
        data['lrc_slope'] = (
            data['close']
            .rolling(window=periods['lrc_period'])
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == periods['lrc_period'] else np.nan, raw=True)
        )
        data['lrc_slope'].fillna(0, inplace=True)
        logger.debug("LRC Slope calculated successfully.")

        # SMA Calculations
        logger.debug("Calculating SMAs...")
        data['sma_short'] = data['close'].rolling(window=periods['sma_short']).mean()
        data['sma_long'] = data['close'].rolling(window=periods['sma_long']).mean()
        logger.debug("SMAs calculated successfully.")

        # Ensure the 'price' column exists
        data['price'] = data['close']  # If 'price' is expected to match 'close'

        logger.info("Indicators calculated using CPU.")

        return data

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise






def validate_adjusted_data(data, original_shape):
    try:
        if data.shape[0] != original_shape[0]:
            logger.warning(f"Row count mismatch: {data.shape[0]} (adjusted) vs {original_shape[0]} (original)")
            return False
        if data.isnull().values.any():
            logger.warning("Adjusted data contains NaN values.")
            return False
        # Adjust column names to match those used in calculate_indicators
        required_columns = {'rsi', 'adx', 'bbw', 'ema1', 'ema2', 'volatility'}
        missing_cols = required_columns - set(data.columns)
        if missing_cols:
            logger.warning(f"Adjusted data is missing required columns: {missing_cols}")
            return False
        logger.info("Adjusted data validated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error validating adjusted data: {e}")
        return False

    
def validate_data(data, one_minute_data):
    if data.empty or one_minute_data is None:
        raise ValueError("Data or one-minute candle data is empty.")
    if 'datetime' not in one_minute_data:
        raise ValueError("One-minute data lacks 'datetime' column.")
    if data.isnull().values.any():
        raise ValueError("Fetched data contains NaN values.")







def prefetch_1m_data(start_time=None, end_time=None):

    query = """
    SELECT 
        original_timestamp AS datetime,
        open,
        high,
        low,
        close,
        volume
    FROM binance_data
    WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
    """
    if start_time and end_time:
        query += " AND original_timestamp BETWEEN %s AND %s ORDER BY original_timestamp;"
    else:
        query += " ORDER BY original_timestamp;"

    try:
        params = ()
        if start_time and end_time:
            start_time_unix = int(pd.Timestamp(start_time).timestamp() * 1000)
            end_time_unix = int(pd.Timestamp(end_time).timestamp() * 1000)
            params = (start_time_unix, end_time_unix)

        with psycopg2.connect(**db_connection_params) as conn:
            # Execute the query and fetch data
            data = pd.read_sql_query(query, conn, params=params)
            data['datetime'] = pd.to_datetime(data['datetime'], unit='ms', errors='coerce')
            if data['datetime'].isnull().any():
                logger.warning("Null values found in 'datetime' after conversion. Removing invalid rows.")
                data = data.dropna(subset=['datetime'])

            # Log number of rows fetched
            logger.info(f"1m data fetched: {len(data)} rows.")

            # Log the first 5 rows of the data
            if not data.empty:
                logger.info("First 5 rows of fetched 1m data:")
                logger.info(data.head())
            else:
                logger.warning("Fetched 1m data is empty.")

            return data

    except Exception as e:
        logger.error(f"Error in prefetch_1m_data: {e}")
        return pd.DataFrame()
    
# Retain invalid areas with NaNs in prefetch_data
def filter_prefetch_data_with_reduced(prefetch_data, reduced_data):
    """
    Bypass filtering prefetch_data with reduced_data indices.
    Simply return the prefetch_data as-is.
    """
    logger.info("Skipping filtering for 1m data. Returning prefetch_data as-is.")
    return prefetch_data




def backtest_strategy(data, params, trade_type='both', prefetch_data=None, timeframe='5m'):
    """
    Backtest a strategy with given parameters and support for long and short trades.
    Adjust stop loss and take profit based on the selected timeframe.
    """
    logger.info("Starting backtest...")

    try:
        # Validate `data` and `prefetch_data`
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("`data` must have a DatetimeIndex.")
            raise ValueError("`data` must have a DatetimeIndex.")
        if prefetch_data is not None and not isinstance(prefetch_data.index, pd.DatetimeIndex):
            logger.error("`prefetch_data` must have a DatetimeIndex.")
            raise ValueError("`prefetch_data` must have a DatetimeIndex.")

        logger.info(f"Validated input data: {len(data)} rows. Prefetched data: {len(prefetch_data) if prefetch_data is not None else 0} rows.")
    except Exception as e:
        logger.error(f"Initial validation error: {e}")
        raise

    # Set stop loss and take profit coefficients based on timeframe
    if timeframe == '5m':
        PROFIT_BORDER_COEFF = 0.05  # 5% for 5m
        STOP_BORDER_COEFF = 0.02   # 2% for 5m
    elif timeframe in ['1h', '1d']:
        PROFIT_BORDER_COEFF = 0.1  # 10% for 1h and 1d
        STOP_BORDER_COEFF = 0.05   # 5% for 1h and 1d
    else:
        logger.error(f"Unsupported timeframe: {timeframe}")
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    logger.info(f"Using PROFIT_BORDER_COEFF={PROFIT_BORDER_COEFF} and STOP_BORDER_COEFF={STOP_BORDER_COEFF} for timeframe={timeframe}")

    # Initialize key variables
    capital = 100000  # Initial capital
    position = 0  # Position status (0 = no position, 1 = long position, -1 = short position)
    trailing_price = None  # Initialize trailing price
    initial_price = None  # Price at which the position was entered
    profit_border = None  # Profit border price
    stop_border = None  # Stop-loss border price
    decision_border = None  # Decision border for trailing updates
    shares = 0  # Number of shares held
    positions = []  # Track all positions for detailed analysis

    # Backtesting loop
    for i in tqdm(range(1, len(data)), desc="Backtesting Progress"):
        try:
            price_i = data['price'].iloc[i]
            if pd.isna(price_i):
                logger.warning(f"Skipping row {i} ({data.index[i]}): Price is NaN.")
                if position != 0:
                    logger.info("Exiting active position due to invalid data.")
                    capital += (price_i - initial_price) * shares  # Close position
                    position = 0
                    trailing_price, initial_price, profit_border, stop_border, decision_border = None, None, None, None, None
                continue


            # Retrieve indicators for the current row
            rsi_i = data['rsi'].iloc[i]
            adx_i = data['adx'].iloc[i]
            bbw_i = data['bbw'].iloc[i]
            lrc_slope_i = data['lrc_slope'].iloc[i]

            # Entry Signal Logic
            if position == 0:  # No active position
                if (trade_type in ['long', 'both'] and
                    rsi_i > params['rsi_oversold'] and
                    bbw_i > params['bbw_threshold'] and
                    adx_i > params['adx_threshold'] and
                    lrc_slope_i < params['lrc_slope_threshold']):  # LRC slope positive logic
                    position = 1
                    initial_price = price_i
                    trailing_price = price_i
                    profit_border = initial_price * (1 + PROFIT_BORDER_COEFF)
                    stop_border = initial_price * (1 - STOP_BORDER_COEFF)
                    decision_border = trailing_price * (1 + PROFIT_BORDER_COEFF / 4)
                    allocation = capital * 0.05
                    shares = allocation / price_i if price_i > 0 else 0
                    capital -= allocation
                    logger.info(f"Entered LONG at {price_i:.2f}. Allocation: {allocation:.2f}, Shares: {shares:.4f}.")
                    positions.append({
                        'entry_price': float(initial_price),
                        'entry_time': data.index[i],
                        'position_type': 'long',
                        'shares': float(shares),
                        'allocation': allocation
                    })
                    continue

                elif (trade_type in ['short', 'both'] and
                      rsi_i < params['rsi_overbought'] and
                      bbw_i > params['bbw_threshold'] and
                      adx_i > params['adx_threshold'] and
                      lrc_slope_i > params['lrc_slope_threshold']):  # LRC slope negative logic
                    position = -1
                    initial_price = price_i
                    trailing_price = price_i
                    profit_border = initial_price * (1 - PROFIT_BORDER_COEFF)
                    stop_border = initial_price * (1 + STOP_BORDER_COEFF)
                    decision_border = trailing_price * (1 - PROFIT_BORDER_COEFF / 4)
                    allocation = capital * 0.05
                    shares = allocation / price_i if price_i > 0 else 0
                    capital -= allocation
                    logger.info(f"Entered SHORT at {price_i:.2f}. Allocation: {allocation:.2f}, Shares: {shares:.4f}.")
                    positions.append({
                        'entry_price': float(initial_price),
                        'entry_time': data.index[i],
                        'position_type': 'short',
                        'shares': float(shares),
                        'allocation': allocation
                    })
                    continue

            # Simulate using 1-minute candles if prefetch_data is available
            if prefetch_data is not None and position != 0:
                start_time = data.index[i - 1]
                end_time = data.index[i]
                one_minute_candles = prefetch_data.loc[start_time:end_time]

                if one_minute_candles.empty:
                    logger.warning(f"No 1m data found for interval {start_time} to {end_time}. Exiting active position.")
                    capital += (price_i - initial_price) * shares  # Close position
                    position = 0
                    trailing_price, initial_price, profit_border, stop_border, decision_border = None, None, None, None, None
                    continue

                for _, minute_candle in one_minute_candles.iterrows():
                    minute_price = minute_candle['close']

                    # Update trailing stop dynamically
                    if position == 1:  # Long position
                        if minute_price >= profit_border:
                            # Exit at profit border
                            exit_price = profit_border
                            profit_or_loss = (exit_price - initial_price) * shares
                            capital += allocation + profit_or_loss
                            positions[-1].update({
                                'exit_price': float(exit_price),
                                'exit_time': minute_candle.name,
                                'profit_or_loss': float(profit_or_loss)
                            })
                            position = 0
                            trailing_price = None
                            break
                        elif minute_price > decision_border:
                            # Update trailing price and adjust profit and stop borders
                            trailing_price = max(trailing_price, minute_price)
                            profit_border = trailing_price * (1 + PROFIT_BORDER_COEFF)
                            stop_border = trailing_price * (1 - STOP_BORDER_COEFF)
                            decision_border = trailing_price * (1 + PROFIT_BORDER_COEFF / 4)
                            logger.info(f"Trailing price updated to {trailing_price:.2f}. Profit border: {profit_border:.2f}, Stop border: {stop_border:.2f}, Decision border: {decision_border:.2f}.")
                        elif minute_price <= stop_border:
                            # Exit at stop border
                            exit_price = stop_border
                            profit_or_loss = (exit_price - initial_price) * shares
                            capital += allocation + profit_or_loss
                            logger.info(f"Exited LONG at {exit_price:.2f} (Stop Border Hit).")
                            positions[-1].update({
                                'exit_price': float(exit_price),
                                'exit_time': minute_candle.name,
                                'profit_or_loss': float(profit_or_loss)
                            })
                            position = 0
                            trailing_price = None
                            break

                    elif position == -1:  # Short position
                        if minute_price <= profit_border:
                            # Exit at profit border
                            exit_price = profit_border
                            profit_or_loss = (initial_price - exit_price) * shares
                            capital += allocation + profit_or_loss
                            logger.info(f"Exited SHORT at {exit_price:.2f} (Profit Border Hit).")
                            positions[-1].update({
                                'exit_price': float(exit_price),
                                'exit_time': minute_candle.name,
                                'profit_or_loss': float(profit_or_loss)
                            })
                            position = 0
                            trailing_price = None
                            break
                        elif minute_price < decision_border:
                            # Update trailing price and adjust profit and stop borders
                            trailing_price = min(trailing_price, minute_price)
                            profit_border = trailing_price * (1 - PROFIT_BORDER_COEFF)
                            stop_border = trailing_price * (1 + STOP_BORDER_COEFF)
                            decision_border = trailing_price * (1 - PROFIT_BORDER_COEFF / 4)
                            logger.info(f"Trailing price updated to {trailing_price:.2f}. Profit border: {profit_border:.2f}, Stop border: {stop_border:.2f}, Decision border: {decision_border:.2f}.")
                        elif minute_price >= stop_border:
                            # Exit at stop border
                            exit_price = stop_border
                            profit_or_loss = (initial_price - exit_price) * shares
                            capital += allocation + profit_or_loss
                            logger.info(f"Exited SHORT at {exit_price:.2f} (Stop Border Hit).")
                            positions[-1].update({
                                'exit_price': float(exit_price),
                                'exit_time': minute_candle.name,
                                'profit_or_loss': float(profit_or_loss)
                            })
                            position = 0
                            trailing_price = None
                            break


        except Exception as e:
            logger.error(f"Error at row {i} during backtesting: {e}")

    logger.info(f"Final capital after backtest: {capital:.2f}")
    return capital, positions


def evaluate_strategy_parallel(args):
    """
    Wrapper for evaluating strategies in parallel.
    """
    # Unpack arguments and directly call fitness_function
    ind, data, crypto_currency, timeframe, one_minute_data = args
    return fitness_function(ind, data, crypto_currency, timeframe, one_minute_data)

def genetic_algorithm(bounds, data, crypto_currency, timeframe, one_minute_data, pop_size=500, generations=10, mutation_prob=0.1, elite_frac=0.1):
    """
    Genetic algorithm optimized for batch fitness evaluation on CPU.
    """
    num_params = len(bounds)

    def initialize_population():
        return [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_params)] for _ in range(pop_size)]

    def mutate_population(population):
        return [
            [
                individual[i] + np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                if random.random() < mutation_prob else individual[i]
                for i in range(num_params)
            ]
            for individual in population
        ]

    def breed_population(population):
        offspring = []
        for _ in range(pop_size - len(population)):
            parents = random.sample(population, 2)
            offspring.append([random.uniform(parents[0][i], parents[1][i]) for i in range(num_params)])
        return offspring

    def select_profitable_strategies(population, fitnesses):
        return [population[i] for i, fitness in enumerate(fitnesses) if fitness < 0]

    # Initialize population
    population = initialize_population()
    elites = []

    for gen in range(generations):
        logger.info(f"--- Generation {gen + 1}/{generations} ---")

        # Evaluate the entire population in one batch #
        fitnesses = fitness_function(population, data, crypto_currency, timeframe, one_minute_data, gen)

        # Convert fitnesses to NumPy array if needed
        fitnesses = np.array(fitnesses)

        # Log fitness statistics #
        best_fitness = np.min(fitnesses)
        avg_fitness = np.mean(fitnesses)
        logger.info(f"Best fitness: {best_fitness}, Average fitness: {avg_fitness}")

        # Elitism #
        elite_count = max(1, int(elite_frac * len(population)))
        sorted_indices = np.argsort(fitnesses)
        current_elites = [population[i] for i in sorted_indices if fitnesses[i] < 0][:elite_count]
        elites.extend(current_elites)
        elites = list({tuple(e): e for e in elites}.values())  # Remove duplicates #

        # Breed and mutate #
        elite_clones = [list(e) for e in elites]
        offspring = breed_population(population + elite_clones)
        mutated_offspring = mutate_population(offspring)
        combined_population = elite_clones + mutated_offspring

        # Evaluate the new combined population #
        fitnesses = fitness_function(combined_population, data, crypto_currency, timeframe, one_minute_data, gen)

        # Convert fitnesses to NumPy array if needed
        fitnesses = np.array(fitnesses)

        # Select profitable strategies for the next generation #
        population = select_profitable_strategies(combined_population, fitnesses)

        # Maintain population size #
        while len(population) < pop_size:
            population.append(random.choice(initialize_population()))

        logger.info(f"Generation {gen + 1} completed. Population size: {len(population)}")

    # Final evaluation #
    final_fitnesses = fitness_function(population, data, crypto_currency, timeframe, one_minute_data, gen)

    # Convert final_fitnesses to NumPy array if needed
    final_fitnesses = np.array(final_fitnesses)

    final_population = [ind for i, ind in enumerate(population) if final_fitnesses[i] < 0]

    return final_population, final_fitnesses



# Configure logging dynamically based on the instance ID
def setup_logging(instance_id, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"instance_{instance_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Parse arguments for instance differentiation
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run strategy generator.")
    parser.add_argument("--instance", type=int, default=1, help="Instance ID for multiprocessing")
    return parser.parse_args()

# Main Optimization Loop
if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    instance_id = args.instance

    # Setup logging for this instance
    logger = setup_logging(instance_id)

    try:
        logger.info(f"Starting instance {instance_id}")

        # Ensure the `binance_strategies` table exists
        initialize_database(db_connection_params, table_name="binance_strategies_filtered_complete")
        logger.info("Database and table initialization complete.")

        # Optimization loop
        with psycopg2.connect(**db_connection_params) as conn:
            for crypto_currency in cryptocurrencies:
                for timeframe in timeframes:
                    logger.info(f"--- Instance {instance_id}: Optimizing {crypto_currency} ({timeframe}) ---")

                    try:
                        # Step 1: Fetch the complete dataset
                        logger.info(f"Fetching complete data for {crypto_currency} ({timeframe})...")
                        full_data = fetch_btc_data_from_db(crypto_currency, timeframe)
                        if full_data.empty:
                            logger.warning(f"No complete data fetched for {crypto_currency} ({timeframe}). Skipping...")
                            continue

                        # Step 2: Calculate indicators for the full dataset
                        logger.info("Calculating indicators for the full dataset...")
                        full_data_with_indicators = calculate_indicators(full_data.copy(), timeframe, gpu=True)

                        # Step 3: Fetch the reduced dataset (if using filtered data)
                        if use_filtered_data:
                            logger.info(f"Fetching reduced data for {crypto_currency} ({timeframe})...")
                            reduced_data = fetch_reduced_btc_data_from_db(crypto_currency, timeframe)
                            if reduced_data.empty:
                                logger.warning(f"No reduced data fetched for {crypto_currency} ({timeframe}). Skipping...")
                                filtered_data_with_indicators = None
                            else:
                                logger.info("Overlaying reduced dataset dates on full dataset indicators...")
                                filtered_data_with_indicators = overlay_with_reduced_dates(full_data_with_indicators, reduced_data)
                        else:
                            filtered_data_with_indicators = None

                        # Plot the data with indicators

                        # Plot the data with indicators (conditional on instance_id)
                        if instance_id == 1:
                            logger.info("Plotting indicators for the full dataset...")
                            plot_indicators(full_data_with_indicators, timeframe)

                            if filtered_data_with_indicators is not None:
                                logger.info("Plotting indicators for the filtered dataset...")
                                plot_indicators(filtered_data_with_indicators, timeframe)
                        else:
                            logger.info(f"Skipping plotting for instance {instance_id}.")


                        # Step 4: Prefetch 1m candles
                        start_time = full_data.index.min()
                        end_time = full_data.index.max()
                        logger.info(f"Prefetching 1m candles from {start_time} to {end_time}...")
                        one_minute_data = prefetch_1m_data(start_time, end_time)
                        
                        if one_minute_data.empty:
                            logger.warning(f"No 1m candle data available. Skipping finer granularity testing for {crypto_currency} ({timeframe}).")
                            one_minute_data = None
                        else:
                            # Step 5: Filter prefetch data using reduced data (if applicable)
                            if use_filtered_data and reduced_data is not None:
                                logger.info("Filtering prefetch data using reduced dataset...")
                                one_minute_data = filter_prefetch_data_with_reduced(one_minute_data, reduced_data)
                                logger.info(f"Filtered prefetch data contains {len(one_minute_data)} rows, with {one_minute_data.isna().sum().sum()} NaNs.")

                        # Step 6: Run the Genetic Algorithm
                        logger.info(f"Running Genetic Algorithm for {crypto_currency} ({timeframe}) using full dataset...")
                        final_population, final_fitnesses = genetic_algorithm(
                            bounds=[
                                (5, 100),   # RSI period
                                (10, 50),   # ADX threshold
                                (0.005, 0.1),  # Stop loss
                                (0.01, 0.1),   # Take profit
                                (10, 50),   # RSI oversold
                                (50, 90),   # RSI overbought
                                (0.01, 0.5),  # BBW threshold
                                (-1.0, 1.0)  # LRC slope threshold
                            ],
                            data=full_data_with_indicators,
                            crypto_currency=crypto_currency,
                            timeframe=timeframe,
                            one_minute_data=one_minute_data
                        )

                        if filtered_data_with_indicators is not None:
                            logger.info(f"Running Genetic Algorithm for {crypto_currency} ({timeframe}) using filtered dataset...")
                            final_population_filtered, final_fitnesses_filtered = genetic_algorithm(
                                bounds=[
                                    (5, 100),   # RSI period
                                    (10, 50),   # ADX threshold
                                    (0.005, 0.1),  # Stop loss
                                    (0.01, 0.1),   # Take profit
                                    (10, 50),   # RSI oversold
                                    (50, 90),   # RSI overbought
                                    (0.01, 0.5),  # BBW threshold
                                    (-1.0, 1.0)  # LRC slope threshold
                                ],
                                data=filtered_data_with_indicators,
                                crypto_currency=crypto_currency,
                                timeframe=timeframe,
                                one_minute_data=one_minute_data
                            )

                        logger.info(f"Optimization completed for {crypto_currency} ({timeframe}).")

                    except Exception as e:
                        logger.error(f"Error during optimization for {crypto_currency} ({timeframe}): {e}")

    except psycopg2.Error as e:
        logger.critical(f"Instance {instance_id}: Database connection error: {e}")
    except Exception as e:
        logger.critical(f"Instance {instance_id}: Critical error: {e}")


