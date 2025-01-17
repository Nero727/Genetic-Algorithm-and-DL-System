import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Database connection parameters
db_connection_params = {
    "dbname": "strategies",
    "user": "admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

# Queries to fetch data from tables
query_filtered_complete = "SELECT * FROM profitable_strategies;"
query_1d_complete = "SELECT * FROM binance_strategies_1d_complete;"


def fetch_and_plot_strategies():
    """
    Fetches strategies from the database, sorts them by percentage_profit,
    displays their profitability as histograms, and plots the top 50 strategies as bar charts.
    Calculates and prints the average indicator values for the top 50 strategies.
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_connection_params)
        cursor = connection.cursor()

        # Fetch strategies for both tables
        df_filtered = pd.read_sql(query_filtered_complete, connection)
        df_1d = pd.read_sql(query_1d_complete, connection)

        # Ensure 'percentage_profit' column exists
        if 'percentage_profit' not in df_filtered.columns or 'percentage_profit' not in df_1d.columns:
            raise ValueError("The column 'percentage_profit' is missing in one or both tables.")

        # Sort by percentage_profit
        df_filtered.sort_values(by='percentage_profit', inplace=True, ascending=False)
        df_1d.sort_values(by='percentage_profit', inplace=True, ascending=False)

        # Extract top 50 strategies
        top_50_filtered = df_filtered.head(50).reset_index(drop=True)
        top_50_1d = df_1d.head(50).reset_index(drop=True)

        # Calculate average indicator values for top 50 strategies
        avg_top_50_filtered = top_50_filtered[['adx_threshold', 'rsi_oversold', 'rsi_overbought', 'bbw_threshold', 'lrc_slope_threshold']].mean()
        avg_top_50_1d = top_50_1d[['adx_threshold', 'rsi_oversold', 'rsi_overbought', 'bbw_threshold', 'lrc_slope_threshold']].mean()

        # Print the two best-performing strategies
        print("\nTop 2 strategies from binance_strategies_filtered_complete:")
        print(df_filtered.head(2).to_dict(orient='records'))  # Prints the top 2 strategies

        print("\nTop 2 strategies from binance_strategies_1d_complete:")
        print(df_1d.head(2).to_dict(orient='records'))  # Prints the top 2 strategies

        # Print average indicator values for the top 50 strategies
        print("\nAverage Indicator Values for Top 50 Filtered Strategies:")
        print(avg_top_50_filtered.to_string())

        print("\nAverage Indicator Values for Top 50 1D Strategies:")
        print(avg_top_50_1d.to_string())

        # Plot percentage_profit as histograms
        plt.figure(figsize=(10, 6))

        # Plot for binance_strategies_filtered_complete
        plt.subplot(1, 2, 1)
        counts, bins, _ = plt.hist(
            df_filtered['percentage_profit'], bins=75, range=(0, 40),
            color='blue', alpha=0.6, edgecolor='black', linewidth=0.7
        )
        highest_bin_center = (bins[counts.argmax()] + bins[counts.argmax() + 1]) / 2
        highest_bin_count = counts.max()
        plt.title(f'Filtered Strategies (Highest Concentration: {highest_bin_center:.2f}%)')
        plt.xlabel('Percentage Profit')
        plt.ylabel('Frequency')

        # Plot for binance_strategies_1d_complete
        plt.subplot(1, 2, 2)
        counts_1d, bins_1d, _ = plt.hist(
            df_1d['percentage_profit'], bins=75, range=(0, 40),
            color='green', alpha=0.6, edgecolor='black', linewidth=0.7
        )
        highest_bin_center_1d = (bins_1d[counts_1d.argmax()] + bins_1d[counts_1d.argmax() + 1]) / 2
        highest_bin_count_1d = counts_1d.max()
        plt.title(f'1D Strategies (Highest Concentration: {highest_bin_center_1d:.2f}%)')
        plt.xlabel('Percentage Profit')
        plt.ylabel('Frequency')

        # Show histogram plots
        plt.tight_layout()
        plt.show()

        # Print highest concentration points
        print(f"\nHighest concentration for Filtered Strategies: {highest_bin_center:.2f}% ({highest_bin_count} strategies)")
        print(f"Highest concentration for 1D Strategies: {highest_bin_center_1d:.2f}% ({highest_bin_count_1d} strategies)")

        # Plot top 50 strategies for each table
        plt.figure(figsize=(10, 6))

        # Top 50 from binance_strategies_filtered_complete
        plt.subplot(1, 2, 1)
        plt.bar(top_50_filtered.index + 1, top_50_filtered['percentage_profit'], color='blue', alpha=0.7)
        plt.title('Top 50 Filtered Strategies by Percentage Profit')
        plt.xlabel('Rank')
        plt.ylabel('Percentage Profit')
        plt.xticks(range(1, 51, 5))  # Add ticks every 5 strategies

        # Top 50 from binance_strategies_1d_complete
        plt.subplot(1, 2, 2)
        plt.bar(top_50_1d.index + 1, top_50_1d['percentage_profit'], color='green', alpha=0.7)
        plt.title('Top 50 1D Strategies by Percentage Profit')
        plt.xlabel('Rank')
        plt.ylabel('Percentage Profit')
        plt.xticks(range(1, 51, 5))  # Add ticks every 5 strategies

        # Show top 50 bar charts
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Database connection closed.")



def plot_top_50_indicators_with_proportional_subplots(df_filtered, df_1d):
    """
    Plots the top 50 strategies with their indicators for each table.
    BBW and LRC occupy 1/3 of the vertical space, while ADX and RSI occupy 2/3.
    """
    # Extract top 50 strategies for each table
    top_50_filtered = df_filtered.head(50).reset_index(drop=True)
    top_50_1d = df_1d.head(50).reset_index(drop=True)

    # Create a gridspec layout to adjust subplot sizes
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1])  # ADX/RSI: 2 parts, BBW/LRC: 1 part

    # Filtered Strategies
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(top_50_filtered.index + 1, top_50_filtered['adx_threshold'], label='ADX Threshold', marker='o')
    ax1.plot(top_50_filtered.index + 1, top_50_filtered['rsi_oversold'], label='RSI Oversold', marker='o')
    ax1.plot(top_50_filtered.index + 1, top_50_filtered['rsi_overbought'], label='RSI Overbought', marker='o')
    ax1.set_title("Filtered Strategies: ADX and RSI")
    ax1.set_ylabel("Values")
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.5)

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(top_50_filtered.index + 1, top_50_filtered['bbw_threshold'], label='BBW Threshold', color='purple', marker='o')
    ax2.set_title("Filtered Strategies: BBW Threshold")
    ax2.set_ylabel("BBW Value")
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.5)

    ax3 = fig.add_subplot(gs[4, 0])
    ax3.plot(top_50_filtered.index + 1, top_50_filtered['lrc_slope_threshold'], label='LRC Slope Threshold', color='orange', marker='o')
    ax3.set_title("Filtered Strategies: LRC Slope Threshold")
    ax3.set_xlabel("Rank")
    ax3.set_ylabel("LRC Value")
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.5)

    # 1D Strategies
    ax4 = fig.add_subplot(gs[0:3, 1])
    ax4.plot(top_50_1d.index + 1, top_50_1d['adx_threshold'], label='ADX Threshold', marker='o')
    ax4.plot(top_50_1d.index + 1, top_50_1d['rsi_oversold'], label='RSI Oversold', marker='o')
    ax4.plot(top_50_1d.index + 1, top_50_1d['rsi_overbought'], label='RSI Overbought', marker='o')
    ax4.set_title("1D Strategies: ADX and RSI")
    ax4.legend(loc='upper left')
    ax4.grid(alpha=0.5)

    ax5 = fig.add_subplot(gs[3, 1])
    ax5.plot(top_50_1d.index + 1, top_50_1d['bbw_threshold'], label='BBW Threshold', color='purple', marker='o')
    ax5.set_title("1D Strategies: BBW Threshold")
    ax5.legend(loc='upper left')
    ax5.grid(alpha=0.5)

    ax6 = fig.add_subplot(gs[4, 1])
    ax6.plot(top_50_1d.index + 1, top_50_1d['lrc_slope_threshold'], label='LRC Slope Threshold', color='orange', marker='o')
    ax6.set_title("1D Strategies: LRC Slope Threshold")
    ax6.set_xlabel("Rank")
    ax6.legend(loc='upper left')
    ax6.grid(alpha=0.5)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def fetch_and_plot_strategies_with_indicators():
    """
    Fetches strategies, sorts by profitability, and plots profitability and indicators.
    """
    try:
        # Connect to the database
        connection = psycopg2.connect(**db_connection_params)

        # Fetch strategies for both tables
        df_filtered = pd.read_sql(query_filtered_complete, connection)
        df_1d = pd.read_sql(query_1d_complete, connection)

        # Ensure 'percentage_profit' column exists
        if 'percentage_profit' not in df_filtered.columns or 'percentage_profit' not in df_1d.columns:
            raise ValueError("The column 'percentage_profit' is missing in one or both tables.")

        # Sort by percentage_profit
        df_filtered.sort_values(by='percentage_profit', inplace=True, ascending=False)
        df_1d.sort_values(by='percentage_profit', inplace=True, ascending=False)

        # Plot histograms for profitability
        fetch_and_plot_strategies()  # This function already includes histogram and top 50 bar plots

        # Plot indicators for the top 50 strategies with separate BBW and LRC subplots
        plot_top_50_indicators_with_proportional_subplots(df_filtered, df_1d)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Database connection closed.")


# Run the script
if __name__ == "__main__":
    fetch_and_plot_strategies_with_indicators()

