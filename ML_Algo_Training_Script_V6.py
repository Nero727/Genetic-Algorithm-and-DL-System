import pandas as pd
import psycopg2
from ta import momentum, trend
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Database connection parameters
db_connection_params = {
    "dbname": "strategies",
    "user": "admin",
    "password": "admin",
    "host": "localhost",
    "port": "5432"
}

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


log_file = os.path.join(log_dir, "training_logs2.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)

logger = logging.getLogger("training_logger")


validation_log_file = os.path.join(log_dir, "validation_logs.txt")
validation_handler = logging.FileHandler(validation_log_file, mode="w")
validation_handler.setLevel(logging.INFO)
validation_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

validation_logger = logging.getLogger("validation_logger")
validation_logger.setLevel(logging.INFO)
validation_logger.addHandler(validation_handler)

# --- Utility Functions ---
def fetch_data(timeframe, symbol='BTCUSDT'):
    query = f"""
    SELECT original_timestamp, open, high, low, close, volume
    FROM binance_data
    WHERE timeframe = '{timeframe}' AND symbol = '{symbol}'
    """
    conn = psycopg2.connect(**db_connection_params)
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Fetched {len(df)} rows for {symbol} at {timeframe} timeframe.")
    df['original_timestamp'] = df['original_timestamp'] / 1000.0  
    return df.sort_values('original_timestamp').reset_index(drop=True)


def plot_evaluation_data(df, columns_to_plot, title="Evaluation Data Overview"):

    num_metrics = len(columns_to_plot)
    num_cols = 3  
    num_rows = (num_metrics + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(title, fontsize=16, y=0.95)  

    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        if col in df.columns:
            axes[i].plot(df[col], label=col)
            axes[i].set_title(col)
            axes[i].set_xlabel("Time Steps")
            axes[i].set_ylabel("Values")
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].set_visible(False) 

    # Hide any remaining unused axes
    for i in range(len(columns_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plots_dir = "evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Saved evaluation plot to {plot_file}")


def add_features(df):
    # Add technical indicators
    df['rsi'] = momentum.RSIIndicator(close=df['close'], window=96).rsi()
    df['bbw'] = (df['close'].ewm(span=96).mean() + 2 * df['close'].ewm(span=96).std()
                 - (df['close'].ewm(span=96).mean() - 2 * df['close'].ewm(span=96).std())) / df['close'].ewm(span=96).mean()
    df['lrc'] = df['close'].rolling(window=192).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    adx = trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=96)
    df['adc'] = df['adx'] = adx.adx()
    return df.dropna()


def create_rl_dataset(df):
    df['returns'] = df['close'].pct_change()
    df['reward'] = (df['returns'] * 100).shift(-1)  
    return df.dropna()


def clip_rewards(values, min_value=-10, max_value=10):
    return np.clip(values, min_value, max_value)

def normalize_data(df):

    columns_to_normalize = ['close', 'rsi', 'bbw', 'lrc', 'adc', 'reward']

    # Normalize the columns
    for col in columns_to_normalize:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()  # Standardization

    if 'bbw' in df.columns:
        logger.info(f"Original BBW stats: {df['bbw'].describe()}")
        df['bbw'] = clip_rewards(df['bbw'], min_value=-10, max_value=10)
        logger.info(f"Clipped BBW stats: {df['bbw'].describe()}")

    if 'reward' in df.columns:
        logger.info(f"Normalized Reward stats: {df['reward'].describe()}")
        df['reward'] = clip_rewards(df['reward'], min_value=-15, max_value=15)
        logger.info(f"Clipped Reward stats: {df['reward'].describe()}")

    return df

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) 
            if m.bias is not None:
                nn.init.zeros_(m.bias)  


def save_metrics_to_excel(metrics, file_name):

    new_data = pd.DataFrame(metrics if isinstance(metrics, list) else [metrics])
    
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:

        updated_data = new_data
    
    updated_data.to_excel(file_name, index=False)
    print(f"Metrics saved to {file_name}")



def plot_loss(step_losses, episode):

    plots_dir = "plots2"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(step_losses, label=f"Episode {episode} Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for Episode {episode}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = os.path.join(plots_dir, f"loss_episode_{episode}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved loss plot for Episode {episode} to {plot_filename}")

def log_and_save_metrics(metrics, filename, logger, title=""):

    if title:
        logger.info(title)
    logger.info(metrics)
    save_metrics_to_excel([metrics], filename)


def evaluate(env, model, data, gamma=0.99):
    rewards_m = []
    portfolio_values = []
    negative_returns = []
    evaluation_step_losses = []  
    profits_v = []
    trade_durations = []  

    model.eval()  
    validation_logger.info(f"Model is in evaluation mode: {not model.training}")

    try:
        state = env._next_observation()  
        validation_logger.info(f"Initial state shape during evaluation: {state.shape}")

        for step in range(len(data) - env.current_step):  

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)


            expected_feature_count = env.observation_space.shape[0]
            if state_tensor.shape[1] != expected_feature_count:
                raise ValueError(
                    f"State feature mismatch: Expected {expected_feature_count}, got {state_tensor.shape[1]}"
                )

            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, done, info = env.step(action)

            if len(next_state) != expected_feature_count:
                raise ValueError(
                    f"Next state feature mismatch: Expected {expected_feature_count}, got {len(next_state)}"
                )

            # Calculate evaluation loss
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            if state_tensor.shape[1] != len(['close', 'rsi', 'bbw', 'lrc', 'adx']):
                raise ValueError(
                    f"State shape mismatch: Expected {len(['close', 'rsi', 'bbw', 'lrc', 'adx'])}, got {state_tensor.shape[1]}"
                )
            with torch.no_grad():
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                next_q_values = model(next_state_tensor)
                max_next_q_value = (
                    next_q_values.max(1)[0].item() if len(next_q_values.shape) > 1 else next_q_values.max().item()
                ) if not done else 0

                # Define target Q-value
                target_q_value = reward + gamma * max_next_q_value

            predicted_q_value = (
                q_values[0, action] if len(q_values.shape) > 1 else q_values[action]
            )
            evaluation_loss = (predicted_q_value - target_q_value) ** 2
            evaluation_step_losses.append(evaluation_loss.item())
            profits_v.append(info.get("profit", 0))


            trade_durations.extend(info.get("trade_durations", []))


            validation_logger.debug(f"Step {step}: Q-Values: {q_values.cpu().numpy()}, Selected Action: {action}")
            validation_logger.debug(
                f"Step {step}: Action Taken: {action}, Reward: {reward}, "
                f"Balance: {info['balance']}, Position: {info['position']}, "
                f"Portfolio Value: {info['portfolio_value']}"
            )

            rewards_m.append(reward)
            portfolio_values.append(info["portfolio_value"])
            if reward < 0:
                negative_returns.append(reward)


            state = next_state
            if done:
                validation_logger.debug(f"Step {step}: Evaluation Complete")
                break

        avg_loss = np.mean(evaluation_step_losses) if evaluation_step_losses else 0
        validation_logger.info(f"Final Evaluation Average Loss: {avg_loss:.6f}")

        realized_profits_v = [p for p in profits_v if p != 0]
        returns_v = [
            profits_v / env.initial_balance
            for profits_v in realized_profits_v
            if env.initial_balance > 0
        ]

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0


        return {
            "Total Reward": sum(rewards_m),
            "Average Trade Return": np.mean(rewards_m) if rewards_m else 0,
            "Win Rate": env.successful_trades / env.total_trades if env.total_trades > 0 else 0,
            "Total Trades": env.total_trades,
            "Successful Trades": env.successful_trades,
            "Profit Factor": (
                sum(r for r in realized_profits_v if r > 0) /
                abs(sum(r for r in realized_profits_v if r < 0)) if realized_profits_v else float('inf')
            ),
            "Sharpe Ratio": (
                (np.mean(returns_v) / (np.std(returns_v) + 1e-8))
                if len(returns_v) > 1 else 0
            ),
            "Sortino Ratio": (
                (np.mean(returns_v) / (np.std(negative_returns) + 1e-8)) * np.sqrt(252)
                if len(negative_returns) > 0 else 0
            ),
            "Maximum Drawdown": np.max(
                (np.maximum.accumulate(np.array(portfolio_values)) - np.array(portfolio_values))
                / np.maximum.accumulate(np.array(portfolio_values))
            ) if portfolio_values else 0,
            "Average Loss": avg_loss,
            "Final Balance": env.balance,
            "Average Trade Duration": avg_trade_duration,
        }
    except Exception as e:
        validation_logger.error(f"Evaluation failed: {e}")
        return {
            "Total Reward": 0,
            "Average Trade Return": 0,
            "Win Rate": 0,
            "Total Trades": 0,
            "Successful Trades": 0,
            "Profit Factor": 0,
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0,
            "Maximum Drawdown": 0,
            "Average Loss": None,
            "Final Balance": env.initial_balance,
            "Average Trade Duration": 0, 
        }
    finally:
        model.train()



        # Calculate evaluation metrics
        total_reward = sum(rewards_m)
        avg_trade_return = np.mean(returns_v) if returns_v else 0

        total_trades = env.total_trades  
        successful_trades = env.successful_trades
        win_rate = successful_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(r for r in returns_v if r > 0)
        gross_loss = abs(sum(r for r in returns_v if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        sharpe_ratio = (
            np.mean(returns_v) / (np.std(returns_v) + 1e-8)
            if len(returns_v) > 1 else 0
        )


        downside_std = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = (
            (np.mean(returns_v) / (downside_std + 1e-8)) * np.sqrt(252)
            if len(negative_returns) > 0 else 0
        )

        portfolio_array = np.array(portfolio_values)
        peaks = np.maximum.accumulate(portfolio_array)
        drawdowns = (peaks - portfolio_array) / peaks
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0


        final_balance = env.balance

        return {
            "Total Reward": total_reward,
            "Average Trade Return": avg_trade_return,
            "Win Rate": win_rate,
            "Total Trades": total_trades,
            "Successful Trades": successful_trades,
            "Profit Factor": profit_factor,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Maximum Drawdown": max_drawdown,
            "Average Loss": avg_loss,  
            "Final Balance": final_balance,  
        }



def plot_normalized_data(df, columns_to_plot, title="Normalized Data Overview"):

    num_metrics = len(columns_to_plot)
    num_cols = 3  
    num_rows = (num_metrics + num_cols - 1) // num_cols  

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(title, fontsize=16)

    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        if col in df.columns:
            axes[i].plot(df[col], label=col)
            axes[i].set_title(col)
            axes[i].set_xlabel("Time Steps")
            axes[i].set_ylabel("Normalized Values")
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].set_visible(False)  

    for i in range(len(columns_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.show()




def save_checkpoint(checkpoint_dir, model, optimizer, episode, epsilon):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'epsilon': epsilon
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        logger.info("No checkpoint found. Starting from scratch.")
        return None

    # Find the most recent checkpoint by episode number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint.get('episode', 0)
    epsilon = checkpoint.get('epsilon', 1.0)
    
    logger.info(f"Loaded checkpoint from {latest_checkpoint}")
    return episode, epsilon



# --- RL Environment ---

class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super(BTCTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.action_space = spaces.Discrete(4)  # 0: Hold, 1: Long, 2: Short, 3: Close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(['close', 'rsi', 'bbw', 'lrc', 'adx']),), dtype=np.float32
        )

        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.position = 0  
        self.position_size = 0
        self.position_value = 0
        self.holding_duration = 0
        self.max_holding_duration = 0

        self.trade_duration = 0  
        self.trade_durations = []  

        # Initialize trade metrics
        self.total_trades = 0
        self.successful_trades = 0
        self.sum_rewards = 0  

    def reset(self):
    
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.position_value = 0
        self.holding_duration = 0
        self.max_holding_duration = 0
        self.trade_duration = 0
        self.trade_durations = []

        # Reset trade metrics
        self.total_trades = 0
        self.successful_trades = 0
        self.sum_rewards = 0

        return self._next_observation()

    def _next_observation(self):
        obs = self.df[['close', 'rsi', 'bbw', 'lrc', 'adx']].iloc[self.current_step].values
        logger.info(f"_next_observation() - Current Step: {self.current_step}, Observation: {obs}")
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        trade_amount = self.initial_balance * 0.05  
        penalty_multiplier = 0.01  # Multiplier to punish negative trades more severely
        reward = 0.0  
        pnl = 0
        profit = 0

        # Handle actions
        if action == 0:  # Hold
            reward += 0.1  
            self.holding_duration += 1  
            self.max_holding_duration = max(self.max_holding_duration, self.holding_duration)
        else:
            self.holding_duration = 0  

        if action == 1:  # Long
            if self.position == 0:  
                self.position = 1
                self.position_size = trade_amount / current_price
                self.balance -= trade_amount
                self.position_value = current_price
                self.trade_duration = 1  
            else:
                reward -= 0.01 # Penalize redundant action
                self.trade_duration += 1  

        elif action == 2:  # Short
            if self.position == 0:  # Open a new short position
                self.position = -1
                self.position_size = trade_amount / current_price
                self.balance -= trade_amount
                self.position_value = current_price
                self.trade_duration = 1  
            else:
                reward -= 0.01 
                self.trade_duration += 1  # Increment trade duration if already in a position

        elif action == 3:  # Close Position
            if self.position != 0:  
                pnl = 0
                if self.position == 1:  
                    pnl = self.position_size * (current_price - self.position_value)
                elif self.position == -1:  
                    pnl = self.position_size * (self.position_value - current_price)

                self.balance += pnl + trade_amount
                if pnl > 0:
                    reward += pnl * 0.01  
                    self.successful_trades += 1  
                else:
                    reward += pnl * penalty_multiplier  # Penalize negative PnL

                self.total_trades += 1  
                profit = pnl

                self.trade_durations.append(self.trade_duration)
                self.trade_duration = 0  

                # Reset position
                self.position = 0
                self.position_size = 0
                self.position_value = 0
            else:
                reward -= 0.1  
                profit = 0

        # Accumulate rewards
        self.sum_rewards += reward
        current_portfolio_value = self.balance + (self.position_size * current_price if self.position != 0 else 0)

        # Check if the environment has reached its end
        done = self.current_step >= len(self.df) - 1
        if not done:
            self.current_step += 1

        next_observation = self._next_observation() if not done else np.zeros(self.observation_space.shape)

       
        feature_names = self.df.columns[:-1]  

        
        current_features = {feature_names[i]: self.df.iloc[self.current_step, i] for i in range(len(feature_names))}
        logger.info(f"Step {self.current_step}: Current Features: {current_features}")

        return next_observation, reward, done, {
            "balance": self.balance,
            "position": self.position,
            "position_size": self.position_size,
            "portfolio_value": current_portfolio_value,
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "sum_rewards": self.sum_rewards,
            "profit": profit,  
            "holding_duration": self.holding_duration,  
            "max_holding_duration": self.max_holding_duration,  
            "trade_durations": self.trade_durations,  
        }






# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
       
        if next_state is not None and state is not None:
            if state.shape != next_state.shape:
                next_state = np.zeros_like(state, dtype=np.float32)
                logging.warning(f"Inconsistent state shapes detected. Replacing next_state with zeros: {state.shape} != {next_state.shape}")

        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to arrays with consistent shapes
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)  


# --- Main Training Loop ---
def main():

    checkpoint_dir = "checkpoints2"
    os.makedirs(checkpoint_dir, exist_ok=True)


    data = fetch_data('5m')
    data = add_features(data)
    data = create_rl_dataset(data)
    data = normalize_data(data)

    #data = data[-25000:]  

    # Plot normalized data
    columns_to_plot = ['close', 'rsi', 'bbw', 'lrc', 'adc', 'reward']  
    plot_normalized_data(data, columns_to_plot, title="Normalized Features for Training")

    # Splitting the data
    train_split = 0.7
    val_split = 0.15
    train_end = int(len(data) * train_split)
    val_end = int(len(data) * (train_split + val_split))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    columns_to_plot = ['close', 'rsi', 'bbw', 'lrc', 'adc', 'reward']
    plot_evaluation_data(val_data, columns_to_plot, title="Evaluation Data Overview")

    # Initialize environments
    train_env = BTCTradingEnv(train_data)
    val_env = BTCTradingEnv(val_data)
    test_env = BTCTradingEnv(test_data)

    # Model setup
    state_size = len(['close', 'rsi', 'bbw', 'lrc', 'adx'])  # Updated to 5 features
    action_size = 4
    model = DQN(state_size, action_size).to(device)
    initialize_weights(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(10000)

    # Hyperparameters
    episodes = 1000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 512

    # Attempt to load from checkpoint
    start_episode = 0
    checkpoint = load_checkpoint(checkpoint_dir, model, optimizer)
    if checkpoint:
        start_episode, epsilon = checkpoint
    else:
        start_episode = 0
        epsilon = 1.0


    for episode in range(start_episode, episodes):
        model.train()

        # --- Training ---
        state = train_env.reset()
        total_reward = 0
        step_losses = []
        portfolio_values = []
        rewards_m = []  
        total_trades = 0
        successful_trades = 0
        profits = []
        trade_durations = [] 

        logger.info(f"Starting Episode {episode + 1}/{episodes} Model set to training mode: {model.training}")

        for step in tqdm(range(len(train_data))):
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Select action
            if random.random() < epsilon:
                action = train_env.action_space.sample()
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
                logger.info(f"Training Step {step}: Q-Values: {q_values.detach().cpu().numpy()}")

            # Perform action in the environment
            next_state, reward, done, info = train_env.step(action)
            total_trades = info.get("total_trades", total_trades)
            successful_trades = info.get("successful_trades", successful_trades)
            portfolio_values.append(info["portfolio_value"])
            rewards_m.append(reward)
            profits.append(info.get("profit", 0))
            trade_durations.extend(info.get("trade_durations", [])) 

            logger.info(
                f"Training Step {step}: Action: {action}, Reward: {reward:.6f}, "
                f"Portfolio Value: {info['portfolio_value']:.2f}, Balance: {info['balance']:.2f}, "
                f"Position: {info['position']}, Position Size: {info['position_size']:.2f}"
            )


            replay_buffer.push(state, action, reward, next_state, done)


            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1)[0]
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = loss_fn(q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()

                # Log gradients for each parameter
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        gradient_mean = param.grad.mean().item() if param.grad is not None else None
                        gradient_std = param.grad.std().item() if param.grad is not None else None
                        logger.info(f"Layer: {name}, Gradient Mean: {gradient_mean}, Gradient Std: {gradient_std}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                step_losses.append(loss.item())

            if done:
                break

        realized_profits = [p for p in profits if p != 0]
        returns = [profit / train_env.initial_balance for profit in realized_profits]
        logger.info(f"Realized profits: {realized_profits}")
        logger.info(f"Returns: {returns}")


        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0


        avg_loss_train = np.mean(step_losses) if step_losses else None
        win_rate = successful_trades / total_trades if total_trades > 0 else 0

        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + 1e-8)
            if len(returns) > 1 else 0
        )

        profit_factor = (
            sum(r for r in realized_profits if r > 0) /
            abs(sum(r for r in realized_profits if r < 0)) if realized_profits else float('inf'))

        portfolio_array = np.array(portfolio_values)
        peaks = np.maximum.accumulate(portfolio_array)
        drawdowns = (peaks - portfolio_array) / peaks
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        avg_trade_return = np.mean(returns) if returns else 0

        log_and_save_metrics(
            metrics={
                "Episode": episode + 1,
                "Profit Factor": profit_factor,
                "Training Total Reward": total_reward,
                "Training Average Loss": avg_loss_train,
                "Training Sharpe Ratio": sharpe_ratio,
                "Training Max Drawdown": max_drawdown,
                "Training Average Trade Return": avg_trade_return,
                "Training Total Trades": total_trades,
                "Training Successful Trades": successful_trades,
                "Training Win Rate": win_rate,
                "Final Balance": train_env.balance,
                "Average Trade Duration": avg_trade_duration,  # New metric
            },
            filename="training_metrics.xlsx",
            logger=logger,
            title=f"Episode {episode + 1} Training Metrics"
        )


        plot_loss(step_losses, episode + 1)


        epsilon = max(epsilon * epsilon_decay, epsilon_min)



        # --- Validation ---
        if (episode + 1) % 5 == 0:  


            val_env = BTCTradingEnv(val_data)  
            val_env.reset() 

            # Log model state before evaluation
            validation_logger.info("Validation: Logging model weights before evaluation.")
            for name, param in model.named_parameters():
                validation_logger.info(f"Layer: {name}, Mean: {param.mean()}, Std: {param.std()}")

            model.eval()  
            validation_results = evaluate(val_env, model, val_data)

            validation_logger.info("Validation: Logging model weights after evaluation.")
            for name, param in model.named_parameters():
                validation_logger.info(f"Layer: {name}, Mean: {param.mean()}, Std: {param.std()}")

            if validation_results is None:
                validation_logger.error("Validation results are None. Skipping validation metrics logging.")
            else:
                validation_logger.info(f"Validation Results for Episode {episode + 1}: {validation_results}")
                log_and_save_metrics(
                    metrics={"Episode": episode + 1, **validation_results},
                    filename="validation_metrics.xlsx",
                    logger=validation_logger,
                    title=f"Validation Results: Episode {episode + 1}"
                )

            model.train()  




        # Save checkpoint every 5 episodes
        if (episode + 1) % 5 == 0:
            save_checkpoint(checkpoint_dir, model, optimizer, episode + 1, epsilon)

    # --- Testing ---
    test_results = evaluate(test_env, model, test_data)
    log_and_save_metrics(
        metrics=test_results,
        filename="test_metrics.xlsx",
        logger=logger,
        title="Testing Results"
    )



if __name__ == "__main__":
    main()


