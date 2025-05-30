import backtrader as bt
import datetime
import pandas as pd
import numpy as np
import os
import ray
from ray.tune.registry import register_env

from .utils import load_historical_data
# We will import TradingAgent when we integrate it more deeply
# from .agent import TradingAgent 
# from .environment import TradingEnvironment # The environment is used for training, not directly in backtesting strategy

class RLTradingStrategy(bt.Strategy):
    params = (
        ('agent', None), # Will hold our RL agent
        ('lookback_window_size', 10), # To match environment's expectation for observation
        ('features', ['Open', 'High', 'Low', 'Close', 'Volume']),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        self.order = None
        self.bar_executed = 0

        if self.p.agent is None:
            raise ValueError("RL Agent must be provided to RLTradingStrategy")
        
        self.agent = self.p.agent
        self.num_features = len(self.p.features)

        # We need to maintain a buffer for the lookback window for the agent
        self.databuffer = pd.DataFrame(columns=self.p.features, 
                                       index=range(self.p.lookback_window_size + 1))
        self.buffer_idx = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.Status[order.status]}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def _get_observation(self):
        # Populate the current bar's data into the buffer
        current_data_point = pd.Series([
            self.dataopen[0],
            self.datahigh[0],
            self.datalow[0],
            self.dataclose[0],
            self.datavolume[0]
        ], index=self.p.features)
        
        # Shift buffer and add new data point
        self.databuffer = pd.concat([
            self.databuffer.iloc[1:], 
            current_data_point.to_frame().T
        ], ignore_index=True)

        if len(self) < self.p.lookback_window_size: # Not enough data yet for a full observation
             # Create a padded observation if needed, or rely on agent to handle smaller inputs if designed for it
             # For now, let's return zeros, assuming agent expects fixed size
            return np.zeros(((self.p.lookback_window_size + 1) * self.num_features, ), dtype=np.float32)

        # Flatten the buffer for the agent
        # The observation should match what the TradingEnvironment produces
        obs_data = self.databuffer.values
        return obs_data.flatten().astype(np.float32)

    def next(self):
        # Skip if an order is pending
        if self.order:
            return

        # Wait for enough bars to form a complete lookback window
        if len(self) < self.p.lookback_window_size:
            # Update buffer even if not acting
            self._get_observation() # Call to maintain buffer
            return

        observation = self._get_observation()
        
        # Get action from RL agent
        # For now, action is an integer: 0 (Hold), 1 (Buy), 2 (Sell)
        action = self.agent.predict(observation) 

        # self.log(f'Close: {self.dataclose[0]:.2f}, Action: {action}')

        # Execute action
        if action == 1: # Buy
            if not self.position: # If no position open
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy()
        elif action == 2: # Sell
            if self.position: # If position is open
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell()
        # Action 0 (Hold) - do nothing


def run_backtest(data_df, agent_instance, env_config, 
                 initial_cash=10000.0, commission_rate=0.001, 
                 strategy_params=None):
    cerebro = bt.Cerebro()

    # Agent is already an instance
    trading_agent = agent_instance 

    # Convert pandas DataFrame to Backtrader data feed
    # Ensure DataFrame index is datetime
    if not isinstance(data_df.index, pd.DatetimeIndex):
        # Try to convert if a common date column name exists, or raise error
        if 'Date' in data_df.columns:
            data_df['Date'] = pd.to_datetime(data_df['Date'])
            data_df.set_index('Date', inplace=True)
        elif 'Timestamp' in data_df.columns:
            data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'])
            data_df.set_index('Timestamp', inplace=True)
        else:
            raise ValueError("DataFrame index must be DatetimeIndex or contain 'Date'/'Timestamp' column for bt.feeds.PandasData")
    
    # Ensure OHLCV names are standard for Backtrader, or map them
    # Default for PandasData: open, high, low, close, volume, openinterest
    # Our load_historical_data already standardizes to 'Open', 'High', etc.
    # We need to pass these as lowercase to PandasData
    feed_params = {
        'datetime': None, # None means use index
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'openinterest': -1 # -1 indicates no open interest column
    }
    data_feed = bt.feeds.PandasData(dataname=data_df, **feed_params)
    cerebro.adddata(data_feed)

    # Add strategy
    actual_strategy_params = {
        'agent': trading_agent,
        'lookback_window_size': env_config.get('lookback_window_size', 10),
        'features': env_config.get('features', ['Open', 'High', 'Low', 'Close', 'Volume'])
    }
    if strategy_params:
        actual_strategy_params.update(strategy_params)
    cerebro.addstrategy(RLTradingStrategy, **actual_strategy_params)

    # Set initial cash and commission
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_rate)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Print analysis results
    strat_results = results[0]
    analysis = {}
    analysis['sharpe_ratio'] = strat_results.analyzers.sharpe_ratio.get_analysis()
    analysis['annual_return'] = strat_results.analyzers.annual_return.get_analysis()
    analysis['drawdown'] = strat_results.analyzers.drawdown.get_analysis()
    analysis['trade_analyzer'] = strat_results.analyzers.trade_analyzer.get_analysis()
    analysis['sqn'] = strat_results.analyzers.sqn.get_analysis()
    
    print("\n--- Backtest Analysis ---")
    print(f"Sharpe Ratio: {analysis['sharpe_ratio'].get('sharperatio', 'N/A')}")
    print("Annualized Return:")
    for year, ret in analysis['annual_return'].items():
        print(f"  {year}: {ret*100:.2f}%")
    print(f"Max Drawdown: {analysis['drawdown'].max.drawdown:.2f}%")
    print(f"Max Drawdown Money: {analysis['drawdown'].max.moneydown:.2f}")
    print(f"SQN: {analysis['sqn'].get('sqn', 'N/A')}")
    # Add more details from trade_analyzer if needed

    # cerebro.plot() # Optional: Plot results if matplotlib is installed and configured
    return analysis

# Path to the checkpoint you want to load for backtesting
# Example: "./rllib_checkpoints/PPO_TradingEnv-v0_..../checkpoint_000010/checkpoint-10"
# You need to replace this with an actual path to a checkpoint file saved by train_agent.py
CHECKPOINT_TO_LOAD_PATH = None # IMPORTANT: SET THIS PATH!

# Backtesting data file
BACKTEST_DATA_FILE_PATH = "dummy_market_data_backtest.csv"

# Environment config for backtesting (should be consistent with training env features)
BACKTEST_ENV_CONFIG = {
    'initial_balance': 10000,
    'lookback_window_size': 10, 
    'features': ['Open', 'High', 'Low', 'Close', 'Volume']
}

# Agent config for initializing the agent structure before loading checkpoint
# Should match the core structure of the agent used for training (e.g., framework)
# The env name here will be specific to backtesting if we re-register.
BACKTEST_AGENT_CONFIG = {
    "framework": "torch",
    "env": "BacktestTradingEnv-v0", # Registered env for backtesting
    "num_workers": 0, # Not needed for inference/backtesting usually
    "num_gpus": 0,
    # Ensure this matches the model config used during training if custom model was used
    # "model": {"custom_model": "your_custom_model_name_if_any"}
}

def create_dummy_backtest_data(file_path, num_rows=252 * 2):
    if not os.path.exists(file_path):
        print(f"Creating dummy backtest data at: {file_path}")
        data = pd.DataFrame({
            'Timestamp': pd.to_datetime('2022-01-01') + pd.to_timedelta(np.arange(num_rows), 'D'),
            'Open': np.random.rand(num_rows) * 50 + 100,
            'High': np.random.rand(num_rows) * 10 + 125,
            'Low': np.random.rand(num_rows) * -10 + 75,
            'Close': np.random.rand(num_rows) * 50 + 100,
            'Volume': np.random.randint(10000, 50000, size=num_rows)
        })
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.rand(num_rows) * 5
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.rand(num_rows) * 5
        data.to_csv(file_path, index=False)
    else:
        print(f"Using existing backtest data from: {file_path}")

# Env creator for backtesting - uses the backtest_df
def backtest_env_creator(env_config_rllib):
    df_for_env = env_config_rllib.pop("df", None)
    if df_for_env is None:
        raise ValueError("DataFrame ('df') must be provided in env_config_rllib for backtest_env_creator")
    
    current_env_config = BACKTEST_ENV_CONFIG.copy()
    current_env_config.update(env_config_rllib)
    current_env_config['df'] = df_for_env
    return TradingEnvironment(**current_env_config)

if __name__ == '__main__':
    print("Initializing Ray for backtesting...")
    ray.init(ignore_reinit_error=True, log_to_driver=False) # Log to driver False for less noise

    # 1. Load Data for Backtesting
    create_dummy_backtest_data(BACKTEST_DATA_FILE_PATH)
    backtest_df = load_historical_data(BACKTEST_DATA_FILE_PATH, date_col='Timestamp',
                                       required_cols=BACKTEST_ENV_CONFIG['features'], dropna=True)

    if backtest_df is None or backtest_df.empty:
        print(f"Failed to load backtest data from {BACKTEST_DATA_FILE_PATH}. Exiting.")
        ray.shutdown()
        exit()
    
    if len(backtest_df) < BACKTEST_ENV_CONFIG['lookback_window_size'] + 20:
        print(f"Backtesting data is too short. Exiting.")
        ray.shutdown()
        exit()

    # 2. Register Env for Backtesting Agent Initialization
    # The agent needs an env to initialize its policy structure before loading checkpoint
    rllib_backtest_env_config = BACKTEST_AGENT_CONFIG.get("env_config", {}).copy()
    rllib_backtest_env_config["df"] = backtest_df 
    
    current_agent_config_for_load = BACKTEST_AGENT_CONFIG.copy()
    current_agent_config_for_load["env_config"] = rllib_backtest_env_config

    register_env("BacktestTradingEnv-v0", backtest_env_creator)
    print("Backtest environment 'BacktestTradingEnv-v0' registered.")

    # 3. Setup Agent and Load Checkpoint
    if CHECKPOINT_TO_LOAD_PATH is None or not os.path.exists(os.path.dirname(CHECKPOINT_TO_LOAD_PATH)):
        print("\n**************************************************************************************")
        print(f"CHECKPOINT_TO_LOAD_PATH is not set or directory does not exist: '{CHECKPOINT_TO_LOAD_PATH}")
        print("Please train an agent using 'train_agent.py' and then set the path to a valid checkpoint file.")
        print("Skipping backtest with trained agent. You can run with a DummyAgent for structure testing if needed.")
        print("**************************************************************************************\n")
        # As a fallback, could run with a DummyAgent like before, but goal is to test trained agent
        # current_agent_to_test = DummyAgent(action_space_n=3)
        # print("Using DummyAgent as CHECKPOINT_TO_LOAD_PATH is not set.")
        ray.shutdown()
        exit()
    else:
        print(f"Attempting to load trained agent from: {CHECKPOINT_TO_LOAD_PATH}")
        trained_agent = TradingAgent(env_name_or_creator="BacktestTradingEnv-v0", 
                                     agent_config=current_agent_config_for_load)
        
        if trained_agent.trainer:
            try:
                trained_agent.load_checkpoint(CHECKPOINT_TO_LOAD_PATH)
                print(f"Agent checkpoint loaded successfully from {CHECKPOINT_TO_LOAD_PATH}")
                current_agent_to_test = trained_agent
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Please ensure the checkpoint path and agent config are correct.")
                print("Exiting.")
                ray.shutdown()
                exit()
        else:
            print("Failed to initialize trainer in TradingAgent for loading checkpoint. Exiting.")
            ray.shutdown()
            exit()

    # 4. Run Backtest with the loaded (or dummy) agent
    print(f"Running backtest with agent: {type(current_agent_to_test).__name__}")
    try:
        analysis_results = run_backtest(
            data_df=backtest_df,
            agent_instance=current_agent_to_test, # Pass the agent *instance*
            env_config=BACKTEST_ENV_CONFIG, # For strategy params like lookback window
            initial_cash=10000.0,
            commission_rate=0.001
        )
    except Exception as e:
        print(f"Error during backtest run: {e}")
        import traceback
        traceback.print_exc()

    print("Shutting down Ray...")
    ray.shutdown()
    print("Backtesting script finished.") 