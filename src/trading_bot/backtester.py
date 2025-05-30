import backtrader as bt
import datetime
import pandas as pd

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


def run_backtest(data_df, agent_class, agent_config, env_config, 
                 initial_cash=10000.0, commission_rate=0.001, 
                 strategy_params=None):
    cerebro = bt.Cerebro()

    # Instantiate agent (assuming agent needs env_config or specific parts of it)
    # This part is tricky because the agent is trained on an *environment instance*
    # For backtesting, we pass the agent instance directly to the strategy
    # The agent is pre-trained.
    # For now, let's assume agent_class can be instantiated with just agent_config, or is already an instance
    # If agent_class is indeed a class:
    #  trading_agent = agent_class(env_name_or_creator=lambda cfg: TradingEnvironment(**cfg), agent_config=agent_config)
    # If agent_class is already an instance (more likely for a pre-trained agent):
    trading_agent = agent_class # if agent_class is actually an agent instance

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

if __name__ == '__main__':
    import numpy as np
    from .agent import TradingAgent # For the dummy agent
    from .environment import TradingEnvironment # For dummy env config reference

    # Create a dummy agent that takes random actions or simple logic
    class DummyAgent:
        def __init__(self, action_space_n):
            self.action_space_n = action_space_n
            self.trainer = True # Mock trainer attribute

        def predict(self, observation):
            # Simple logic: if close > open buy, if close < open sell, else hold
            # This requires observation to be structured in a way we can infer this.
            # For a generic dummy, let's just pick randomly for now.
            return np.random.randint(0, self.action_space_n)
        
        def get_policy(self): # Mock method
            return None

    # 1. Load Data
    # Create a dummy CSV for testing if utils.py was not run to create it
    dummy_csv_path = "dummy_market_data_backtest.csv"
    if not pd.io.common.file_exists(dummy_csv_path):
        num_rows = 252 * 2 # Approx 2 years of daily data
        test_data = pd.DataFrame({
            'Timestamp': pd.to_datetime('2022-01-01') + pd.to_timedelta(np.arange(num_rows), 'D'),
            'Open': np.random.rand(num_rows) * 50 + 100,
            'High': np.random.rand(num_rows) * 10 + 125,
            'Low': np.random.rand(num_rows) * -10 + 75,
            'Close': np.random.rand(num_rows) * 50 + 100,
            'Volume': np.random.randint(10000, 50000, size=num_rows)
        })
        test_data['High'] = test_data[['Open', 'Close']].max(axis=1) + np.random.rand(num_rows) * 5
        test_data['Low'] = test_data[['Open', 'Close']].min(axis=1) - np.random.rand(num_rows) * 5
        test_data.to_csv(dummy_csv_path, index=False)
        print(f"Created dummy CSV for backtesting: {dummy_csv_path}")

    df = load_historical_data(dummy_csv_path, date_col='Timestamp')

    if df is not None and not df.empty:
        # 2. Define Environment Configuration (consistent with agent training)
        env_config_params = {
            'df': df, # The actual df is not used by agent init, but for reference to params
            'initial_balance': 10000,
            'lookback_window_size': 10,
            'features': ['Open', 'High', 'Low', 'Close', 'Volume']
        }

        # 3. Setup Agent
        # For this example, we use a DummyAgent. In a real scenario, you'd load a trained TradingAgent.
        # The TradingAgent expects an env_name_or_creator for its internal RLlib trainer config.
        # However, for backtesting with a *pre-trained* agent, we pass the agent instance directly.
        dummy_action_space_size = 3 # Corresponds to Buy/Sell/Hold in our TradingEnvironment
        
        # For a real agent, you would load it, e.g.:
        # agent_config_ppo = { 'framework': 'torch', 'num_workers': 0 }
        # trained_agent = TradingAgent(env_name_or_creator=lambda cfg: TradingEnvironment(**cfg), agent_config=agent_config_ppo) # if defining here
        # trained_agent.load_checkpoint("path/to/your/checkpoint")
        # current_agent_to_test = trained_agent

        current_agent_to_test = DummyAgent(action_space_n=dummy_action_space_size)
        print("Using DummyAgent for this backtest run.")

        # Agent config (can be empty if agent is fully configured or pre-trained)
        agent_config_params = {}

        # 4. Run Backtest
        try:
            analysis_results = run_backtest(
                data_df=df,
                agent_class=current_agent_to_test, # Pass the agent *instance*
                agent_config=agent_config_params, 
                env_config=env_config_params, # Used for strategy params like lookback
                initial_cash=10000.0,
                commission_rate=0.001
            )
        except Exception as e:
            print(f"Error during backtest run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to load data, skipping backtest.")

    # Optional: Clean up dummy CSV
    # import os
    # if pd.io.common.file_exists(dummy_csv_path):
    #     os.remove(dummy_csv_path)
    #     print(f"Cleaned up {dummy_csv_path}") 