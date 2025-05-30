import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, lookback_window_size=10, 
                 features=['Open', 'High', 'Low', 'Close', 'Volume']):
        super(TradingEnvironment, self).__init__()

        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert all(col in df.columns for col in features), \
            f"DataFrame must contain all specified features: {features}"
        assert 'Close' in features, "'Close' must be in features for price reference."

        self.df = df.copy()
        self.features = features
        self.num_features = len(features)
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space: 0 (Hold), 1 (Buy), 2 (Sell)
        self.action_space = spaces.Discrete(3)

        # Observation space: (lookback_window_size + 1) * num_features values
        # We add 1 to include the current step's data in the observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=( (self.lookback_window_size + 1) * self.num_features, ), 
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.net_worth = 0
        self.prev_net_worth = 0
        self.done = False

    def _get_observation(self):
        # Ensure there's enough data for the lookback window plus current step
        start_idx = max(0, self.current_step - self.lookback_window_size)
        end_idx = self.current_step + 1 # Inclusive of current step

        # Get relevant window of data
        frame = self.df[self.features].iloc[start_idx:end_idx].values

        # Pad with zeros if at the beginning of the dataset and frame is too short
        if frame.shape[0] < (self.lookback_window_size + 1):
            padding = np.zeros(((self.lookback_window_size + 1) - frame.shape[0], self.num_features))
            frame = np.vstack((padding, frame))
        
        return frame.flatten().astype(np.float32) # Flatten for RLlib compatibility

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        # Start at a point where a full lookback window is available
        self.current_step = self.lookback_window_size 
        self.done = False

        if self.current_step >= len(self.df) -1: # Not enough data to start
             # This case should ideally be handled by checking df length in __init__ or before creating env
             print("Warning: DataFrame too short for lookback window. Consider a larger df or smaller window.")
             # Return a zero observation and mark as done, or raise an error
             empty_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
             self.done = True
             return empty_obs

        return self._get_observation()

    def step(self, action):
        if self.done:
            # Should not happen if used correctly, but as a safeguard
            print("Episode already done. Call reset().")
            return self._get_observation(), 0, True, {}

        current_price = self.df['Close'].iloc[self.current_step]
        self.prev_net_worth = self.balance + self.shares_held * current_price

        # Execute action
        if action == 1: # Buy
            if self.balance >= current_price: # Can afford one share
                self.balance -= current_price
                self.shares_held += 1
        elif action == 2: # Sell
            if self.shares_held > 0:
                self.balance += current_price
                self.shares_held -= 1
        # Action 0 (Hold) means no change to balance or shares

        self.current_step += 1

        # Calculate new net worth and reward
        current_net_worth = self.balance + self.shares_held * self.df['Close'].iloc[self.current_step]
        reward = current_net_worth - self.prev_net_worth
        self.net_worth = current_net_worth

        # Check if episode is done
        if self.current_step >= len(self.df) - 1: # Reached end of data
            self.done = True
        # Optional: Add other done conditions, e.g., balance too low
        # if self.net_worth <= self.initial_balance / 2: 
        #     self.done = True

        obs = self._get_observation()
        info = {'current_price': current_price, 'net_worth': self.net_worth}

        return obs, reward, self.done, info

    def render(self, mode='human', close=False):
        # For now, just print current status
        if self.done:
            profit_percent = ((self.net_worth - self.initial_balance) / self.initial_balance) * 100
            print(f"Episode finished after {self.current_step - self.lookback_window_size} steps.")
            print(f"Final Net Worth: {self.net_worth:.2f}")
            print(f"Profit/Loss: {self.net_worth - self.initial_balance:.2f} ({profit_percent:.2f}%)")
        else:
            print(f"Step: {self.current_step - self.lookback_window_size}")
            print(f"Current Price: {self.df['Close'].iloc[self.current_step]:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Net Worth: {self.net_worth:.2f}")

    def close(self):
        pass # Implement if any resources need to be released


if __name__ == '__main__':
    # Example Usage (requires dummy data)
    # Create dummy data for demonstration
    data_size = 200
    dummy_data = pd.DataFrame({
        'Open': np.random.rand(data_size) * 100 + 100,
        'High': np.random.rand(data_size) * 10 + 100,
        'Low': np.random.rand(data_size) * -10 + 100,
        'Close': np.random.rand(data_size) * 100 + 100,
        'Volume': np.random.rand(data_size) * 10000 + 1000
    })
    # Ensure High is highest and Low is lowest
    dummy_data['High'] = dummy_data[['Open', 'Close']].max(axis=1) + np.random.rand(data_size) * 5
    dummy_data['Low'] = dummy_data[['Open', 'Close']].min(axis=1) - np.random.rand(data_size) * 5

    env_config = {
        'df': dummy_data,
        'initial_balance': 5000,
        'lookback_window_size': 5,
        'features': ['Open', 'High', 'Low', 'Close', 'Volume']
    }

    env = TradingEnvironment(**env_config)
    obs = env.reset()

    print("Trading Environment Initialized.")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print(f"Action Space Size: {env.action_space.n}")
    print(f"Initial Observation: {obs[:10]}... (first 10 values)")

    done = False
    total_reward_ep = 0
    episode_length = 0

    for _ in range(150): # Simulate a few steps
        action = env.action_space.sample() # Sample a random action
        obs, reward, done, info = env.step(action)
        total_reward_ep += reward
        episode_length +=1
        # env.render()
        if done:
            break
    
    print("\n--- Episode Finished ---")
    env.render() # Final render
    print(f"Total reward: {total_reward_ep:.2f}")
    print(f"Episode length: {episode_length}")

    # Example of how to register with Gym (if needed for RLlib string lookup)
    # from ray.tune.registry import register_env
    # def env_creator(env_config_dict):
    #     # You might need to load data or pass df in env_config_dict
    #     # For this example, assuming dummy_data is accessible or passed via env_config_dict
    #     # data = env_config_dict.get("df", dummy_data) 
    #     return TradingEnvironment(df=dummy_data, **env_config_dict) 
    # register_env("TradingEnv-v0", env_creator)
    # print("\nTradingEnv-v0 registered with Gym.") 