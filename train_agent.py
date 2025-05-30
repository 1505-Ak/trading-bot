import os
import pandas as pd
import numpy as np
import ray
from ray.tune.registry import register_env

from src.trading_bot.agent import TradingAgent
from src.trading_bot.environment import TradingEnvironment
from src.trading_bot.utils import load_historical_data

# Configuration
DATA_FILE_PATH = "market_data_train.csv" # Path to your training data CSV
CHECKPOINT_DIR = "./rllib_checkpoints"   # Directory to save training checkpoints
N_TRAINING_ITERATIONS = 10             # Number of training iterations
SAVE_CHECKPOINT_FREQ = 5               # How often to save a checkpoint (in iterations)

# Environment Configuration - should match or be consistent with agent's needs
ENV_CONFIG = {
    # 'df': dataframe will be loaded and passed directly to env_creator
    'initial_balance': 10000,
    'lookback_window_size': 10, # Must be consistent with agent's model if it has fixed input size
    'features': ['Open', 'High', 'Low', 'Close', 'Volume']
}

# Agent (PPO) Configuration
# See RLlib PPO docs for more options: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
AGENT_CONFIG = {
    "framework": "torch",        # or "tf"
    "env": "TradingEnv-v0",    # Must match the name registered with register_env
    # "model": {
    #     "custom_model": "my_custom_model", # If using a custom model from model.py
    #     "custom_model_config": {},
    # },
    "num_workers": 1,           # Number of rollout workers for parallel sampling
                                # Set to 0 for debugging on a single thread
    "num_gpus": 0,              # Number of GPUs to assign to the trainer process (driver)
    # "_disable_preprocessor_api": True, # If your observation space is already preprocessed
    "log_level": "INFO",         # "DEBUG", "INFO", "WARN", "ERROR"
    # Example PPO specific hyperparams (can be tuned)
    "lr": 5e-5,                 # Learning rate
    "gamma": 0.99,              # Discount factor
    "lambda": 0.95,             # GAE parameter
    "clip_param": 0.2,          # PPO clipping parameter
    "kl_coeff": 0.2,            # KL divergence coefficient
    "train_batch_size": 4000,   # Total batch size collected from workers
    "sgd_minibatch_size": 128,  # Size of minibatches for SGD
    "num_sgd_iter": 10,         # Number of SGD epochs per training iteration
    "rollout_fragment_length": 200, # Size of fragments sent from workers to driver
    "no_done_at_end": False,     # If episodes should not be done at the end of a rollout fragment
                                 # Set to True if your env is continuous and doesn't have natural end.
}

def create_dummy_training_data(file_path, num_rows=500):
    if not os.path.exists(file_path):
        print(f"Creating dummy training data at: {file_path}")
        data = pd.DataFrame({
            'Timestamp': pd.to_datetime('2021-01-01') + pd.to_timedelta(np.arange(num_rows), 'D'),
            'Open': np.random.rand(num_rows) * 100 + 100,
            'High': np.random.rand(num_rows) * 10 + 105,
            'Low': np.random.rand(num_rows) * -10 + 95,
            'Close': np.random.rand(num_rows) * 100 + 100,
            'Volume': np.random.randint(1000, 5000, size=num_rows)
        })
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.rand(num_rows) * 2
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.rand(num_rows) * 2
        # Add some NaNs for testing missing value handling in load_historical_data
        for col in ['Open', 'Close', 'Volume']:
            idx_to_nan = np.random.choice(data.index, size=num_rows // 20, replace=False)
            data.loc[idx_to_nan, col] = np.nan
        data.to_csv(file_path, index=False)
    else:
        print(f"Using existing training data from: {file_path}")

# RLlib Environment Creator Function
# RLlib will call this function to create instances of your environment.
# The `env_config_rllib` will be a dictionary passed from the agent's config.
def env_creator(env_config_rllib):
    # env_config_rllib typically contains the 'df' and other parameters
    # We are loading the DataFrame once globally and then passing it through here
    # Ensure the DataFrame used for training is passed correctly into the environment.
    df_for_env = env_config_rllib.pop("df", None) # Extract df, remove from rllib config
    if df_for_env is None:
        raise ValueError("DataFrame ('df') must be provided in env_config_rllib for env_creator")
    
    # Combine with the global ENV_CONFIG, but specific df passed by RLlib takes precedence
    current_env_config = ENV_CONFIG.copy() # Start with global defaults
    current_env_config.update(env_config_rllib) # Overlay any RLlib-specific runtime configs
    current_env_config['df'] = df_for_env # Ensure the correct df is used
    
    return TradingEnvironment(**current_env_config) 


def main():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=True)

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load and Prepare Data
    create_dummy_training_data(DATA_FILE_PATH) # Create if not exists
    training_df = load_historical_data(DATA_FILE_PATH, date_col='Timestamp',
                                         required_cols=ENV_CONFIG['features'], dropna=True)
    if training_df is None or training_df.empty:
        print(f"Failed to load training data from {DATA_FILE_PATH}. Exiting.")
        ray.shutdown()
        return
    
    if len(training_df) < ENV_CONFIG['lookback_window_size'] + 20: # Arbitrary small number for min length
        print(f"Training data is too short (length: {len(training_df)}) for the lookback window and meaningful training. Exiting.")
        ray.shutdown()
        return

    # 2. Register Custom Environment with RLlib
    # Pass the loaded df into the env_config that RLlib will use when creating env instances
    # This makes the specific DataFrame available to each environment worker created by RLlib.
    # We add it to a copy of AGENT_CONFIG's env_config or create one if not present.
    rllib_env_config = AGENT_CONFIG.get("env_config", {}).copy()
    rllib_env_config["df"] = training_df # Pass the loaded DataFrame here
    
    current_agent_config = AGENT_CONFIG.copy()
    current_agent_config["env_config"] = rllib_env_config

    register_env("TradingEnv-v0", env_creator) 
    print(f"Environment 'TradingEnv-v0' registered with RLlib.")

    # 3. Initialize Trading Agent
    print("Initializing Trading Agent (PPO)...")
    # The TradingAgent expects the env name or creator, and the merged agent config
    agent = TradingAgent(env_name_or_creator="TradingEnv-v0", agent_config=current_agent_config)

    if not agent.trainer: # Check if PPOTrainer initialized successfully
        print("Failed to initialize PPO trainer in TradingAgent. Exiting.")
        ray.shutdown()
        return

    print(f"Agent initialized. Using PPO with framework: {agent.trainer.framework}")
    print(f"Training on data with {len(training_df)} steps.")

    # 4. Training Loop
    print("Starting training loop...")
    for i in range(N_TRAINING_ITERATIONS):
        print(f"--- Iteration {i + 1}/{N_TRAINING_ITERATIONS} ---")
        try:
            result = agent.train() # This calls trainer.train()
            if result:
                print(f"  Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
                print(f"  Episode length mean: {result.get('episode_len_mean', 'N/A')}")
                # Add more metrics from result if needed
                # print(result)
            else:
                print("  Training iteration returned no result.")

            if (i + 1) % SAVE_CHECKPOINT_FREQ == 0 or (i + 1) == N_TRAINING_ITERATIONS:
                checkpoint_path = agent.save_checkpoint(CHECKPOINT_DIR)
                if checkpoint_path:
                    print(f"  Checkpoint saved at: {checkpoint_path}")
                else:
                    print("  Failed to save checkpoint.")
        except Exception as e:
            print(f"Error during training iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Optionally, decide if you want to break or continue on error
            # break 

    print("\nTraining finished.")

    # 5. Save Final Model (optional, as checkpoints are saved)
    final_checkpoint_path = agent.save_checkpoint(CHECKPOINT_DIR) # Save one last time
    if final_checkpoint_path:
        print(f"Final agent checkpoint saved at: {final_checkpoint_path}")
    else:
        print("Failed to save final agent checkpoint.")

    # 6. Shutdown Ray
    print("Shutting down Ray...")
    ray.shutdown()
    print("Done.")

if __name__ == "__main__":
    main() 