import uvicorn
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, Field
from typing import List, Optional, Union, Dict, Any, Tuple
import datetime
import pandas as pd
import ray
from ray.tune.registry import register_env

# Assuming these are in the same project structure
from src.trading_bot.agent import TradingAgent
from src.trading_bot.environment import TradingEnvironment
from src.trading_bot.utils import load_historical_data # May not be needed directly here

# --- Configuration for API Agent Loading ---
# IMPORTANT: SET THIS PATH to a valid checkpoint created by train_agent.py
API_CHECKPOINT_TO_LOAD_PATH = None 
# Example: "./rllib_checkpoints/PPO_TradingEnv-v0_.../checkpoint_000010/checkpoint-10"

# Env config for agent initialization (observation/action space setup)
# The 'df' will be a dummy one for initialization purposes within the API.
API_ENV_CONFIG = {
    'initial_balance': 10000, # This value isn't critical for API prediction if not used by agent logic directly
    'lookback_window_size': 10, 
    'features': ['Open', 'High', 'Low', 'Close', 'Volume']
}

# Agent config for initializing the agent structure before loading checkpoint
API_AGENT_CONFIG = {
    "framework": "torch",
    "env": "ApiTradingEnv-v0", # Must match the name registered below
    "num_workers": 0, 
    "num_gpus": 0,
    "log_level": "WARNING", # Reduce verbosity for API
    # "model": { ... } # If a custom model was used during training, specify it here too
}

# Global variable to hold the loaded agent
AGENT_INSTANCE: Optional[TradingAgent] = None
RAY_INITIALIZED = False

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Trading Bot API",
    description="API for real-time market data ingestion and trading bot operations.",
    version="0.1.0"
)

# --- Pydantic Models for Data Ingestion ---
class MarketDataItem(BaseModel):
    timestamp: Union[datetime.datetime, datetime.date, str] # Flexible timestamp input
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Optional: add other features the bot might expect
    # feature1: Optional[float] = None 

class MarketDataIngestPayload(BaseModel):
    symbol: str
    data: List[MarketDataItem]
    metadata: Optional[Dict[str, Any]] = None

# Expects a list of [O, H, L, C, V] lists/tuples, 
# with length = lookback_window_size + 1
FeaturePoint = conlist(item_type=float, min_length=len(API_ENV_CONFIG['features']), max_length=len(API_ENV_CONFIG['features']))
class PredictionObservationInput(BaseModel):
    observation_window: conlist(item_type=FeaturePoint, 
                                 min_length=API_ENV_CONFIG['lookback_window_size'] + 1,
                                 max_length=API_ENV_CONFIG['lookback_window_size'] + 1)
    symbol: Optional[str] = None # Optional, context might be in observation

# --- Global State / In-memory Storage (Illustrative) ---
# In a real application, you'd use a proper database or message queue.
latest_market_data: Dict[str, pd.DataFrame] = {}

# --- RLlib Environment Creator for API ---
def api_env_creator(env_config_rllib):
    # Create a minimal dummy DataFrame just for environment initialization (obs/action space)
    # The actual data for prediction will come from the API request.
    num_features = len(API_ENV_CONFIG['features'])
    dummy_data_for_init = pd.DataFrame(
        np.zeros((API_ENV_CONFIG['lookback_window_size'] + 5, num_features)), # A bit more than lookback
        columns=API_ENV_CONFIG['features']
    )
    
    current_env_config = API_ENV_CONFIG.copy()
    current_env_config.update(env_config_rllib) # Overlay any RLlib-specific runtime configs
    current_env_config['df'] = dummy_data_for_init
    return TradingEnvironment(**current_env_config)

# --- Agent Loading Logic ---
async def load_trading_agent():
    global AGENT_INSTANCE, RAY_INITIALIZED
    if AGENT_INSTANCE is not None:
        print("Trading agent already loaded.")
        return

    if API_CHECKPOINT_TO_LOAD_PATH is None or not (os.path.exists(API_CHECKPOINT_TO_LOAD_PATH) and os.path.isfile(API_CHECKPOINT_TO_LOAD_PATH)):
        print("**************************************************************************************")
        print(f"API_CHECKPOINT_TO_LOAD_PATH is not set or is not a valid file: '{API_CHECKPOINT_TO_LOAD_PATH}'")
        print("API will run without a loaded agent. /predict endpoint will not function.")
        print("Please train an agent and set API_CHECKPOINT_TO_LOAD_PATH in src/api/main.py to the specific checkpoint FILE (not directory).")
        print("**************************************************************************************")
        return

    print("Initializing Ray for API...")
    if not RAY_INITIALIZED:
        ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
        RAY_INITIALIZED = True
    
    print(f"Registering API environment 'ApiTradingEnv-v0'...")
    register_env("ApiTradingEnv-v0", api_env_creator)

    # The agent needs an env_config with a 'df' for the creator, even if it's dummy for init
    rllib_api_env_config = API_AGENT_CONFIG.get("env_config", {}).copy()
    # No need to pass a real df here, api_env_creator handles the dummy df

    current_agent_config_for_load = API_AGENT_CONFIG.copy()
    current_agent_config_for_load["env_config"] = rllib_api_env_config

    print(f"Attempting to load agent from checkpoint: {API_CHECKPOINT_TO_LOAD_PATH}")
    try:
        agent_to_load = TradingAgent(env_name_or_creator="ApiTradingEnv-v0", 
                                     agent_config=current_agent_config_for_load)
        if agent_to_load.trainer:
            agent_to_load.load_checkpoint(API_CHECKPOINT_TO_LOAD_PATH)
            AGENT_INSTANCE = agent_to_load
            print("Trading agent loaded successfully into API.")
        else:
            print("Failed to initialize trainer in TradingAgent for API loading.")
    except Exception as e:
        print(f"Error loading trading agent in API: {e}")
        import traceback
        traceback.print_exc()

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI application startup...")
    await load_trading_agent() # Load the agent

@app.on_event("shutdown")
async def shutdown_event():
    global RAY_INITIALIZED
    print("FastAPI application shutdown...")
    if RAY_INITIALIZED:
        ray.shutdown()
        RAY_INITIALIZED = False
        print("Ray shut down.")

# --- API Endpoints ---

@app.get("/", tags=["General"], summary="Health check endpoint")
async def read_root():
    """Provides a simple health check message."""
    return {"message": "Trading Bot API is running."}

@app.post("/data/ingest", tags=["Data Ingestion"], summary="Ingest market data for a symbol")
async def ingest_market_data(payload: MarketDataIngestPayload):
    """
    Receives market data for a specific symbol.
    
    - **symbol**: The trading symbol (e.g., 'BTC/USD').
    - **data**: A list of market data points (OHLCV + timestamp).
    - **metadata**: Optional dictionary for additional info.
    """
    print(f"Received data for symbol: {payload.symbol}")
    print(f"Number of data points: {len(payload.data)}")
    if payload.metadata:
        print(f"Metadata: {payload.metadata}")

    # Convert to DataFrame for potential further processing or storage
    try:
        data_list = [item.model_dump() for item in payload.data]
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Store or process the DataFrame (example: update in-memory store)
        latest_market_data[payload.symbol] = df
        print(f"Successfully processed and stored data for {payload.symbol}. DataFrame shape: {df.shape}")
        
    except Exception as e:
        print(f"Error processing ingested data for {payload.symbol}: {e}")
        # In a real app, you might not want to raise HTTPException for internal processing errors
        # if it doesn't affect the client's request directly, but log it.
        # For now, we'll just print and return success as data was received.
        # raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

    return {
        "message": f"Data for symbol '{payload.symbol}' received successfully.",
        "items_received": len(payload.data)
    }

@app.get("/data/latest/{symbol}", tags=["Data Retrieval"], summary="Get latest ingested data for a symbol (illustrative)")
async def get_latest_data(symbol: str):
    """
    Retrieves the latest ingested (and processed into DataFrame) data for a symbol.
    This is an illustrative endpoint; in a real system, data access would be more robust.
    """
    if symbol in latest_market_data:
        # Pydantic can't directly serialize a DataFrame with DatetimeIndex easily for JSON response.
        # Convert to a more JSON-friendly format (e.g., list of dicts with string timestamp)
        df_to_return = latest_market_data[symbol].reset_index()
        df_to_return['timestamp'] = df_to_return['timestamp'].astype(str)
        return {"symbol": symbol, "data": df_to_return.to_dict(orient="records")}
    else:
        raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")

@app.post("/predict", tags=["Trading Agent"], summary="Get trading prediction based on observation window")
async def get_prediction(input_data: PredictionObservationInput):
    global AGENT_INSTANCE
    if AGENT_INSTANCE is None or AGENT_INSTANCE.trainer is None:
        raise HTTPException(status_code=503, detail="Agent not loaded or not ready. Check API_CHECKPOINT_TO_LOAD_PATH.")

    # 1. Convert input_data.observation_window to a flat numpy array
    #    The structure of input_data.observation_window is List[List[float]]
    #    where inner list has features in order: Open, High, Low, Close, Volume
    #    and outer list has length lookback_window_size + 1
    try:
        # Convert list of lists to a 2D numpy array
        observation_array_2d = np.array(input_data.observation_window, dtype=np.float32)
        
        # Expected shape: (lookback_window_size + 1, num_features)
        expected_rows = API_ENV_CONFIG['lookback_window_size'] + 1
        expected_cols = len(API_ENV_CONFIG['features'])

        if observation_array_2d.shape != (expected_rows, expected_cols):
            raise ValueError(f"Observation data shape mismatch. Expected ({expected_rows}, {expected_cols}), got {observation_array_2d.shape}")

        # Flatten the array to 1D for the agent's predict method
        observation_for_agent = observation_array_2d.flatten()
        # print(f"Observation for agent (shape {observation_for_agent.shape}): {observation_for_agent[:10]}...")

    except Exception as e:
        print(f"Error processing input observation data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid observation data format: {str(e)}")

    # 2. Get prediction from the agent
    try:
        action = AGENT_INSTANCE.predict(observation_for_agent)
        # RLlib PPO typically returns a single action value (e.g., int for Discrete space)
        # Ensure it's JSON serializable
        if isinstance(action, np.generic):
            action = action.item() # Convert numpy type to Python native type
            
        return {
            "symbol": input_data.symbol or "N/A", 
            "predicted_action": action, 
            "message": "Prediction successful"
        }
    except Exception as e:
        print(f"Error during agent prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during agent prediction: {str(e)}")

# --- Main execution for development ---
if __name__ == "__main__":
    print("Starting FastAPI server for Trading Bot API...")
    # Make sure API_CHECKPOINT_TO_LOAD_PATH is set correctly if you want to test /predict
    if API_CHECKPOINT_TO_LOAD_PATH is None:
        print("WARNING: API_CHECKPOINT_TO_LOAD_PATH is not set in src/api/main.py.")
        print("The /predict endpoint will not function without a trained agent checkpoint.")
    
    # Determine the directory of the current file to correctly set app_dir for uvicorn
    # This allows running 'python src/api/main.py' from the project root
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=current_file_dir) 