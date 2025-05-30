import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional, Union, Dict, Any
import datetime
import pandas as pd

# Initialize FastAPI app
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

# --- Global State / In-memory Storage (Illustrative) ---
# In a real application, you'd use a proper database or message queue.
latest_market_data: Dict[str, pd.DataFrame] = {}

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

# --- Placeholder for Agent Interaction (to be developed further) ---
# Example: Load a pre-trained agent (this is complex and needs careful thought on lifecycle)
# AGENT_INSTANCE = None 
# def load_trading_agent():
#     global AGENT_INSTANCE
#     if AGENT_INSTANCE is None:
#         print("Loading trading agent...")
#         # Logic to initialize Ray, register env, load agent config, load checkpoint
#         # This would be similar to the __main__ block of backtester.py
#         # Ensure Ray is initialized only once, or managed carefully.
#         # AGENT_INSTANCE = ... 
#         print("Trading agent loaded.")
# load_trading_agent() # Potentially load on startup, or on first request (with locking)

# @app.post("/predict/{symbol}", tags=["Trading Agent"], summary="Get trading prediction for a symbol")
# async def get_prediction(symbol: str, current_observation_data: Dict[str, Any]): # Define a Pydantic model for observation
#     if AGENT_INSTANCE is None:
#         raise HTTPException(status_code=503, detail="Agent not loaded or not ready.")
    
#     # 1. Preprocess current_observation_data into the format expected by the agent
#     #    This might involve fetching recent data, applying transformations, etc.
#     #    This observation must match the TradingEnvironment's observation space.
#     # observation_for_agent = ... 

#     # 2. Get prediction
#     # action = AGENT_INSTANCE.predict(observation_for_agent)
#     # return {"symbol": symbol, "action": action, "details": "..."}
#     return {"message": "Prediction endpoint not fully implemented yet."}

# --- Main execution for development ---
if __name__ == "__main__":
    print("Starting FastAPI server for Trading Bot API...")
    # Make sure uvicorn is installed: pip install uvicorn[standard]
    # Run with: python -m src.api.main or uvicorn src.api.main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=os.path.dirname(__file__)) 