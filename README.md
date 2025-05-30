# Reinforcement Learning Algorithmic Trading Bot

This project implements a reinforcement learning agent for algorithmic trading. 
The agent is built using Python, TensorFlow, PyTorch, and RLlib for the core RL 
capabilities. Backtrader is used for backtesting trading strategies. 
A RESTful API is provided using FastAPI for real-time market data 
ingestion and interaction with the trading bot. The system is designed for 
deployment on AWS, ensuring high availability and fault tolerance.

## Project Goals

- Develop an RL trading agent capable of outperforming baseline strategies.
- Achieve a simulated ROI of at least 12.4%.
- Evaluate agent robustness through 5-year backtesting.
- Deploy the trading bot on AWS with 99.9% uptime.
- Enable real-time market data ingestion via RESTful APIs.

## Technologies

- Python
- TensorFlow
- PyTorch
- RLlib
- Backtrader
- FastAPI
- Docker
- AWS (EC2, S3, CloudWatch, etc.)
