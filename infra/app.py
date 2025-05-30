#!/usr/bin/env python3
import aws_cdk as cdk

from trading_bot_stack import TradingBotStack # Assuming our stack file is trading_bot_stack.py

app = cdk.App()

# You can specify your AWS account and region here or rely on environment/profile configuration
# For example:
# env_usa = cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION'))
# TradingBotStack(app, "TradingBotStack", env=env_usa)

TradingBotStack(app, "TradingBotStack")

app.synth() 