#!/usr/bin/env python3
# Sample run script for Alpaca bot

import os
from final_production_bot import FinalProductionBot

# Option 1: Use environment variables
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

# Option 2: Direct input (replace with your keys)
# api_key = "your_paper_api_key_here"
# secret_key = "your_paper_secret_key_here"

# Initialize and run bot
bot = FinalProductionBot(api_key=api_key, secret_key=secret_key, paper=True)
bot.execute_final_trades()
