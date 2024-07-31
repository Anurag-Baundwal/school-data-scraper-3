# config.py for scraping player info from college softball rosters

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS').split(',')
OXYLABS_USERNAME = os.getenv('OXYLABS_USERNAME')
OXYLABS_PASSWORD = os.getenv('OXYLABS_PASSWORD')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')