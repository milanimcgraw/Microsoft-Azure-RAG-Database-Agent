# utils.py

import os
from dotenv import load_dotenv, find_dotenv

def load_env():
    """Load environment variables from a .env file."""
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    """Return the OpenAI API key."""
    load_env()
    return os.getenv("OPENAI_API_KEY")

def get_azure_openai_config():
    """Return a dictionary of Azure OpenAI credentials."""
    load_env()
    return {
        "api_key": os.getenv("AZURE_OPENAI_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-01-preview"),
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    }
