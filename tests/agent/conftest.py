import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_environment():
    """Automatically load .env variables for all tests under tests/agent/"""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n\033[93mWARNING: OPENAI_API_KEY is not set in .env file.\033[0m")
        print("Some functional tests will be skipped.\n")
    else:
        print("\n\033[92m✓ OPENAI_API_KEY loaded successfully.\033[0m")
