import pytest
import os
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

# Verify API key is set
if not os.getenv("GROQ_API_KEY"):
    pytest.skip("GROQ_API_KEY not set", allow_module_level=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()