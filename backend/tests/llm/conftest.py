"""
Test configuration for LLM tests
"""

import pytest
import sys
from unittest.mock import MagicMock

# Set up mocks for any external modules needed for testing
@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """
    Set up the test environment for all tests
    This runs once before any tests are executed
    """
    # Import our mock modules - these should be imported by mock_imports.py
    from tests.llm.mock_imports import MOCK_MODULES
    
    # Set up any additional test configuration here
    yield
    
    # Clean up after all tests are done
    # Remove any mocks we added to avoid affecting other tests
    for mod_name in MOCK_MODULES:
        if mod_name in sys.modules:
            sys.modules.pop(mod_name, None)
