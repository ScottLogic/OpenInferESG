"""
Mock imports to help with testing by avoiding real dependencies
"""

import sys
from unittest.mock import MagicMock

# Create mocks for modules that might be problematic
MOCK_MODULES = ['fastapi', 'aiohttp']

# Add any classes or functions that need to be available from these modules
class MockHTTPException(Exception):
    """Mock HTTPException for testing"""
    def __init__(self, status_code=500, detail="Test error"):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")

# Create mocks for each module
for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        mock = MagicMock()
        # Add any specific attributes or functions needed
        if mod_name == 'fastapi':
            mock.HTTPException = MockHTTPException
        sys.modules[mod_name] = mock
