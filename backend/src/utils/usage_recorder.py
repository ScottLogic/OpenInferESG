from abc import ABC, abstractmethod
import logging
import csv
import datetime
from pathlib import Path

from typing import Optional, Dict, Union
from src.utils import Config

CSV_DIR = Path("logs")
DEFAULT_CSV_FILENAME = "llm_usage.csv"
CSV_HEADERS = [
    "timestamp",
    "model",
    "provider",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_seconds",
    "request_type",
]

# Ensure the logs directory exists
CSV_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
config = Config()


class UsageRecorder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def record_activity(
        self,
        model: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        request_type: str = "chat",
    ):
        pass


class LogUsageRecorder(UsageRecorder):
    def record_activity(
        self,
        model: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        request_type: str = "chat",
    ):
        logger.info({"model": model, "token_usage": token_usage, "duration": duration, "request_type": request_type})


class CSVUsageRecorder(UsageRecorder):
    def record_activity(
        self,
        model: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        request_type: str = "chat",
    ) -> None:
        """
        Log LLM usage information to a CSV file.

        Args:
            model: The model name used for the request
            token_usage: Dictionary containing token usage information
            duration: Time taken for the request in seconds
            request_type: Type of request (chat, file-chat, etc.)
        """
        timestamp = datetime.datetime.now().isoformat()
        provider = self.__class__.__name__.lower()

        # Extract token information with fallbacks for missing data
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens", "N/A")
            completion_tokens = token_usage.get("completion_tokens", "N/A")
            total_tokens = token_usage.get("total_tokens", "N/A")
        else:
            prompt_tokens = "N/A"
            completion_tokens = "N/A"
            total_tokens = "N/A"

        # Get the configured CSV filename, or use default if not set
        csv_filename = config.llm_usage_log_filename or DEFAULT_CSV_FILENAME
        csv_file_path = CSV_DIR / csv_filename

        # Create the file with headers if it doesn't exist
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write headers if file is new
            if not file_exists:
                writer.writerow(CSV_HEADERS)

            # Write the data row
            writer.writerow(
                [
                    timestamp,
                    model,
                    provider,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    f"{duration:.2f}",
                    request_type,
                ]
            )

        logger.debug(f"Logged {provider} usage data to {csv_file_path}")
