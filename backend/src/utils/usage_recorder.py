from abc import ABC, abstractmethod
import logging
import csv
import datetime
from pathlib import Path
import os

from typing import Optional, Dict, Union
from src.utils import Config

CSV_DIR = Path("logs")
DEFAULT_CSV_FILENAME = "llm_usage.csv"
CSV_HEADERS = [
    "timestamp",
    "model",
    "provider",
    "agent",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_seconds",
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
        provider: str,
        agent: str = "default",
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
    ):
        pass


class ConsoleUsageRecorder(UsageRecorder):

    def __init__(self):
        logger.info("Usage will be logged to the console")

    def record_activity(
        self,
        model: str,
        provider: str,
        agent: str = "default",
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
    ):
        logger.info({"model": model, "provider": provider, "token_usage": token_usage, "duration": duration})


class CSVUsageRecorder(UsageRecorder):

    def __init__(self):
        # Get the configured CSV filename, or use default if not set
        csv_filename = config.llm_usage_log_filename or DEFAULT_CSV_FILENAME
        self.csv_file_path = CSV_DIR / csv_filename

        logger.info(f"Usage logs will be saved to the following path: {self.csv_file_path}")

    def record_activity(
        self,
        model: str,
        provider: str,
        agent: str = "default",
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
    ) -> None:
        """
        Log LLM usage information to a CSV file.

        Args:
            model: The model name used for the request
            provider: The provider name used for the request
            agent: The name of the agent making the request
            token_usage: Dictionary containing token usage information
            duration: Time taken for the request in seconds
        """
        timestamp = datetime.datetime.now().isoformat()

        # Extract token information with fallback for missing data
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens", "N/A")
            completion_tokens = token_usage.get("completion_tokens", "N/A")
            total_tokens = token_usage.get("total_tokens", "N/A")
        else:
            prompt_tokens = "N/A"
            completion_tokens = "N/A"
            total_tokens = "N/A"

        # Create the file with headers if it doesn't exist
        file_exists = os.path.isfile(self.csv_file_path)

        with open(self.csv_file_path, mode="a", newline="") as file:
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
                    agent,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    f"{duration:.2f}",
                ]
            )

        logger.debug(f"Logged {model} / {provider} usage data to {self.csv_file_path}")
