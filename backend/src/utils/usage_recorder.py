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
    "power_usage_wh",
    "cpu_time_seconds",
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
        agent: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        power_usage: Optional[float] = None,
        cpu_time: Optional[float] = None,
    ):
        pass


class ConsoleUsageRecorder(UsageRecorder):
    def __init__(self):
        logger.info("Usage will be logged to the console")

    def record_activity(
        self,
        model: str,
        provider: str,
        agent: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        power_usage: Optional[float] = None,
        cpu_time: Optional[float] = None,
    ):
        logger.info(
            {
                "model": model,
                "provider": provider,
                "agent": agent,
                "token_usage": token_usage,
                "duration": duration,
                "power_usage": power_usage,
                "cpu_time": cpu_time,
            }
        )


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
        agent: str,
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0,
        power_usage: Optional[float] = None,
        cpu_time: Optional[float] = None,
    ) -> None:
        """
        Log LLM usage information to a CSV file.

        Args:
            model: The model name used for the request
            provider: The provider name used for the request
            agent: The name of the agent making the request
            token_usage: Dictionary containing token usage information
            duration: Time taken for the request in seconds
            power_usage: Power usage of the request in watt-hours
        """
        # Use consistent timezone-aware timestamp format
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        power_usage_wh = f"{power_usage:.4f}" if power_usage is not None else "N/A"
        cpu_time_seconds = f"{cpu_time:.2f}" if cpu_time is not None else "N/A"

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
                    power_usage_wh,
                    cpu_time_seconds,
                ]
            )

        logger.debug(f"Logged {model} / {provider} usage data to {self.csv_file_path}")
