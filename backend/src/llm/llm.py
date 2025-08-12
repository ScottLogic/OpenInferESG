from abc import ABC, ABCMeta, abstractmethod
import csv
import datetime
import logging
import os
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Coroutine, Dict, Optional, Union

from src.utils import Config
from .count_calls import count_calls


count_calls_of_functions = ["chat", "chat_with_file"]

# Default CSV settings
CSV_DIR = Path("logs")
DEFAULT_CSV_FILENAME = "llm_usage.csv"
CSV_HEADERS = ["timestamp", "model", "provider", "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds", "request_type"]

# Ensure the logs directory exists
CSV_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
config = Config()


@dataclass
class LLMFile(ABC):
    filename: str
    file: PathLike[str] | bytes


class LLMMeta(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, "instances"):
            cls.instances = {}

        cls.instances[name.lower()] = cls()

    def __new__(cls, name, bases, attrs):
        for function in count_calls_of_functions:
            if function in attrs:
                attrs[function] = count_calls(attrs[function])

        return super().__new__(cls, name, bases, attrs)


class LLM(ABC, metaclass=LLMMeta):
    @classmethod
    def get_instances(cls):
        return cls.instances
        
    def log_usage_to_csv(
        self, 
        model: str, 
        token_usage: Optional[Union[Dict, str]] = None, 
        duration: float = 0.0, 
        request_type: str = "chat"
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
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(CSV_HEADERS)
            
            # Write the data row
            writer.writerow([
                timestamp,
                model,
                provider,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                f"{duration:.2f}",
                request_type
            ])
        
        logger.debug(f"Logged {provider} usage data to {csv_file_path}")

    @abstractmethod
    def chat(
        self, model: str, system_prompt: str, user_prompt: str, return_json: bool = False
    ) -> Coroutine[Any, Any, str]:
        pass

    @abstractmethod
    def chat_with_file(
        self, model: str, system_prompt: str, user_prompt: str, files: list[LLMFile], return_json: bool = False
    ) -> Coroutine:
        pass


class LLMFileUploadManager(ABC):
    @abstractmethod
    async def upload_files(self, files: list[LLMFile]) -> list[str]:
        pass

    @abstractmethod
    async def delete_all_files(self):
        pass
