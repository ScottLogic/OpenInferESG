from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, Coroutine, Dict, Optional, Union

from src.utils.usage_recorder import UsageRecorder, CSVUsageRecorder

from .count_calls import count_calls


count_calls_of_functions = ["chat", "chat_with_file"]

# Default CSV settings


@dataclass
class LLMFile(ABC):
    filename: str
    file: PathLike[str] | bytes


class LLMMeta(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, "instances"):
            cls.instances = {}

        cls.instances[name.lower()] = cls(CSVUsageRecorder())

    def __new__(cls, name, bases, attrs):
        for function in count_calls_of_functions:
            if function in attrs:
                attrs[function] = count_calls(attrs[function])

        return super().__new__(cls, name, bases, attrs)


class LLM(ABC, metaclass=LLMMeta):
    def __init__(self, usage_recorder: UsageRecorder):
        self.usage_recorder = usage_recorder

    @classmethod
    def get_instances(cls):
        return cls.instances

    def log_usage_to_csv(
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
        self.usage_recorder.record_activity(model, token_usage, duration, request_type)

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
