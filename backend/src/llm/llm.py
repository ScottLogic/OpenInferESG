from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, Coroutine, Dict, Optional, Union

from src.utils.usage_recorder import UsageRecorder, CSVUsageRecorder

from .count_calls import count_calls


count_calls_of_functions = ["chat", "chat_with_file"]


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

    def record_usage(
        self,
        model: str,
        provider: str,
        agent: str = "default",
        token_usage: Optional[Union[Dict, str]] = None,
        duration: float = 0.0
    ) -> None:
        """
        Record usage information

        Args:
            model: The model name used for the request
            provider: The provider name used for the request
            agent: The name of the agent making the call
            token_usage: Dictionary containing token usage information
            duration: Time taken for the request in seconds
        """
        self.usage_recorder.record_activity(model, provider, agent, token_usage, duration)

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
