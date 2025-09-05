import pytest
from dataclasses import dataclass
from pathlib import Path

from unittest.mock import patch, AsyncMock
from openai.types.beta.threads import Text, FileCitationAnnotation, TextContentBlock
from openai.types.beta.threads.file_citation_annotation import FileCitation

from src.llm import LLMFile
from src.llm.openai import OpenAI

from src.utils.usage_recorder import ConsoleUsageRecorder


@dataclass
class MockResponse:
    id: str

@dataclass
class MockUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class MockResponsesResponse:
    output_text: str
    usage: MockUsage

@dataclass
class MockFileResponse:
    id: str
    filename: str


@dataclass
class MockMessage:
    content: list[TextContentBlock]


@pytest.mark.asyncio
@patch("src.llm.openai.AsyncOpenAI")
@patch("src.llm.openai.OpenAILLMFileUploadManager.add_files_to_vector_store", new_callable=AsyncMock)
@patch("src.llm.openai.OpenAILLMFileUploadManager.upload_files", new_callable=AsyncMock)
async def test_chat_with_file_removes_citations(upload_files_method, add_files_to_vector_store_method, mock_async_openai):
    upload_files_method.return_value = AsyncMock(return_value=[MockResponse("file_id_1")])
    add_files_to_vector_store_method.return_value = AsyncMock(return_value=MockResponse("vector_store_id_1"))

    mock_instance = mock_async_openai.return_value

    mock_instance.responses.create = AsyncMock(return_value=MockResponsesResponse(
        output_text="Response with quote",
        usage=MockUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15
        )
    ))

    client = OpenAI(ConsoleUsageRecorder())
    response = await client.chat_with_file(
        model="",
        system_prompt="",
        user_prompt="",
        files=[LLMFile("filename", Path("./backend/library/AstraZeneca-Sustainability-Report-2023.pdf"))],
        agent="test-agent"
    )
    assert response == "Response with quote"
