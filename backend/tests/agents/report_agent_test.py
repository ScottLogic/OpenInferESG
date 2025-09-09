import pytest
from unittest.mock import AsyncMock, MagicMock

import src.agents.report_agent as report_agent_module

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_with_file = AsyncMock()
    llm.chat = AsyncMock()
    return llm

@pytest.fixture
def mock_engine(monkeypatch):
    engine = MagicMock()
    engine.load_prompt = MagicMock(side_effect=lambda name, **kwargs: f"prompt-{name}")
    engine.load_template = MagicMock(side_effect=lambda template_name, **kwargs: f"template-{template_name}-{kwargs}")
    monkeypatch.setattr(report_agent_module, "engine", engine)
    return engine

@pytest.fixture
def mock_questions(monkeypatch):
    mock_questions = {
        "Environment": [
            {"report_heading": "Env Heading 1", "prompt": "Env Q1"},
            {"report_heading": "Env Heading 2", "prompt": "Env Q2"},
        ],
        "Social": [
            {"report_heading": "Soc Heading 1", "prompt": "Soc Q1"},
        ]
    }
    monkeypatch.setattr(report_agent_module, "QUESTIONS", mock_questions)
    return mock_questions

@pytest.fixture
def agent(mock_llm, mock_engine, mock_questions):
    class DummyAgent(report_agent_module.ReportAgent):
        def __init__(self):
            self.llm = mock_llm
            self.model = "test-model"
    return DummyAgent()

@pytest.mark.asyncio
async def test_get_company_name(agent):
    agent.llm.chat_with_file.return_value = "Test Company"
    file = MagicMock()
    result = await agent.get_company_name(file)
    assert result == "Test Company"
    agent.llm.chat_with_file.assert_awaited_once()

@pytest.mark.asyncio
async def test_create_report_synchronous(agent, mock_engine, mock_questions):
    agent.llm.chat_with_file.side_effect = [
        "Overview text",
        "Env Answer 1",
        "Env Answer 2",
        "Soc Answer 1",
        "Materiality text"
    ]
    agent.llm.chat.return_value = "Conclusion text"
    file = MagicMock()
    materiality_topics = {"topic1": "desc1"}

    result = await agent.create_report_synchronous(file, materiality_topics)
    assert "Overview text" in result
    assert "Env Answer 1" in result
    assert "Env Answer 2" in result
    assert "Soc Answer 1" in result
    assert "Materiality text" in result
    assert "Conclusion text" in result
    assert "report-template" in result

@pytest.mark.asyncio
async def test_create_report_synchronous_no_materiality(agent, mock_engine):
    agent.llm.chat_with_file.side_effect = [
        "Overview text",
        "Env Answer 1",
        "Env Answer 2",
        "Soc Answer 1",
        "Materiality text"
    ]
    agent.llm.chat.return_value = "Conclusion text"
    file = MagicMock()
    materiality_topics = {}

    await agent.create_report_synchronous(file, materiality_topics)
    assert "No Materiality topics identified." in mock_engine.load_prompt.call_args_list[5][1]['materiality']
