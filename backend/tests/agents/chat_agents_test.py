import pytest
from unittest.mock import patch

from src.agents import get_chat_agents, config

config.web_agent_llm = "openai"
config.web_agent_model = "web_agent_model"
config.file_agent_llm = "openai"
config.file_agent_model = "file_agent_model"
config.datastore_agent_llm = "openai"
config.datastore_agent_model = "datastore_agent_model"
config.materiality_agent_llm = "openai"
config.materiality_agent_model = "materiality_agent_model"

def test_get_chat_agents_given_config_returns_list():
    config.allowed_chat_agents = ["WebAgent", "FileAgent"]
    agents = get_chat_agents()
    assert len(agents) == 2
    assert any(agent.name == "WebAgent" for agent in agents)
    assert any(agent.name == "FileAgent" for agent in agents)

def test_get_chat_agents_returns_default_list_if_empty_env_var():
    config.allowed_chat_agents = []
    agents = get_chat_agents()
    assert len(agents) == 4
    assert agents[0].name == "DatastoreAgent"
    assert agents[1].name == "WebAgent"
    assert agents[2].name == "MaterialityAgent"
    assert agents[3].name == "FileAgent"

def test_get_chat_agents_returns_default_list_if_no_env_var():
    config.allowed_chat_agents = None
    agents = get_chat_agents()
    assert len(agents) == 4
    assert agents[0].name == "DatastoreAgent"
    assert agents[1].name == "WebAgent"
    assert agents[2].name == "MaterialityAgent"
    assert agents[3].name == "FileAgent"

def test_get_chat_agents_skips_unknown_agents():
    config.allowed_chat_agents = ["WebAgent", "UnknownAgent"]
    with patch('logging.Logger.warning') as mocked_logger:
        agents = get_chat_agents()
        assert len(agents) == 1
        assert agents[0].name == "WebAgent"
        mocked_logger.assert_called_once_with("Skipped invalid chat agents: unknownagent")

def test_get_chat_agents_removes_duplicates():
    config.allowed_chat_agents = ["WebAgent", "WebAgent", "FileAgent"]
    agents = get_chat_agents()
    assert len(agents) == 2
    assert any(agent.name == "WebAgent" for agent in agents)
    assert any(agent.name == "FileAgent" for agent in agents)

def test_get_chat_agents_errors_when_all_agents_skipped():
    config.allowed_chat_agents = ["UnknownAgent"]
    with pytest.raises(Exception, match="No valid chat agents found in ALLOWED_CHAT_AGENTS configuration."):
        get_chat_agents()
