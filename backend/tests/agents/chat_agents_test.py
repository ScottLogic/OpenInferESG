import os
import pytest
from pytest import raises
from unittest import mock

from src.agents import get_chat_agents

def test_get_chat_agents_returns_list(monkeypatch):
    # mocker.patch("src.utils.config.allowed_chat_agents", [])
    # with mock.patch.dict(os.environ, {"ALLOWED_CHAT_AGENTS": "WebAgent"}):
    # mock environment here

    agents = get_chat_agents()
    assert len(agents) == 1
