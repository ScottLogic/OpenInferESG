import logging
from typing import List

from src.utils import Config
from src.agents.agent import Agent, ChatAgent, chat_agent
from src.agents.datastore_agent import DatastoreAgent
from src.agents.web_agent import WebAgent
from src.agents.intent_agent import IntentAgent
from src.agents.tool import tool, Parameter
from src.agents.validator_agent import ValidatorAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.file_agent import FileAgent
from src.agents.report_agent import ReportAgent
from src.agents.materiality_agent import MaterialityAgent
from src.agents.generalist_agent import GeneralistAgent


config = Config()
logger = logging.getLogger(__name__)


def get_validator_agent() -> ValidatorAgent:
    return ValidatorAgent(config.validator_agent_llm, config.validator_agent_model)


def get_intent_agent() -> IntentAgent:
    return IntentAgent(config.intent_agent_llm, config.intent_agent_model)


def get_answer_agent() -> AnswerAgent:
    return AnswerAgent(config.answer_agent_llm, config.answer_agent_model)


def get_report_agent() -> ReportAgent:
    return ReportAgent(config.report_agent_llm, config.report_agent_model)


def get_materiality_agent() -> MaterialityAgent:
    return MaterialityAgent(config.materiality_agent_llm, config.materiality_agent_model)


def get_generalist_agent() -> GeneralistAgent:
    return GeneralistAgent(config.generalist_agent_llm, config.generalist_agent_model)


def get_chat_agents() -> List[ChatAgent]:
    allowed_agents = config.allowed_chat_agents
    if allowed_agents:
        allowed_agents = set(a.lower() for a in allowed_agents)
        agents = []
        skipped_agents = []
        for agent_name in allowed_agents:
            match agent_name:
                case "datastoreagent":
                    agents.append(DatastoreAgent(config.datastore_agent_llm, config.datastore_agent_model))
                case "webagent":
                    agents.append(WebAgent(config.web_agent_llm, config.web_agent_model))
                case "materialityagent":
                    agents.append(MaterialityAgent(config.materiality_agent_llm, config.materiality_agent_model))
                case "fileagent":
                    agents.append(FileAgent(config.file_agent_llm, config.file_agent_model))
                case _:
                    skipped_agents.append(agent_name)
        if skipped_agents:
            logger.warning(f"Skipped invalid chat agents: {', '.join(skipped_agents)}")
        if not agents:
            raise Exception("No valid chat agents found in ALLOWED_CHAT_AGENTS configuration.")
        else:
            logger.info(f"Using allowed chat agents: {', '.join([agent.__class__.__name__ for agent in agents])}")
            return agents
    else:
        # Default agents if no specific agents are allowed
        return [
            DatastoreAgent(config.datastore_agent_llm, config.datastore_agent_model),
            WebAgent(config.web_agent_llm, config.web_agent_model),
            MaterialityAgent(config.materiality_agent_llm, config.materiality_agent_model),
            FileAgent(config.file_agent_llm, config.file_agent_model)
        ]


__all__ = [
    "Agent",
    "ChatAgent",
    "chat_agent",
    "get_chat_agents",
    "get_answer_agent",
    "get_intent_agent",
    "get_validator_agent",
    "get_report_agent",
    "get_materiality_agent",
    "get_generalist_agent",
    "Parameter",
    "tool"
]
