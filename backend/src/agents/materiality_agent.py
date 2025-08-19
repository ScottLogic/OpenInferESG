import json
from pathlib import Path
import logging

from src.agents.tool import ToolActionSuccess, ToolActionFailure, CommonParameters, Parameter
from src.llm import LLM
from src.agents import tool
from src.llm import LLMFile
from src.agents import chat_agent
from src.agents.base_chat_agent import BaseChatAgent
from src.prompts import PromptEngine

engine = PromptEngine()
logger = logging.getLogger(__name__)


def create_llm_files(filenames: list[str]) -> list[LLMFile]:
    return [
        LLMFile(filename=filename, file=Path(f"./library/{filename}"))
        for filename in filenames
    ]


async def select_material_files(user_question: str, llm: LLM, model) -> list[str]:
    with open('./library/catalogue.json') as file:
        catalogue = json.load(file)
        files_json = await llm.chat(
            model,
            system_prompt=engine.load_prompt(
                "select-material-files-system-prompt",
                catalogue=catalogue
            ),
            user_prompt=user_question,
            agent="materiality",
            return_json=True
        )
        return json.loads(files_json)["files"]


@tool(
    name="answer_materiality_question",
    description="This tool can answer questions about ESG Materiality about an industry or sector, including "
                "typical sector activities, value chain and business relationships.",
    parameters={
        **CommonParameters.USER_QUESTION,
        "sector": Parameter(
            type="string",
            description="The sector to ask materiality questions about."
        )
    }
)
async def answer_materiality_question(
    user_question: str, sector: str, llm: LLM, model
) -> ToolActionSuccess | ToolActionFailure:
    materiality_files = await select_material_files(user_question, llm, model)
    if materiality_files:
        answer = await llm.chat_with_file(
            model,
            system_prompt=engine.load_prompt("answer-materiality-question"),
            user_prompt=user_question,
            files=create_llm_files(materiality_files),
            agent="materiality"
        )
    else:
        return ToolActionFailure(
            f"Materiality Agent cannot find suitable reference documents to answer: {user_question}"
        )
    return ToolActionSuccess(answer)


@chat_agent(
    name="MaterialityAgent",
    description="The Materiality Agent can answer questions about specific industry wide ESG Materiality standards and "
                "reporting practices. The agent cannot provide information about companies themselves, only industries"
                "and sectors as a whole.",
    tools=[answer_materiality_question]
)
class MaterialityAgent(BaseChatAgent):
    async def list_material_topics_for_company(self, company_name: str) -> dict[str, str]:
        materiality_files = await select_material_files(company_name, self.llm, self.model)
        if not materiality_files:
            logger.info(f"No materiality reference documents could be found for {company_name}")
            return {}
        materiality_topics = await self.llm.chat_with_file(
            self.model,
            system_prompt=engine.load_prompt("list-material-topics-system-prompt"),
            user_prompt=f"What topics are material for {company_name}?",
            files=create_llm_files(materiality_files),
            agent="materiality",
            return_json=True
        )
        return json.loads(materiality_topics)["material_topics"]
