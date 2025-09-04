import asyncio
import logging
import json
from aiohttp import ClientTimeout

from src.llm.llm import LLMFile
from src.agents import Agent
from src.prompts import PromptEngine
from src.agents.report_questions import QUESTIONS

logger = logging.getLogger(__name__)
engine = PromptEngine()


class ReportAgent(Agent):
    async def create_report(self, file: LLMFile, materiality_topics: dict[str, str]) -> str:
        materiality = materiality_topics if materiality_topics else "No Materiality topics identified."

        async with asyncio.TaskGroup() as tg:
            overview = tg.create_task(
                self.llm.chat_with_file(
                    self.model,
                    system_prompt=engine.load_prompt("create-report-overview"),
                    user_prompt="Generate an ESG report about the attached document.",
                    files=[file],
                    agent="report"
                ),
            )

            categorized_tasks = {
                category: [
                    {
                        "report_heading": question["report_heading"],
                        "task": tg.create_task(
                            self.llm.chat_with_file(
                                self.model,
                                system_prompt=engine.load_prompt("report-question-system-prompt"),
                                user_prompt=question["prompt"],
                                files=[file],
                                agent="report"
                            ),
                        ),
                    }
                    for question in QUESTIONS[category]
                ]
                for category in QUESTIONS.keys()
            }

            materiality = tg.create_task(
                self.llm.chat_with_file(
                    self.model,
                    system_prompt=engine.load_prompt("create-report-materiality"),
                    user_prompt=engine.load_prompt("create-report-materiality-user-prompt", materiality=materiality),
                    files=[file],
                    agent="report"
                ),
            )

        esg_report_result = ""
        for category, tasks in categorized_tasks.items():
            esg_report_result += f"\n## {category}\n"
            for i, task in enumerate(tasks, start=1):
                esg_report_result += f"\n### {i}. {task['report_heading']}\n{task['task'].result()}\n"

        report = engine.load_template(
            template_name="report-template",
            overview=overview.result(),
            esg_report_result=esg_report_result,
            materiality=materiality.result(),
        )

        report_conclusion = await self.llm.chat(
            self.model,
            system_prompt=engine.load_prompt("create-report-conclusion"),
            user_prompt=f"The document is as follows\n{report}",
            agent="report"
        )

        return f"{report}\n\n{report_conclusion}"

    async def create_local_report(self, file: LLMFile, materiality_topics: dict[str, str]) -> str:
        materiality = materiality_topics if materiality_topics else "No Materiality topics identified."
        timeout = ClientTimeout(total=60*10)

        logger.info("Starting report generation process")
        overview = await self.llm.chat_with_file(
            self.model,
            system_prompt=engine.load_prompt("create-report-overview"),
            user_prompt="Generate an ESG report about the attached document.",
            files=[file],
            agent="report",
            timeout=timeout
        )

        categories = {}

        for category in QUESTIONS.keys():
            categories[category] = []
            for question in QUESTIONS[category]:
                logger.info(f"Processing report question for category: {category}")
                task = await self.llm.chat_with_file(
                    self.model,
                    system_prompt=engine.load_prompt("report-question-system-prompt"),
                    user_prompt=question["prompt"],
                    files=[file],
                    agent="report",
                    timeout=timeout
                )
                categories[category].append({
                    "report_heading": question["report_heading"],
                    "task": task
                })

        logger.info("Processing materiality section")
        materiality = await self.llm.chat_with_file(
            self.model,
            system_prompt=engine.load_prompt("create-report-materiality"),
            user_prompt=engine.load_prompt("create-report-materiality-user-prompt", materiality=materiality),
            files=[file],
            agent="report",
            timeout=timeout
        )

        esg_report_result = ""
        for category in categories.keys():
            esg_report_result += f"\n## {category}\n"
            for i, task in enumerate(categories[category], start=1):
                esg_report_result += f"\n### {i}. {task['report_heading']}\n{task['task']}\n"

        report = engine.load_template(
            template_name="report-template",
            overview=overview,
            esg_report_result=esg_report_result,
            materiality=materiality,
        )

        report_conclusion = await self.llm.chat(
            self.model,
            system_prompt=engine.load_prompt("create-report-conclusion"),
            user_prompt=f"The document is as follows\n{report}",
            agent="report"
        )

        return f"{report}\n\n{report_conclusion}"

    async def get_company_name(self, file: LLMFile, create_local_report: bool) -> str:
        system_prompt_file = (
            "find-company-name-from-file-system-prompt-name-only" if create_local_report
            else "find-company-name-from-file-system-prompt"
        )
        response = await self.llm.chat_with_file(
            self.model,
            system_prompt=engine.load_prompt(system_prompt_file),
            user_prompt=engine.load_prompt("find-company-name-from-file-user-prompt"),
            files=[file],
            agent="report",
            return_json = not create_local_report
        )
        return response if create_local_report else json.loads(response)["company_name"]
