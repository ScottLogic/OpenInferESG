import datetime
import sys
from fastapi import HTTPException
from src.agents.report_agent import ReportAgent
from src.utils import Config
from src.llm.llm import LLMFile
from src.session.file_uploads import FileUpload, ReportResponse, store_report, update_session_file_uploads
from src.agents import get_report_agent, get_materiality_agent
from pathlib import Path

MAX_FILE_SIZE = 40 * 1024 * 1024
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
config = Config()


def prepare_file_for_report(file_contents: bytes, filename: str, file_id: str):
    file_size = sys.getsizeof(file_contents)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File upload must be less than {MAX_FILE_SIZE} bytes")

    session_file = FileUpload(id=file_id, filename=filename, upload_id=None, content=None)

    update_session_file_uploads(session_file)


async def create_report_from_file(file_contents: bytes, filename: str, file_id: str) -> ReportResponse:
    file = LLMFile(filename=filename, file=file_contents)

    report_agent = get_report_agent()

    create_local_report = config.report_agent_llm == "lmstudio"

    company_name = await report_agent.get_company_name(file)

    topics = await get_materiality_agent().list_material_topics_for_company(company_name)

    report = await create_report(file, topics, create_local_report, report_agent)

    report_response = ReportResponse(
        filename=filename,
        id=file_id,
        report=report,
        answer=create_report_chat_message(filename, company_name, topics),
    )

    if create_local_report:
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        report_file_name = f"{Path(filename).stem}_{timestamp}.md"
        filepath = f"{REPORT_DIR}/{report_file_name}"
        with open(filepath, "w") as text_file:
            text_file.write(report)

    store_report(report_response)

    return report_response


def create_report_chat_message(file_name: str, company_name: str, topics: dict[str, str]) -> str:
    topics_with_markdown = [f"{key}\n{value}" for key, value in topics.items()]
    topics_summary = "\n\n".join(topics_with_markdown)

    return (
        f"Your report for {file_name} is ready to view.\n\n"
        f"The following materiality topics were identified for {company_name} which the report focuses on:\n\n"
        f"{topics_summary}"
    )


async def create_report(
    file: LLMFile, materiality_topics: dict[str, str], create_local_report: bool, report_agent: ReportAgent
) -> str:
    if create_local_report:
        return await report_agent.create_report_synchronous(file, materiality_topics)
    else:
        return await report_agent.create_report_async(file, materiality_topics)
