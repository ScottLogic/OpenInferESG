from io import BytesIO
from fastapi import UploadFile
from fastapi.datastructures import Headers
import pytest
import uuid

from src.session.file_uploads import FileUpload
from src.directors.report_director import create_report_from_file, prepare_file_for_report


mock_topics = {"topic1": "topic1 description", "topic2": "topic2 description"}
mock_report = "#Report on upload as markdown"
expected_answer_async = ("Your report for report-test-1.txt is ready to view.\n\n"
                   "The following materiality topics were identified for "
                   "CompanyABC which the report focuses on:\n\n"
                   "topic1\ntopic1 description\n\n"
                   "topic2\ntopic2 description")
expected_answer_sync = ("Your report for report-test-2.txt is ready to view.\n\n"
                   "The following materiality topics were identified for "
                   "CompanyABC which the report focuses on:\n\n"
                   "topic1\ntopic1 description\n\n"
                   "topic2\ntopic2 description")

async def test_prepare_file_for_report(mocker):

    mock_id = str(uuid.uuid4())
    filename = "test.pdf"

    session_file = FileUpload(
        id=str(mock_id),
        filename=filename,
        upload_id=None,
        content=None
    )

    mock_update_session_file_uploads = mocker.patch("src.directors.report_director.update_session_file_uploads")

    file_contents = b"test"
    prepare_file_for_report(file_contents, filename, mock_id)


    mock_update_session_file_uploads.assert_called_once_with(session_file)


@pytest.mark.asyncio
async def test_create_report_from_file(mocker):
    file_upload = FileUpload(id="1", filename="report-test-1.txt", content="report-test-1", upload_id=None)

    mock_id = uuid.uuid4()
    mocker.patch("uuid.uuid4", return_value=mock_id)
    filename = "report-test-1.txt"

    mock_config = mocker.patch("src.directors.report_director.config")
    mock_config.report_agent_llm = "openai"

    # Mock report agent
    mock_report_agent = mocker.AsyncMock()
    mock_report_agent.get_company_name.return_value = "CompanyABC"
    mock_report_agent.create_report_async.return_value = mock_report
    mocker.patch("src.directors.report_director.get_report_agent", return_value=mock_report_agent)

    # Mock materiality agent
    mock_materiality_agent = mocker.AsyncMock()
    mock_materiality_agent.list_material_topics_for_company.return_value = mock_topics
    mocker.patch("src.directors.report_director.get_materiality_agent", return_value=mock_materiality_agent)


    mock_store_report = mocker.patch("src.directors.report_director.store_report", return_value=file_upload)

    file = UploadFile(
        file=BytesIO(b"report-test-1"), size=12, headers=Headers({"content-type": "text/plain"}), filename=filename
    )
    file_contents = await file.read()
    response = await create_report_from_file(file_contents, filename, str(mock_id) )

    expected_response = {
        "filename": filename,
        "id": str(mock_id),
        "report": mock_report,
        "answer": expected_answer_async
    }

    mock_store_report.assert_called_once_with(expected_response)

    mock_materiality_agent.list_material_topics_for_company.assert_called_once_with("CompanyABC")

    assert response == expected_response

@pytest.mark.asyncio
async def test_create_report_from_file_lmstudio(mocker, tmp_path):
    file_upload = FileUpload(id="2", filename="report-test-2.txt", content="report-test-2", upload_id=None)
    mock_id = uuid.uuid4()
    mocker.patch("uuid.uuid4", return_value=mock_id)
    filename = "report-test-2.txt"

    # Mock config to use lmstudio
    mock_config = mocker.patch("src.directors.report_director.config")
    mock_config.report_agent_llm = "lmstudio"

    # Mock report agent
    mock_report_agent = mocker.AsyncMock()
    mock_report_agent.get_company_name.return_value = "CompanyABC"
    mock_report_agent.create_report_synchronous.return_value = mock_report
    mocker.patch("src.directors.report_director.get_report_agent", return_value=mock_report_agent)

    # Mock materiality agent
    mock_materiality_agent = mocker.AsyncMock()
    mock_materiality_agent.list_material_topics_for_company.return_value = mock_topics
    mocker.patch("src.directors.report_director.get_materiality_agent", return_value=mock_materiality_agent)

    # Patch REPORT_DIR to tmp_path
    mocker.patch("src.directors.report_director.REPORT_DIR", tmp_path)
    mock_store_report = mocker.patch("src.directors.report_director.store_report", return_value=file_upload)

    file = UploadFile(
        file=BytesIO(b"test2"), size=12, headers=Headers({"content-type": "text/plain"}), filename=filename
    )
    file_contents = await file.read()
    response = await create_report_from_file(file_contents, filename, str(mock_id))

    expected_response = {
        "filename": filename,
        "id": str(mock_id),
        "report": mock_report,
        "answer": expected_answer_sync
    }

    mock_store_report.assert_called_once_with(expected_response)
    mock_materiality_agent.list_material_topics_for_company.assert_called_once_with("CompanyABC")
    assert response == expected_response

    # Check that a markdown file was created in tmp_path
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    with open(files[0], "r") as f:
        assert f.read() == mock_report

