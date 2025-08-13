import csv
from io import StringIO

from unittest.mock import mock_open

from src.utils.usage_recorder import CSV_HEADERS, ConsoleUsageRecorder, CSVUsageRecorder


def test_console_usage_recorder_record_activity(caplog):
    """Test that ConsoleUsageRecorder logs activity details to console"""
    recorder = ConsoleUsageRecorder()

    # Test with dictionary token usage
    with caplog.at_level("INFO"):
        token_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        recorder.record_activity(model="test-model", provider="test-provider", token_usage=token_usage, duration=1.5)

        assert "test-model" in caplog.text
        assert str(token_usage) in caplog.text
        assert "1.5" in caplog.text


def test_csv_usage_recorder_record_activity(mocker):
    """Test recording activity with dictionary token usage"""
    with StringIO() as csv_output:
        mock_file = mocker.patch("builtins.open", mock_open())
        mocker.patch("os.path.isfile", return_value=False)  # Simulate new file creation

        # Patch the file handle to use StringIO for CSV writing
        mock_file.return_value.__enter__.return_value = csv_output

        recorder = CSVUsageRecorder()
        token_usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        recorder.record_activity(model="test-model", provider="test-provider", token_usage=token_usage, duration=1.5)

        # Reset the StringIO position to read from the beginning
        csv_output.seek(0)

        # Read and check the CSV content
        csv_reader = csv.reader(csv_output)
        rows = list(csv_reader)

        # First row should be headers
        assert rows[0] == CSV_HEADERS

        # Second row should be the data
        assert "test-model" in rows[1]
        assert "10" in rows[1]
        assert "20" in rows[1]
        assert "30" in rows[1]

def test_csv_usage_recorder_uses_default_filename(mocker):
    """Test that CSVUsageRecorder uses the default filename when config is None"""
    # Mock Config to return None for filename
    mocker.patch("src.utils.usage_recorder.config.llm_usage_log_filename", None)

    recorder = CSVUsageRecorder()

    # Check that the default filename is used
    assert recorder.csv_file_path.name == "llm_usage.csv"
