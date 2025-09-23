"""
Data processing utilities for Ragas evaluation pipeline.
"""

import bisect
import datetime
import json
from typing import Dict, List, Any, Optional


def create_simplified_record(
    question: str,
    api_response: Optional[Dict[str, Any]],
    record: Dict[str, Any],
    usage_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a simplified record with only the required fields

    Args:
        question: The question with file reference
        api_response: The API response, or None if there was an error
        record: The original record with reference and context
        usage_data: Optional LLM usage data from the CSV log

    Returns:
        A simplified record with only user_input, response, reference, reference_contexts, and llm_usage if available
    """
    # Extract just the answer text from the API response
    answer_text = ""
    if api_response and isinstance(api_response, dict):
        answer_text = api_response.get("answer", "")

    # Create simplified record
    result = {
        "user_input": question,
        "response": answer_text,
        "reference": record["reference"],
        "reference_contexts": record["reference_contexts"],
    }

    # Add usage data if available
    if usage_data:
        result["llm_usage"] = usage_data

    return result


def read_jsonl(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Read records from a JSONL file

    Args:
        file_path: Path to the JSONL file
        limit: Optional limit on number of records to read

    Returns:
        List of records from the JSONL file
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    if limit is not None and limit > 0:
        records = records[:limit]
        print(f"Loaded {len(records)} records (limited to {limit})")
    else:
        print(f"Loaded {len(records)} records")

    return records


def write_jsonl(file_path: str, records: List[Dict]) -> None:
    """
    Write records to a JSONL file

    Args:
        file_path: Path to the JSONL file
        records: Records to write
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {file_path}")


def save_error_log(file_path: str, errors: List[Dict]) -> None:
    """
    Save error log to a JSON file

    Args:
        file_path: Path to the JSON file
        errors: List of errors to save
    """
    if not errors:
        return

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(errors, file, indent=2, ensure_ascii=False)

    print(f"Error log saved to {file_path}")


def load_and_convert_usage_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load LLM usage data from CSV file and convert types upfront

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of usage records with converted types
    """
    import csv
    import datetime

    usage_records = []
    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert numeric fields
                for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    if row.get(field) != "N/A" and row.get(field):
                        try:
                            row[field] = int(row[field])
                        except ValueError:
                            row[field] = 0
                    else:
                        row[field] = 0

                # Convert float fields
                for field in ["duration_seconds"]:
                    if row.get(field) != "N/A" and row.get(field):
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            row[field] = 0.0
                    else:
                        row[field] = 0.0

                # Parse timestamp field
                if row.get("timestamp"):
                    try:
                        # Parse consistent timezone-aware timestamp format
                        row["timestamp"] = datetime.datetime.fromisoformat(row["timestamp"])
                    except ValueError:
                        # If parsing fails, remove the row to avoid comparison issues
                        continue

                usage_records.append(row)

        # Sort records by timestamp for efficient binary search operations
        usage_records.sort(key=lambda x: x.get("timestamp", datetime.datetime.min))
        
        print(f"Loaded and converted {len(usage_records)} LLM usage records")
        return usage_records
    except Exception as e:
        print(f"Error loading LLM usage CSV: {str(e)}")
        return []


def find_and_remove_usage_in_timerange(usage_data: List[Dict], start_time: str, end_time: str) -> List[Dict]:
    """
    Efficiently find usage records within a time range using binary search and remove them from the dataset.
    Assumes usage_data is sorted by timestamp. Modifies the input list by removing used records.

    Args:
        usage_data: List of usage records sorted by timestamp (datetime objects) - modified in place
        start_time: Start time as ISO string
        end_time: End time as ISO string

    Returns:
        List of usage records within the time range (removed from input list)
    """
    if not usage_data:
        return []

    # Parse the timestamp strings
    try:
        start_dt = datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except ValueError as e:
        print(f"Error parsing timestamps: {e}")
        return []

    # Use binary search to find the range
    # Find the leftmost insertion point for start_time
    start_idx = bisect.bisect_left(usage_data, start_dt, key=lambda x: x.get("timestamp", datetime.datetime.min))

    # Find the leftmost insertion point for end_time (exclusive)
    end_idx = bisect.bisect_left(usage_data, end_dt, key=lambda x: x.get("timestamp", datetime.datetime.min))

    # Extract the matching records
    matching_records = usage_data[start_idx:end_idx]

    # Remove the used records from the original list (in reverse order to maintain indices)
    del usage_data[start_idx:end_idx]

    return matching_records
