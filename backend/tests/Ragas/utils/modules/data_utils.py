"""
Data processing utilities for Ragas evaluation pipeline.
"""

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

                usage_records.append(row)

        print(f"Loaded and converted {len(usage_records)} LLM usage records")
        return usage_records
    except Exception as e:
        print(f"Error loading LLM usage CSV: {str(e)}")
        return []
