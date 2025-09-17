"""
RAGAS Evaluation Utilities
--------------------------
Utility functions for RAGAS evaluation, including file loading, API key setup, etc.
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path


def setup_api_key() -> None:
    """
    Set up the OpenAI API key from environment variables.
    Will exit if the key is not found.

    First tries OPENAI_API_KEY, then falls back to OPENAI_KEY.

    Returns:
        None
    """
    # Find the project root (where .env is located)
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    # Check if API key is available as OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")

    # If not found, try OPENAI_KEY
    if not api_key:
        openai_key = os.environ.get("OPENAI_KEY")
        if openai_key:
            # Set OPENAI_API_KEY from OPENAI_KEY
            os.environ["OPENAI_API_KEY"] = openai_key
            api_key = openai_key
            print("Set OPENAI_API_KEY from OPENAI_KEY environment variable.")

    # Final check if we have a key
    if not api_key:
        print("Error: OpenAI API key not found in environment variables.")
        print("Please set the OPENAI_API_KEY or OPENAI_KEY environment variable.")
        sys.exit(1)


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the loaded data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def save_results_to_json(results_df, output_path):
    """
    Save DataFrame results to a JSON file

    Args:
        results_df: DataFrame with results
        output_path: Path to save the JSON file

    Returns:
        None
    """
    # Convert DataFrame to JSON format
    json_results = results_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {output_path}")
