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

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the loaded data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
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
    json_results = results_df.to_dict(orient='records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {output_path}")
