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

def setup_api_key():
    """
    Set up the OpenAI API key from environment variables.
    Exits the program if no API key is found.
    
    Returns:
        bool: True if API key was found
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("""
        OpenAI API key not found! Please set one of these environment variables:
        - OPENAI_KEY
        - OPENAI_API_KEY
        
        You can set it using:
        - Windows: set OPENAI_API_KEY=your-key-here
        - macOS/Linux: export OPENAI_API_KEY=your-key-here
        - Or create a .env file in this directory with OPENAI_API_KEY=your-key-here
        """)
        sys.exit(1)
    else:
        print(f"Using OpenAI API key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        os.environ["OPENAI_API_KEY"] = api_key
        # Also set the key as OPENAI_KEY for backward compatibility
        os.environ["OPENAI_KEY"] = api_key
    
    return True

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
