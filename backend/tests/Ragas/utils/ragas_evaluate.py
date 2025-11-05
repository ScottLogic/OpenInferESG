#!/usr/bin/env python3
"""
RAGAS Evaluation Command Line Tool
---------------------------------
A command-line tool to evaluate question-answering systems using RAGAS metrics.
"""

import os
import sys
import argparse
import asyncio
from modules.ragas_evaluation import evaluate_with_ragas


async def run_evaluation(input_path: str, output_path: str, llm: str) -> None:
    """
    Run the RAGAS evaluation process end-to-end

    Args:
        input_path: Path to input JSONL file with responses
        output_path: Path to save JSON output results
        llm: The LLM model to use for evaluation
    """

    # Run RAGAS evaluation
    print(f"Running RAGAS evaluation on {input_path}...")
    results_df = await evaluate_with_ragas(input_path)
    results_df["llm"] = llm
    print("RAGAS evaluation completed.")
    print(f"Appending results to CSV file... {output_path}")

    file_exists = os.path.isfile(output_path)
    results_df.to_csv(output_path, mode="a", header=not file_exists, index=False)

    print("Results appended to CSV file.")


async def main():
    """Main entry point for command-line execution"""

    # Set up default file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate to Ragas root
    default_input = os.path.normpath(os.path.join(base_dir, "files/ragas_evaluation_with_responses.jsonl"))
    default_output = os.path.normpath(os.path.join(base_dir, "files/ragas_eval_result.csv"))

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate responses using RAGAS metrics")
    parser.add_argument("--llm", "-l", help="LLM model to use for evaluation")
    parser.add_argument("--input", "-i", dest="input_jsonl", help="Path to input JSONL file", default=default_input)
    parser.add_argument("--output", "-o", help="Path to save CSV output", default=default_output)
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_jsonl):
        print(f"Error: Input file not found: {args.input_jsonl}")
        sys.exit(1)

    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output}")
    print(f"LLM model: {args.llm}")

    # Run evaluation
    try:
        await run_evaluation(args.input_jsonl, args.output, args.llm)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
