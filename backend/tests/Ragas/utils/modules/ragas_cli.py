"""
RAGAS CLI Module
--------------
Command-line interface for running RAGAS evaluations.
"""
import os
import sys
import argparse
import asyncio
import nest_asyncio
import pandas as pd

async def run_evaluation(input_path: str, output_path: str, skip_chart: bool = False) -> pd.DataFrame:
    """
    Run the RAGAS evaluation process end-to-end
    
    Args:
        input_path: Path to input JSONL file with responses
        output_path: Path to save JSON output results
        skip_chart: Whether to skip chart generation
        
    Returns:
        DataFrame with evaluation results
    """
    # Import locally to avoid circular imports
    from .ragas_utils import setup_api_key
    from .ragas_evaluation import evaluate_with_ragas
    
    # Ensure API key is set up (will exit if not found)
    setup_api_key()
    
    # Run RAGAS evaluation
    print(f"Running RAGAS evaluation on {input_path}...")
    results_df = await evaluate_with_ragas(input_path, output_path, skip_chart)
    
    print(f"Evaluation complete! Results saved to {output_path}")
    if not skip_chart:
        chart_path = output_path.replace('.json', '_chart.png')
        print(f"Chart saved to {chart_path}")
        
    return results_df


async def main():
    """Main entry point for command-line execution"""
    
    # Set up default file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Navigate to Ragas root
    default_input = os.path.normpath(os.path.join(base_dir, "files/ragas_evaluation_with_responses.jsonl"))
    default_output = os.path.normpath(os.path.join(base_dir, "files/ragas_eval_result.json"))

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate responses using RAGAS metrics")
    parser.add_argument("--input", "-i", dest="input_jsonl", 
                        help="Path to input JSONL file", default=default_input)
    parser.add_argument("--output", "-o", help="Path to save JSON output", default=default_output)
    parser.add_argument("--no-chart", action="store_true", help="Skip chart visualization")
    args = parser.parse_args()

    # Apply nest_asyncio for Jupyter compatibility
    if nest_asyncio:
        nest_asyncio.apply()
    
    # Validate input file
    if not os.path.exists(args.input_jsonl):
        print(f"Error: Input file not found: {args.input_jsonl}")
        sys.exit(1)
    
    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output}")
    if args.no_chart:
        print("Chart generation is disabled")
    
    # Run evaluation
    try:
        await run_evaluation(args.input_jsonl, args.output, args.no_chart)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)
