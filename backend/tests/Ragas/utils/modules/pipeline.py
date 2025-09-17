"""
Main pipeline for running Ragas evaluations with OpenInferESG.
"""

import os
import sys
import time
from typing import List, Dict, Optional
from pathlib import Path
from modules.api_client import OpenInferESGClient
from modules.data_utils import create_simplified_record, read_jsonl, write_jsonl, save_error_log
from dotenv import load_dotenv

# Find the project root (where .env is located)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def collect_api_responses(
    input_jsonl_path: str,
    output_jsonl_path: str,
    file_path: str,
    api_url: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: int = 3,
) -> None:
    """
    Make API calls to OpenInferESG's endpoint with file reference and collect responses

    Args:
        input_jsonl_path: Path to the input JSONL file containing questions
        output_jsonl_path: Path to save the enriched JSONL file with API responses
        file_path: Path to the file to upload and reference in questions
        api_url: The base URL of the OpenInferESG API
        limit: Optional limit on number of questions to process
        batch_size: Number of questions to process in each batch
    """
    api_url = os.environ.get("BACKEND_URL", "http://localhost:8250")  # Provide default value

    # Get the client (removed unused filename variable)
    client = OpenInferESGClient(api_url)

    # Read the existing JSONL file with questions
    records = read_jsonl(input_jsonl_path, limit)

    # Process records in smaller batches
    num_batches = (len(records) + batch_size - 1) // batch_size  # Ceiling division

    # Process each record and make API calls with file reference
    enriched_records = []
    errors = []

    # Upload the file once before processing batches
    print("Uploading file...")
    file_id = client.upload_file(file_path)
    if not file_id:
        for _ in range(2):  # Try one more time
            print("Retrying upload...")
            time.sleep(30)
            file_id = client.upload_file(file_path)
            if file_id:
                break

    if not file_id:
        print("Failed to upload file after multiple attempts. Exiting.")
        return

    # Wait for report generation once before processing batches
    print("Waiting for report generation...")
    report_ready = client.wait_for_report(file_id, max_wait_time=60000)
    if report_ready:
        print("Report is ready!")
    else:
        print("Report may not be fully ready. Proceeding with caution.")

    # Loop through batches
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(records))
        batch_records = records[start_idx:end_idx]

        print(
            f"\nProcessing batch {batch_num + 1}/{num_batches} "
            f"(questions {start_idx + 1}-{end_idx} of {len(records)})..."
        )

        # Process records in the current batch
        process_batch(
            client=client,
            batch_records=batch_records,
            file_id=file_id,
            start_idx=start_idx,
            total_records=len(records),
            enriched_records=enriched_records,
            errors=errors,
        )

    # Write the enriched records to the output JSONL file
    write_jsonl(output_jsonl_path, enriched_records)

    print(f"\nCompleted processing {len(records)} questions.")
    print(f"Errors encountered: {len(errors)}")

    # Save error log if there are errors
    if errors:
        error_log_path = os.path.join(os.path.dirname(output_jsonl_path), "api_errors.json")
        save_error_log(error_log_path, errors)


def process_batch(
    client: OpenInferESGClient,
    batch_records: List[Dict],
    file_id: str,
    start_idx: int,
    total_records: int,
    enriched_records: List[Dict],
    errors: List[Dict],
) -> None:
    """
    Process a batch of records

    Args:
        client: The OpenInferESG API client
        batch_records: The records to process in this batch
        filename: The name of the file being referenced
        file_id: The ID of the uploaded file
        start_idx: The starting index of this batch in the overall record list
        total_records: The total number of records being processed
        enriched_records: The list of enriched records to append to
        errors: The list of errors to append to
    """
    for batch_i, record in enumerate(batch_records):
        # Calculate the global index for this record
        i = start_idx + batch_i

        original_question = record["user_input"]
        print(f"Processing question {i + 1}/{total_records}: {original_question[:70]}...")

        # Check if we should skip due to too many timeouts
        if i > 0 and len(errors) >= 3 and all("timeout" in err.get("error", "").lower() for err in errors[-3:]):
            print("Multiple consecutive timeouts detected. Skipping to prevent API overload.")
            error_msg = "Question skipped due to multiple previous timeouts"
            errors.append({"question": original_question, "error": error_msg})
            enriched_record = create_simplified_record(original_question, None, record)
            enriched_records.append(enriched_record)

            # Take a longer break to let the API recover
            time.sleep(60)
            continue

        # Add a pause before starting a new question
        time.sleep(10)

        # Get answer from the API
        api_response, error_msg = client.get_answer(original_question, file_id=file_id)

        if api_response:
            enriched_record = create_simplified_record(original_question, api_response, record)
            enriched_records.append(enriched_record)
            print(f"Got API response ({len(str(api_response))} chars)")
        else:
            errors.append({"question": original_question, "error": error_msg})
            enriched_record = create_simplified_record(original_question, None, record)
            enriched_records.append(enriched_record)
            print(f"Error: {error_msg}")

        # Add a delay between questions to avoid overwhelming the API
        time.sleep(5)


def print_backend_help() -> None:
    """Print helpful information about starting the backend"""
    print("\n=============================================")
    print("ERROR: OpenInferESG Backend is not available!")
    print("=============================================")
    print("\nHOW TO START THE BACKEND:")
    print("1. Open a terminal in the root project directory")
    print("2. Run the following command to start all services:")
    print("   docker-compose up -d")
    print("\n3. OR run these commands to start specific services:")
    print("   docker-compose up -d backend")
    print("   docker-compose up -d redis")
    print("\n4. Verify the backend is running by accessing:")
    print("   http://localhost:8250/health")
    print("\nADDITIONAL TROUBLESHOOTING:")
    print("- Check Docker status: 'docker ps'")
    print("- Look for errors in logs: 'docker-compose logs backend'")
    print("=============================================\n")


def main(file_path: Optional[str] = None, question_limit: Optional[int] = None) -> None:
    """
    Main function to run the entire process with file upload:
    1. Convert CSV to JSONL
    2. Upload a file
    3. Collect API responses with file references and add them to the JSONL file

    Args:
        file_path: Path to the file to upload
        question_limit: Optional limit on number of questions to process
    """
    # Get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.dirname(script_dir)  # utils folder
    ragas_dir = os.path.dirname(utils_dir)  # Ragas folder

    # Get the API URL
    api_url = os.environ.get("BACKEND_URL", "http://localhost:8250")  # Provide default value

    # Check if the server is available before proceeding
    client = OpenInferESGClient(api_url)
    if not client.check_availability():
        print_backend_help()
        return

    # Step 1: Convert CSV to JSONL
    print("Step 1: Converting CSV to JSONL...")

    # Dynamically import csv_to_jsonl_converter.py from utils directory
    converter_path = os.path.join(utils_dir, "csv_to_jsonl_converter.py")
    try:
        # Add the utils directory to the Python path temporarily
        sys.path.insert(0, utils_dir)

        # Use a simple import
        from csv_to_jsonl_converter import convert_csv_to_jsonl

        # Call the convert function
        convert_csv_to_jsonl()

        # Remove the utils directory from the Python path
        sys.path.pop(0)
    except Exception as e:
        print(f"Error importing or running csv_to_jsonl_converter: {str(e)}")
        print(f"Looking for file at: {converter_path}")
        print("Attempting to list available files:")
        try:
            files = os.listdir(utils_dir)
            print(f"Files in {utils_dir}: {files}")
        except Exception as list_error:
            print(f"Error listing directory: {str(list_error)}")
        sys.exit(1)

    # Get the path of the created JSONL file
    input_jsonl_path = os.path.join(ragas_dir, "files", "ragas_evaluation_dataset.jsonl")
    output_jsonl_path = os.path.join(ragas_dir, "files", "ragas_evaluation_with_responses.jsonl")

    # Require a file path to be provided
    if file_path is None:
        print("ERROR: No file path provided. Please provide a path to a PDF file.")
        print("Usage: python enhanced_run_evaluation_pipeline.py [file_path] [question_limit]")
        sys.exit(1)

    # Step 2: Collect API responses and enrich JSONL with file references
    print(f"\nStep 2: Collecting API responses with file references for {file_path}...")
    collect_api_responses(input_jsonl_path, output_jsonl_path, file_path, api_url, question_limit)

    print("\nProcess completed successfully!")
    print(f"Final enriched JSONL file saved to: {output_jsonl_path}")


def check_backend_status() -> None:
    """Check if the backend is running without running the full pipeline"""
    api_url = os.environ.get("BACKEND_URL", "http://localhost:8250")
    print("\n=== BACKEND STATUS CHECK ===")
    client = OpenInferESGClient(api_url)
    available = client.check_availability()

    if available:
        print("\n✅ SUCCESS: The OpenInferESG backend is running and responding!")
        print(f"API URL: {api_url}")
        print("\nYou can now run the full pipeline with:")
        print("python enhanced_run_evaluation_pipeline.py [file_path] [question_limit]")
    else:
        print("\n❌ FAILED: The OpenInferESG backend is not available.")
        print_backend_help()

    print("\n===========================")


if __name__ == "__main__":
    # Check if we're just checking server status
    if len(sys.argv) > 1 and sys.argv[1] in ["--check", "-c", "check", "status"]:
        check_backend_status()
        sys.exit(0)

    # Check for file path argument
    file_path = None
    if len(sys.argv) > 1 and not sys.argv[1].isdigit():
        file_path = sys.argv[1]
        print(f"Using specified file: {file_path}")

    # Check if limit argument is provided
    question_limit = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        question_limit = int(sys.argv[1])
        print(f"Limiting to {question_limit} question(s)")
    elif len(sys.argv) > 2:
        try:
            question_limit = int(sys.argv[2])
            print(f"Limiting to {question_limit} question(s)")
        except ValueError:
            print(f"Invalid limit argument: {sys.argv[2]}. Must be an integer.")
            sys.exit(1)

    # Display help if no arguments provided
    if len(sys.argv) <= 1:
        print("OpenInferESG Enhanced Evaluation Pipeline")
        print("----------------------------------------")
        print("This script uploads a file to the OpenInferESG API and runs evaluation questions.")
        print("\nUsage:")
        print("  python enhanced_run_evaluation_pipeline.py [file_path] [question_limit]")
        print("  python enhanced_run_evaluation_pipeline.py --check")
        print("\nArguments:")
        print("  file_path       - Path to the file to upload (PDF) [REQUIRED]")
        print("  question_limit  - Number of questions to process")
        print("  --check/-c      - Just check if the backend is running")
        print("\nExamples:")
        print("  python enhanced_run_evaluation_pipeline.py report.pdf 5")
        print("  python enhanced_run_evaluation_pipeline.py report.pdf")
        print("  python enhanced_run_evaluation_pipeline.py --check")
        print("\nChecking if backend is available:")
        check_backend_status()
        sys.exit(0)

    main(file_path, question_limit)
