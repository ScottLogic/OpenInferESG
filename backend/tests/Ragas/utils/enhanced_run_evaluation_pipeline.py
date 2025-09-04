"""
Enhanced Run Evaluation Pipeline for OpenInferESG Ragas evaluation.

This script uploads a file to the OpenInferESG API, waits for report generation,
and runs a series of evaluation questions against the uploaded file.
"""
import sys
from modules.pipeline import main, check_backend_status

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
