# OpenInferESG API Testing and Evaluation

This directory contains scripts for testing the OpenInferESG API with a set of questions and evaluating the responses.

## Setup and Installation

### Setting up a Virtual Environment

Before running the scripts, it's recommended to create a virtual environment:

```bash
# Navigate to the OpenInferESG directory
cd /path/to/OpenInferESG

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### Installing Dependencies

After activating the virtual environment, install the required dependencies:

```bash
# Navigate to the Ragas utils directory
cd backend/tests/Ragas/utils

# Install the requirements
pip install -r requirements.txt
```

## Files

- `csv_to_jsonl_converter.py`: Converts question and ground truth CSV data to JSONL format
- `enhanced_run_evaluation_pipeline.py`: Enhanced pipeline with improved error handling and batch processing
- `requirements.txt`: Contains all the Python dependencies needed to run the scripts

## Process Overview

1. **CSV to JSONL Conversion**: 
   - Converts the question_and_groundtruth_dataset.csv to JSONL format
   - The JSONL file includes: user_input, reference, reference_contexts

2. **API Response Collection**:
   - Makes API calls to OpenInferESG's chat endpoint
   - Adds API responses to the JSONL file
   - The enriched JSONL includes: user_input, response, reference, reference_contexts

(Pending)
3. **Response Evaluation**:
   - Uses RAGAS metrics to evaluate the quality of responses
   - Calculates metrics such as accuracy, faithfulness, answer relevancy, etc.
   - Saves evaluation results to a CSV file

## Usage

### Using the Enhanced Pipeline

The enhanced pipeline includes better error handling, batch processing, and improved reporting.

#### Check Backend Status

Before running the pipeline, check if the OpenInferESG backend is available:

```bash
cd backend/tests/Ragas/utils
python enhanced_run_evaluation_pipeline.py --check
```

#### Run the Enhanced Pipeline

A PDF file path is required:

```bash
cd backend/tests/Ragas/utils
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf
```

#### Process Limited Number of Questions

To process only a specific number of questions (e.g., just three), provide both the PDF path and question limit:

```bash
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf 3
```

#### Running the Evaluation

The evaluation is integrated into the enhanced pipeline. Results will be saved automatically.

### Test a Single Question Directly

```bash
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf 1
```

### Run Individual Steps

1. Convert CSV to JSONL:
```bash
python csv_to_jsonl_converter.py
```

### Configuration

#### Environment Variables

Before running the scripts, make sure to set up the required environment variables:

```bash
# On Windows:
set OPENINFERESG_API_URL=http://localhost:8250
set OPENAI_KEY=your-openai-api-key-here

# On macOS/Linux:
# export OPENINFERESG_API_URL=http://localhost:8250
# export OPENAI_KEY=your-openai-api-key-here
```

Or create a `.env` file in the utils directory with the following content:

```
OPENINFERESG_API_URL=http://localhost:8250
OPENAI_KEY=your-openai-api-key-here
```

- API URL: Set the environment variable `OPENINFERESG_API_URL` to specify the API endpoint
  - Default: http://localhost:8250
- RAGAS Evaluation: Requires OpenAI API key set as `OPENAI_KEY` environment variable

## Output Files

All output files will be stored inside the Files folder within the Ragas folder:

- `ragas_evaluation_dataset.jsonl`: Initial JSONL file with questions and references
- `ragas_evaluation_with_responses.jsonl`: Enriched JSONL with API responses
- `evaluation_results.csv`: Evaluation metrics for each question and average scores (will be generated after implementation of the next evaluation step of calling Ragas to generate metrics)

## Troubleshooting

If you encounter issues with the enhanced pipeline:

1. **Backend not available**: 
   - Make sure Docker containers are running: `docker-compose up -d backend redis`
   - Check logs: `docker-compose logs -f backend`

2. **API timeouts**: 
   - The enhanced pipeline has automatic retries and batch processing
   - If timeouts persist, try reducing the batch size in the script

3. **RAGAS evaluation errors**:
   - Ensure your OpenAI API key is set correctly
   - Check that the JSONL file has the correct format
