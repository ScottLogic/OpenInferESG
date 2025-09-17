# OpenInferESG RAGAS Evaluation

This directory contains scripts for evaluating OpenInferESG API responses using the RAGAS framework.

## Quick Start

### Prerequisites

1. Set up environment variables:
```bash
# Ensure the following variables are set in the project root .env file:
# OPENAI_KEY=your-openai-api-key-here
# RAGAS_OPENAI_MODEL=gpt-4o  # Options: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.

# For API testing, you can set:
# BACKEND_URL=http://localhost:8250
```

2. Setup Python environment:
```bash
# Navigate to OpenInferESG root directory
cd /path/to/OpenInferESG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
cd backend/tests/Ragas/utils
pip install -r requirements.txt
```

## Evaluation Process

The evaluation follows this two-step process (each step requires running a separate script):

1. **Run the enhanced pipeline** (`enhanced_run_evaluation_pipeline.py`) to:
   - Convert CSV data to JSONL
   - Query the OpenInferESG API
   - Generate response JSONL files

2. **Run RAGAS evaluation** (`ragas_evaluate.py`) on the responses to:
   - Calculate quality metrics
   - Generate results and visualization

## Usage

### Step 1: Run Enhanced Evaluation Pipeline

This first step creates the necessary JSONL file with API responses, but does not perform RAGAS evaluation:

> **Important**: Before running the enhanced evaluation pipeline, it's adviced to set `ALLOWED_CHAT_AGENTS="fileagent"` in your `.env` file. This ensures agent usage to just the file agent, which is all that's needed for document processing during evaluation.

```bash
# Check backend availability
python enhanced_run_evaluation_pipeline.py --check

# Run pipeline with document (prepares data for RAGAS evaluation)
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf

# Limit number of questions (e.g., just 3)
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf 3
```

### Run RAGAS Evaluation (Step 2)

After running the enhanced pipeline which produces the `ragas_evaluation_with_responses.jsonl` file, you must run the RAGAS evaluation script:

```bash
python ragas_evaluate.py

# With custom input/output paths
python ragas_evaluate.py --input path/to/input.jsonl --output path/to/output.json

# Skip chart generation
python ragas_evaluate.py --no-chart
```

## Environment Variables

The RAGAS evaluation uses the following environment variables from the root `.env` file:

- **OPENAI_KEY**: Your OpenAI API key required for RAGAS evaluation
- **RAGAS_OPENAI_MODEL**: The OpenAI model to use for evaluations (default: gpt-4o)
- **RAGAS_METRICS**: Comma-separated list of metrics to evaluate:
  - `factual_correctness`: Measures accuracy of factual content
  - `semantic_similarity`: Evaluates meaning preservation
  - `answer_accuracy`: Assesses overall answer quality

## Key Files

- **Input/Process Scripts**:
  - `enhanced_run_evaluation_pipeline.py`: Main pipeline that converts CSV to JSONL and collects API responses
  - `csv_to_jsonl_converter.py`: Converts question and ground truth CSV data to JSONL format

- **Evaluation Script**:
  - `ragas_evaluate.py`: Evaluates responses using RAGAS metrics

- **Module Structure**:
   - **For Enhanced Pipeline**:
    - `modules/pipeline.py`: Main pipeline orchestration and workflow management
    - `modules/api_client.py`: API communication with OpenInferESG backend
    - `modules/data_utils.py`: Data processing and transformation utilities
   
   - **For RAGAS Evaluation**:
    - `modules/ragas_utils.py`: Utility functions for file operations and API key setup
    - `modules/ragas_evaluation.py`: Core evaluation functions using RAGAS metrics
    - `modules/ragas_visualization.py`: Chart generation for result visualization
    - `modules/ragas_cli.py`: Command-line interface for evaluation
  

## Output Files

All output files are stored in `../files/`:

- `ragas_evaluation_dataset.jsonl`: Initial questions and references
- `ragas_evaluation_with_responses.jsonl`: Questions with API responses
- `ragas_eval_result.json`: Evaluation metrics as configured in RAGAS_METRICS (default: factual_correctness, semantic_similarity, answer_accuracy)
- `ragas_eval_result_chart.png`: Visualization of evaluation results

## Troubleshooting

- **Backend issues**: Ensure Docker containers are running (`docker-compose up -d backend redis`)
- **API timeouts**: The pipeline has automatic retries; try reducing batch size if needed
- **RAGAS errors**: 
  - Verify the OPENAI_KEY is set correctly in the root .env file
  - Check that RAGAS_OPENAI_MODEL is set to a valid model (default: gpt-4o)
  - Ensure RAGAS version 0.3.0+ is installed
