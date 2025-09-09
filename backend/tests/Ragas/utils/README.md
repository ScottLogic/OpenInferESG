# OpenInferESG RAGAS Evaluation

This directory contains scripts for evaluating OpenInferESG API responses using the RAGAS framework.

## Quick Start

### Prerequisites

1. Set up environment variables:
```bash
# On Windows
set OPENINFERESG_API_URL=http://localhost:8250
set OPENAI_KEY=your-openai-api-key-here

# Or create a .env file in the utils directory
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

The evaluation follows this flow:

1. **Run the enhanced pipeline** to:
   - Convert CSV data to JSONL
   - Query the OpenInferESG API
   - Generate response JSONL files

2. **Run RAGAS evaluation** on the responses to:
   - Calculate quality metrics
   - Generate results and visualization

## Usage

### Complete Evaluation Pipeline

```bash
# Check backend availability
python enhanced_run_evaluation_pipeline.py --check

# Run full pipeline with document
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf

# Limit number of questions (e.g., just 3)
python enhanced_run_evaluation_pipeline.py /path/to/your/document.pdf 3
```

### Run RAGAS Evaluation Separately

If you already have the `ragas_evaluation_with_responses.jsonl` file:

```bash
python ragas_evaluate.py

# With custom input/output paths
python ragas_evaluate.py --input path/to/input.jsonl --output path/to/output.json

# Skip chart generation
python ragas_evaluate.py --no-chart
```

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
- `ragas_eval_result.json`: Evaluation metrics (answer correctness, faithfulness, relevancy)
- `ragas_eval_result_chart.png`: Visualization of evaluation results

## Troubleshooting

- **Backend issues**: Ensure Docker containers are running (`docker-compose up -d backend redis`)
- **API timeouts**: The pipeline has automatic retries; try reducing batch size if needed
- **RAGAS errors**: 
  - Verify OpenAI API key is set correctly
  - Ensure RAGAS version 0.3.0+ is installed
