"""
RAGAS Evaluation Module
----------------------
Core functions for running RAGAS evaluations on question-answering data.
"""
import os
from typing import Optional
import pandas as pd

# Set flag for RAGAS availability
RAGAS_AVAILABLE = False

# Try to import RAGAS components - all these imports might be None if import fails
try:
    # Import RAGAS evaluation components
    from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    # Import LangChain components
    from langchain_openai import ChatOpenAI
    # Set flag to indicate successful imports
    RAGAS_AVAILABLE = True
except ImportError as e:
    # In case of import failure, define empty variables to avoid undefined errors
    faithfulness = answer_relevancy = answer_correctness = None
    evaluate = EvaluationDataset = SingleTurnSample = None
    LangchainLLMWrapper = ChatOpenAI = None
    print(f"Warning: RAGAS imports failed: {e}")
    print("RAGAS evaluation will not be available unless you install required packages.")


def create_ragas_dataset(data):
    """
    Create a RAGAS evaluation dataset from the input data.

    Args:
        data: List of dictionaries containing evaluation samples

    Returns:
        Tuple of (EvaluationDataset object, processed samples list)
    """
    if not RAGAS_AVAILABLE or SingleTurnSample is None or EvaluationDataset is None:
        raise ImportError("RAGAS components are not available. Please install the RAGAS package.")

    samples = []
    for sample in data:
        # Skip samples without responses
        if not sample.get("response"):
            print("Skipping sample - no response")
            continue

        # Create SingleTurnSample
        # If no explicit reference is provided, use the first reference context as the reference
        reference = sample.get("reference", "")
        reference_contexts = sample.get("reference_contexts", [])
        if not reference and reference_contexts and len(reference_contexts) > 0:
            reference = reference_contexts[0]

        # Create a sample using the RAGAS SingleTurnSample class
        eval_sample = SingleTurnSample(
            user_input=sample.get("user_input", ""),
            retrieved_contexts=[context for context in sample.get("reference_contexts", []) if context],
            response=sample.get("response", ""),
            reference=reference  # Use either provided reference or first context
        )
        samples.append(eval_sample)

    print(f"Created {len(samples)} samples for evaluation")
    # Create a dataset using the RAGAS EvaluationDataset class
    return EvaluationDataset(samples=samples), samples


def create_ragas_llm():
    """
    Create and configure the LLM for RAGAS evaluation.

    Returns:
        LangchainLLMWrapper instance
    """
    if not RAGAS_AVAILABLE or ChatOpenAI is None or LangchainLLMWrapper is None:
        raise ImportError("RAGAS components or LangChain are not available. Please install required packages.")

    # Use gpt-4o as the judge LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running evaluation.")

    print(f"API Key found for ChatOpenAI: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")

    # Create the ChatOpenAI model and wrap it with LangchainLLMWrapper
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0  # Use temperature=0 for more consistent evaluations
    )

    return LangchainLLMWrapper(chat_model)


async def evaluate_with_ragas(jsonl_path: str, output_json_path: Optional[str] = None, 
                              skip_chart: bool = False) -> pd.DataFrame:
    """
    Evaluate responses using RAGAS metrics

    Args:
        jsonl_path: Path to the input JSONL file with responses
        output_json_path: Path to save the JSON results
        skip_chart: Whether to skip generating the bar chart

    Returns:
        DataFrame with evaluation results
    """
    # Import locally to avoid circular imports
    from .ragas_utils import load_jsonl_data, save_results_to_json
    from .ragas_visualization import generate_bar_chart

    print("Setting up RAGAS evaluation...")
    print(f"Loading data from {jsonl_path}...")
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} samples for evaluation")

    try:
        # Make sure all required RAGAS functions are available
        if not RAGAS_AVAILABLE or None in (evaluate, answer_correctness, faithfulness, answer_relevancy):
            raise ImportError("Required RAGAS metrics or evaluation function not available")

        # Create LLM for evaluation
        llm = create_ragas_llm()

        # Create dataset
        dataset, samples = create_ragas_dataset(data)

        # Define standard metrics to evaluate
        metrics = [answer_correctness, faithfulness, answer_relevancy]

        # Run the evaluation
        print("Running RAGAS evaluation (this may take a while)...")
        # Check if evaluate is None before calling it
        if evaluate is None:
            raise ImportError("RAGAS evaluate function is not available")
        results = evaluate(dataset=dataset, metrics=metrics, llm=llm)

        # Process results into a pandas DataFrame
        if not hasattr(results, 'to_pandas'):
            error_msg = (
                "Incompatible RAGAS version detected. The RAGAS library doesn't provide the 'to_pandas' method "
                "which is required for extracting evaluation results. Please use RAGAS version 0.3.0 or later."
            )
            raise ImportError(error_msg)

        # Extract metrics from results
        scores_df = results.to_pandas()
        print(f"Results DataFrame columns: {list(scores_df.columns)}")

        # Define expected metrics and find which ones are available
        expected_metrics = ["answer_correctness", "faithfulness", "answer_relevancy"]
        available_metrics = [col for col in scores_df.columns if col in expected_metrics]
        print(f"Found metrics: {available_metrics}")

        # Process results into a consistent format
        result_data = []
        for i, (_, row) in enumerate(scores_df.iterrows()):
            # Create base result row with question and null metrics
            result_row = {
                "question": samples[i].user_input if i < len(samples) else f"Question {i+1}",
                **{metric: None for metric in expected_metrics}  # Initialize all metrics as None
            }

            # Fill in available metrics
            for metric in available_metrics:
                try:
                    value = row[metric]
                    # Convert Series to scalar if needed
                    # Check for pandas Series type specifically
                    if isinstance(value, pd.Series) and len(value) > 0:
                        value = value.iloc[0]

                    # Validate the value
                    if value is not None and (not isinstance(value, float) or
                                            (value == value and value != float('inf') and value != float('-inf'))):
                        result_row[metric] = value
                except Exception as e:
                    print(f"Error extracting {metric}: {e}")
            result_data.append(result_row)

        # Create DataFrame from results data
        results_df = pd.DataFrame(result_data)

        # Calculate and add average scores
        expected_metrics = ["answer_correctness", "faithfulness", "answer_relevancy"]

        # Create average row with the proper types
        avg_data = {}
        avg_data["question"] = "AVERAGE"

        # Calculate means for each metric using non-null values
        for metric in expected_metrics:
            if metric in results_df.columns:
                non_null_values = results_df[metric].dropna()
                if len(non_null_values) > 0:
                    avg_data[metric] = float(non_null_values.mean())
                    print(f"Average {metric}: {avg_data[metric]:.4f} from {len(non_null_values)} values")
                else:
                    avg_data[metric] = float('nan')  # Use NaN for missing values
                    print(f"No valid values for {metric}")

        # Add averages row to the DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([avg_data])], ignore_index=True)

        # Save results and generate visualization
        if output_json_path:
            # Save results to JSON
            save_results_to_json(results_df, output_json_path)

            # Generate visualization if not disabled
            if not skip_chart:
                try:
                    chart_path = generate_bar_chart(output_json_path)
                    if chart_path:
                        print(f"Chart generated: {chart_path}")
                except Exception as e:
                    print(f"Chart generation failed: {e}")

        return results_df

    except ImportError as e:
        raise ImportError(f"RAGAS components could not be imported. Please ensure RAGAS is installed: {e}")
    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation failed: {e}")
