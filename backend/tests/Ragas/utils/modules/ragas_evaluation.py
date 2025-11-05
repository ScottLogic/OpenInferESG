"""
RAGAS Evaluation Module
----------------------
Core functions for running RAGAS evaluations on question-answering data.
"""

import os
from pathlib import Path
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import ChatOpenAI
from ragas.metrics import AnswerAccuracy, SemanticSimilarity, FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from .ragas_utils import load_jsonl_data


# Find the project root (where .env is located)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def aggregate_usage_field(usage_records, field_name: str, default_value=0):
    """
    Aggregate a specific field from usage records.

    Args:
        usage_records: Collection of usage record objects
        field_name: Name of the field to aggregate
        default_value: Default value to use if field is missing

    Returns:
        Sum of the specified field across all valid records
    """
    return sum(record.get(field_name, default_value) for record in usage_records if isinstance(record, dict))


def create_ragas_dataset(data):
    """
    Create a RAGAS evaluation dataset from the input data.

    Args:
        data: List of dictionaries containing evaluation samples

    Returns:
        Tuple of (EvaluationDataset object, processed samples list, processed data list)
    """

    samples = []
    processed_data = []
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
            response=sample.get("response", ""),
            reference=reference,  # Use either provided reference or first context
        )
        samples.append(eval_sample)
        processed_data.append(sample)  # Keep track of the original data for this sample

    print(f"Created {len(samples)} samples for evaluation")
    # Create a dataset using the RAGAS EvaluationDataset class
    return EvaluationDataset(samples=samples), samples, processed_data


def create_ragas_llm():
    """
    Create and configure the LLM for RAGAS evaluation.

    Returns:
        tuple: (LangchainLLMWrapper, LangchainEmbeddingsWrapper)
    """
    import sys

    # Use the specified OpenAI model from .env or default to gpt-4o
    model_name = os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o")

    # Get API key from the root .env file
    api_key = os.environ.get("OPENAI_KEY")

    if not api_key:
        print("OpenAI API key not set. Please set OPENAI_KEY in the root .env file.")
        sys.exit(1)

    # Set OPENAI_API_KEY to support libraries that require this specific env variable
    os.environ["OPENAI_API_KEY"] = api_key

    print(f"API Key found for ChatOpenAI: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    print(f"Using model: {model_name}")

    # Create the ChatOpenAI model and wrap it with LangchainLLMWrapper
    chat_model = ChatOpenAI(
        model=model_name,
        temperature=0,  # Use temperature=0 for more consistent evaluations
    )

    # Initialize the OpenAIEmbeddings - will use the environment variable
    embeddings = OpenAIEmbeddings()

    # Create RAGAS embeddings wrapper
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return LangchainLLMWrapper(chat_model), ragas_embeddings


async def evaluate_with_ragas(jsonl_path: str) -> pd.DataFrame:
    """
    Evaluate responses using RAGAS metrics

    Args:
        jsonl_path: Path to the input JSONL file with responses

    Returns:
        DataFrame with evaluation results
    """
    # Import locally to avoid circular imports

    print("Setting up RAGAS evaluation...")
    print(f"Loading data from {jsonl_path}...")
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} samples for evaluation")

    try:
        # Create LLM and embeddings for evaluation
        llm, embeddings_wrapper = create_ragas_llm()

        # Create dataset
        dataset, samples, processed_data = create_ragas_dataset(data)

        # Define metrics to use for evaluation
        print("Configuring default RAGAS metrics: semantic_similarity,factual_correctness, answer_accuracy")
        metrics = [
            SemanticSimilarity(),
            FactualCorrectness(llm=llm),
            AnswerAccuracy(llm=llm),
        ]

        # Run the evaluation
        print("Running RAGAS evaluation (this may take a while)...")
        results = evaluate(dataset=dataset, metrics=metrics, llm=llm)
        try:
            print("Processing evaluation results including llm_usage if present...")
            # Define expected metrics for alignment and output naming
            expected_metrics = [
                ("factual_correctness(mode=f1)", "factual_correctness"),
                ("nv_accuracy", "answer_accuracy"),
                ("semantic_similarity", "semantic_similarity"),
            ]

            df = results.to_pandas()
            available_columns = list(df.columns)
            print(f"Results DataFrame columns: {available_columns}")

            # Verify required columns
            missing = [raw for raw, _ in expected_metrics if raw not in available_columns]
            if missing:
                raise ValueError(
                    f"Missing expected columns in RAGAS output: {missing}. Update column mappings or metric extraction."
                )

            # Build per-sample rows with metrics and attach llm_usage from original processed_data if present
            rows = []
            for idx in range(len(df)):
                row_dict = {
                    "question": df.loc[idx, "user_input"]
                    if "user_input" in df.columns
                    else processed_data[idx].get("user_input", "")
                }
                for raw, mapped in expected_metrics:
                    row_dict[mapped] = df.loc[idx, raw]
                # Attach llm_usage if supplied in original input sample
                if "llm_usage" in processed_data[idx]:
                    for key, val in processed_data[idx]["llm_usage"].items():
                        row_dict[key] = val
                rows.append(row_dict)

            results_df = pd.DataFrame(rows)

            # Compute averages manually (exclude llm_usage from mean calc)
            avg_data: dict = {"question": "AVERAGE"}
            metric_names = [mapped for _, mapped in expected_metrics]
            for metric in metric_names:
                if metric in results_df.columns:
                    non_null = results_df[metric].dropna()
                    avg_data[metric] = float(non_null.mean()) if len(non_null) else None

            # Aggregate llm usage across samples
            if "llm_usage" in results_df.columns:
                usage_records = results_df["llm_usage"].dropna()
                if len(usage_records) > 0:
                    total_prompt_tokens = aggregate_usage_field(usage_records, "prompt_tokens")
                    total_completion_tokens = aggregate_usage_field(usage_records, "completion_tokens")
                    total_tokens = aggregate_usage_field(usage_records, "total_tokens")
                    total_duration = aggregate_usage_field(usage_records, "duration_seconds")
                    avg_data["llm_usage"] = {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens,
                        "total_duration_seconds": total_duration,
                    }

            # Append average row
            results_df = pd.concat([results_df, pd.DataFrame([avg_data])], ignore_index=True)
        except Exception as e:
            print(f"Could not process RAGAS results with llm_usage: {e}")
            raise

        return results_df

    except Exception as e:
        print(f"RAGAS evaluation failed: {str(e)}")
        raise RuntimeError(f"RAGAS evaluation failed: {e}")
