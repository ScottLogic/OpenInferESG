"""
RAGAS Evaluation Module
----------------------
Core functions for running RAGAS evaluations on question-answering data.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import FactualCorrectness, SemanticSimilarity
from ragas.metrics._nv_metrics import AnswerAccuracy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Find the project root (where .env is located)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


def create_ragas_dataset(data):
    """
    Create a RAGAS evaluation dataset from the input data.

    Args:
        data: List of dictionaries containing evaluation samples

    Returns:
        Tuple of (EvaluationDataset object, processed samples list)
    """

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
            reference=reference,  # Use either provided reference or first context
        )
        samples.append(eval_sample)

    print(f"Created {len(samples)} samples for evaluation")
    # Create a dataset using the RAGAS EvaluationDataset class
    return EvaluationDataset(samples=samples), samples


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


async def evaluate_with_ragas(
    jsonl_path: str, output_json_path: Optional[str] = None, skip_chart: bool = False
) -> pd.DataFrame:
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
        # Create LLM and embeddings for evaluation
        llm, embeddings_wrapper = create_ragas_llm()

        # Create dataset
        dataset, samples = create_ragas_dataset(data)

        # Define metrics to use for evaluation
        print("Configuring default RAGAS metrics: factual_correctness, semantic_similarity, answer_accuracy")
        metrics = [
            FactualCorrectness(llm=llm),
            SemanticSimilarity(embeddings=embeddings_wrapper),
            AnswerAccuracy(llm=llm),
        ]

        # Run the evaluation
        print("Running RAGAS evaluation (this may take a while)...")
        results = evaluate(dataset=dataset, metrics=metrics, llm=llm)

        try:
            print("Processing evaluation results...")
            # Define column mappings for RAGAS output to our desired output format
            columns = {
                "nv_accuracy": "answer_accuracy",
                "factual_correctness(mode=f1)": "factual_correctness",
                "semantic_similarity": "semantic_similarity",
                "user_input": "question",
            }

            # Print available columns for debugging
            available_columns = list(results.to_pandas().columns)
            print(f"Results DataFrame columns: {available_columns}")

            # Check that all required columns exist
            missing_columns = [col for col in columns.keys() if col not in available_columns]
            if missing_columns:
                raise ValueError(
                    f"Missing expected columns in RAGAS output: {missing_columns}. Update column mappings."
                )

            # Create DataFrame with only needed columns and renamed according to our convention
            selected_df = results.to_pandas()[list(columns.keys())]
            results_df = selected_df.rename(columns=columns)

            avg = results_df.select_dtypes(include=["number"]).mean(axis=0, numeric_only=True).to_dict()

            # Add question identifier
            avg["question"] = "AVERAGE"

            # Add averages row to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([avg])], ignore_index=True)
        except Exception as e:
            print(f"Could not process RAGAS results: {e}")
            raise

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

    except Exception as e:
        print(f"RAGAS evaluation failed: {str(e)}")
        raise RuntimeError(f"RAGAS evaluation failed: {e}")
