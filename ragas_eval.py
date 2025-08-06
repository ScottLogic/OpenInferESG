import os
import asyncio
from dotenv import load_dotenv

#Load OpenAI API Key
load_dotenv()
open_api_key = os.getenv("OPENAI_KEY") 
os.environ["OPENAI_API_KEY"] = open_api_key if open_api_key else ""

#Set up LLM and Embeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

# Initialize LLM and Embeddings for Evaluation
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Create a Sample Input
test_data = {
    "user_input": "What is the percentage of reduction of the baseline emissions for Scope 1 and Scope 2?",
    "response": "Scott Logic has set a target of to achieve a reduction in their baseline emissions for Scope 1 and Scope 2 by the year 2026",
}

metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
test_data = SingleTurnSample(**est_data)

# Create a Single Evaluation Sample
test_data = SingleTurnSample(**test_data)


# Define a Custom Metric
async def evaluate_summary():
    result = await metric.single_turn_ascore(test_data)
    print(f"Result: {result}")

# Run the async function
asyncio.run(evaluate_summary())
