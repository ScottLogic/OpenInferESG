import os
import asyncio
from dotenv import load_dotenv

#Load OpenAI API Key
load_dotenv()
open_api_key = os.getenv("OPENAI_KEY") 
if not open_api_key:
    raise RuntimeError("OPENAI_KEY environment variable not found. Please set it in your environment or .env file.")
os.environ["OPENAI_API_KEY"] = open_api_key

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

# Create a Single Evaluation Sample
single_turn_sample = SingleTurnSample(user_input=test_data["user_input"], response=test_data["response"])


# Define a Custom Metric
async def evaluate_summary(data):
    result = await metric.single_turn_ascore(data)
    print(f"Result: {result}")

# Run the async function
asyncio.run(evaluate_summary(single_turn_sample))
