import os
import asyncio
import openai
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas import EvaluationDataset, SingleTurnSample
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

#Load OpenAI API Key
load_dotenv()
open_api_key = os.getenv("OPENAI_KEY") 
if not open_api_key:
    raise RuntimeError("OPENAI_KEY environment variable not found. Please set it in your environment or .env file.")
os.environ["OPENAI_API_KEY"] = open_api_key

client = AsyncOpenAI(api_key=open_api_key)

filepath = r"C:\Users\afonseca\Downloads\report.md.txt"

#Loads the document from a file
def load_document(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    elif path.endswith(".pdf"):
        text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    else:
        raise ValueError("Unsupported file format. Use .txt or .pdf")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Remove duplicate questions
def remove_duplicate_questions(qa_pairs, threshold=0.9):
    questions = [pair["question"] for pair in qa_pairs]
    embeddings = embedding_model.encode(questions)

    to_remove = set()
    for i in range(len(questions)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(questions)):
            if j in to_remove:
                continue
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > threshold:
                to_remove.add(j)

    cleaned_qa_pairs = [pair for idx, pair in enumerate(qa_pairs) if idx not in to_remove]
    print(f"\n Removed {len(to_remove)} duplicate questions.")
    return cleaned_qa_pairs

# Generate Q&A pairs from full context

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "../prompts/generate_qa_prompt.txt")

# Generate Q&A pairs from full context
async def generate_qa_pairs(context, num_pairs=20):
    prompt_template = load_prompt(PROMPT_PATH)
    prompt = prompt_template.format(context=context, num_pairs=num_pairs)
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        import json
        qa_pairs = json.loads(content)
        return qa_pairs
    except Exception as e:
        print("Failed to parse LLM output as JSON:", e)
        print("Raw response:", content)
        return []

#Generate 20 Q&A pairs
async def main():
    context = load_document(filepath)
    qa_pairs = await generate_qa_pairs(context, num_pairs=20)

    if not qa_pairs:
        print("No valid QA pairs generated.")
        return
    
#Evaluate the generated Q&A pairs
    dataset = EvaluationDataset([
        SingleTurnSample(
            user_input=pair["question"],
            retrieved_contexts=[context],
            response=pair["answer"]
        )
        for pair in qa_pairs
    ])

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    print("\n Evaluation Results:\n")
    for i, pair in enumerate(qa_pairs):
        print(f" Sample {i+1}")
        print(f"   - Question              : {pair['question']}")
        print(f"   - Answer                : {pair['answer']}")
        print(f"   - Faithfulness Score    : {results['faithfulness'][i]:.2f}")
        print(f"   - Answer Relevancy Score: {results['answer_relevancy'][i]:.2f}\n")


if __name__ == "__main__":
    asyncio.run(main())
