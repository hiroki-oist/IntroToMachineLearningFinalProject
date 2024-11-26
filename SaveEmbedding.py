import os
import json
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import time
import tiktoken
import pickle

# Load OpenAI API settings
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"

# Function to load minute data
def load_json_files(base_dir="./", start=1, end=13):
    agendas = set()
    contents = []
    print("*" * 100)
    print("Loading data for embedding calculations...")
    for i in range(start, end + 1):
        file_path = os.path.join(base_dir, f"Volume_{str(i).zfill(3)}", "full_text.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                agendas.update(data.get("agenda", []))
                contents.extend(data.get("content", []))
        else:
            print(f"File not found: {file_path}")
    print("Data loading is complete.")
    return list(agendas), contents

# Function to split text based on token count
def split_text(text, max_tokens):
    """
    Splits text into smaller parts if it exceeds max_tokens.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    tokens = encoder.encode(text)
    splits = []
    for i in range(0, len(tokens), max_tokens):
        split_tokens = tokens[i:i+max_tokens]
        split_text = encoder.decode(split_tokens)
        splits.append(split_text)
    return splits

# Process contents to ensure token limit compliance
def process_contents(contents, max_content_tokens=8000):
    """
    Processes contents, splitting text where necessary, and returns a list of texts 
    and a mapping of each text to its original content index.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    texts = []
    text_to_content_index = []
    content_index = 0

    for content in contents:
        content_text = json.dumps(content)
        content_tokens = len(encoder.encode(content_text))
        if content_tokens > max_content_tokens:
            # Split text if it exceeds max token count
            content_splits = split_text(content_text, max_content_tokens)
            texts.extend(content_splits)
            text_to_content_index.extend([content_index] * len(content_splits))
        else:
            texts.append(content_text)
            text_to_content_index.append(content_index)
        content_index += 1

    return texts, text_to_content_index

# Split texts into chunks
def chunk_texts(texts, max_tokens=8000):
    """
    Splits texts into chunks ensuring the token count does not exceed max_tokens.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for text in texts:
        text_tokens = len(encoder.encode(text))
        if current_tokens + text_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(text)
        current_tokens += text_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Main function to calculate and save embeddings
def main():
    # Load data
    agendas, contents = load_json_files()

    # Process contents
    texts, text_to_content_index = process_contents(contents, max_content_tokens=8000)
    text_chunks = chunk_texts(texts, max_tokens=8000)

    all_embeddings = []
    all_texts = []

    print("\nStarting embedding calculations. Please wait.")
    total_chunks = len(text_chunks)
    for idx, chunk in enumerate(text_chunks):
        response = openai.Embedding.create(
            model=embedding_model,
            input=chunk
        )
        embeddings = [np.array(data["embedding"]) for data in response["data"]]
        all_embeddings.extend(embeddings)
        all_texts.extend(chunk)
        # Display progress
        print(f"Progress: {idx+1}/{total_chunks} chunks have been processed for embeddings.")
    print("All embeddings have been calculated.")

    # Save results
    with open("embeddings.pkl", "wb") as f:
        pickle.dump({
            "all_embeddings": all_embeddings,
            "all_texts": all_texts,
            "text_to_content_index": text_to_content_index,
            "contents": contents
        }, f)
    print("Embedding results have been saved to 'embeddings.pkl'.")

if __name__ == "__main__":
    main()
