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
# This section loads the OpenAI API key from the .env file.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the embedding model to be used.
embedding_model = "text-embedding-ada-002"

# Function to load minute data
# Reads agendas and content from JSON files stored in a structured directory.
def load_json_files(base_dir="./", start=1, end=13):
    agendas = set()  # Store unique agendas.
    contents = []    # Store all content.
    print("*" * 100)
    print("Loading data for embedding calculations...")
    for i in range(start, end + 1):
        # Construct the file path for each volume.
        file_path = os.path.join(base_dir, f"Volume_{str(i).zfill(3)}", "full_text.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                agendas.update(data.get("agenda", []))  # Add agendas to the set.
                contents.extend(data.get("content", []))  # Append content to the list.
        else:
            # Notify if the file is not found.
            print(f"File not found: {file_path}")
    print("Data loading is complete.")
    return list(agendas), contents  # Convert the set of agendas to a list.

# Function to split text based on token count
# Ensures that text segments do not exceed the token limit for the embedding model.
def split_text(text, max_tokens):
    """
    Splits text into smaller parts if it exceeds max_tokens.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)  # Initialize tokenizer for the embedding model.
    tokens = encoder.encode(text)  # Tokenize the text.
    splits = []
    for i in range(0, len(tokens), max_tokens):
        split_tokens = tokens[i:i+max_tokens]  # Extract a chunk of tokens.
        split_text = encoder.decode(split_tokens)  # Decode tokens back into text.
        splits.append(split_text)
    return splits

# Process contents to ensure token limit compliance
# Splits content if necessary and tracks the mapping of splits to their original indices.
def process_contents(contents, max_content_tokens=8000):
    """
    Processes contents, splitting text where necessary, and returns a list of texts 
    and a mapping of each text to its original content index.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)  # Initialize tokenizer for the embedding model.
    texts = []
    text_to_content_index = []
    content_index = 0

    for content in contents:
        content_text = json.dumps(content)  # Convert content to JSON string.
        content_tokens = len(encoder.encode(content_text))  # Count tokens in the content.
        if content_tokens > max_content_tokens:
            # Split text if it exceeds the max token count.
            content_splits = split_text(content_text, max_content_tokens)
            texts.extend(content_splits)
            text_to_content_index.extend([content_index] * len(content_splits))
        else:
            texts.append(content_text)
            text_to_content_index.append(content_index)
        content_index += 1

    return texts, text_to_content_index

# Split texts into chunks
# Groups smaller texts together into chunks that fit within the token limit.
def chunk_texts(texts, max_tokens=8000):
    """
    Splits texts into chunks ensuring the token count does not exceed max_tokens.
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for text in texts:
        text_tokens = len(encoder.encode(text))  # Count tokens in the current text.
        if current_tokens + text_tokens > max_tokens:
            # Save the current chunk and start a new one if token limit is exceeded.
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(text)
        current_tokens += text_tokens

    if current_chunk:
        chunks.append(current_chunk)  # Add the final chunk if it exists.

    return chunks

# Main function to calculate and save embeddings
# Manages the workflow of data loading, processing, and embedding generation.
def main():
    # Load data
    agendas, contents = load_json_files()

    # Process contents to ensure token limit compliance.
    texts, text_to_content_index = process_contents(contents, max_content_tokens=8000)
    text_chunks = chunk_texts(texts, max_tokens=8000)

    all_embeddings = []
    all_texts = []

    print("\nStarting embedding calculations. Please wait.")
    total_chunks = len(text_chunks)
    for idx, chunk in enumerate(text_chunks):
        # Generate embeddings for each chunk using OpenAI's API.
        response = openai.Embedding.create(
            model=embedding_model,
            input=chunk
        )
        embeddings = [np.array(data["embedding"]) for data in response["data"]]  # Extract embeddings.
        all_embeddings.extend(embeddings)  # Store embeddings.
        all_texts.extend(chunk)  # Store the processed text.
        # Display progress to the user.
        print(f"Progress: {idx+1}/{total_chunks} chunks have been processed for embeddings.")
    print("All embeddings have been calculated.")

    # Save results to a file for later use.
    with open("embeddings.pkl", "wb") as f:
        pickle.dump({
            "all_embeddings": all_embeddings,
            "all_texts": all_texts,
            "text_to_content_index": text_to_content_index,
            "contents": contents
        }, f)
    print("Embedding results have been saved to 'embeddings.pkl'.")

# Entry point for the script execution
if __name__ == "__main__":
    main()