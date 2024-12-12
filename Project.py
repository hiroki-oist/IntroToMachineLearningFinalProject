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

# Define the embedding model and GPT model used in the script.
embedding_model = "text-embedding-ada-002"
gpt_model = "gpt-3.5-turbo"

# Function to load minute data from JSON files
# Reads agendas and content from structured JSON files for processing.
def load_json_files(base_dir="./", start=1, end=13):
    agendas = set()  # Store unique agendas using a set.
    contents = []    # Store all content.
    for i in range(start, end + 1):
        # Construct the file path based on the volume number.
        file_path = os.path.join(base_dir, f"Volume_{str(i).zfill(3)}", "full_text.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                agendas.update(data.get("agenda", []))  # Add agendas to the set.
                contents.extend(data.get("content", []))  # Add content to the list.
        else:
            # Print a warning if the file is not found.
            print(f"File not found: {file_path}")
    return list(agendas), contents  # Convert the set of agendas to a list.

# Function to translate agendas into English using OpenAI's GPT model
# This function creates a prompt to translate a list of agendas.
def translate_agendas(agendas):
    prompt = f"""
    Translate the following list of agendas into English. 
    Return the results in this format: {{'translated_agendas': ['translation1', 'translation2', ...]}}
    Agendas: {agendas}
    """
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        # Extract and parse the translated agendas from the GPT response.
        content = response["choices"][0]["message"]["content"]
        translated_agendas = ast.literal_eval(content)  # Convert the string response to a Python dictionary.
        return translated_agendas["translated_agendas"]
    except Exception as e:
        # Raise an exception if translation fails.
        raise Exception(f"Translation failed: {e}")

# Function to calculate similarity between a user question and precomputed content embeddings
# Uses cosine similarity to find the most relevant content.
def calculate_similarity(question, contents, all_embeddings, text_to_content_index):
    # Generate an embedding for the user's question.
    question_embedding = get_embedding(question)

    # Compute cosine similarities between the question embedding and all content embeddings.
    similarities = [cosine_similarity([question_embedding], [ce])[0][0] for ce in all_embeddings]

    # Identify the indices of the top 3 most similar content items.
    top_indices = np.argsort(similarities)[-3:][::-1]

    # Map the top indices back to the original content indices.
    top_content_indices = [text_to_content_index[i] for i in top_indices]

    # Collect unique contents while maintaining the top order.
    unique_content_indices = []
    seen_indices = set()
    for idx in top_content_indices:
        if idx not in seen_indices:
            seen_indices.add(idx)
            unique_content_indices.append(idx)
        if len(unique_content_indices) >= 10:  # Limit to 10 unique contents.
            break

    # Return the relevant content based on the calculated similarities.
    return [contents[i] for i in unique_content_indices]

# Function to generate embeddings for input text
# Calls OpenAI's embedding model to generate a dense vector representation of the input text.
def get_embedding(text):
    response = openai.Embedding.create(
        model=embedding_model,
        input=text
    )
    if "data" in response:
        # Return the embedding as a NumPy array.
        return np.array(response["data"][0]["embedding"])
    else:
        # Raise an exception if embedding generation fails.
        raise Exception(f"Embedding failed: {response}")

# Function to generate an answer based on top related content
# Uses GPT to create a detailed and context-aware answer to the user's question.
def generate_answer(question, top_contents):
    prompt = f"""
    Based on the following list of statements, answer the user's question in detail and as accurately as possible.
    Question: {question}
    Relevant data: {top_contents}
    """
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can read Japanese. Answer to the given question in about 100 words."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        # Return the generated answer from the GPT response.
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        # Raise an exception if answer generation fails.
        raise Exception(f"Answer generation failed: {e}")

# Main function to run the application
# Manages the workflow of loading data, translating agendas, and answering user questions.
def main():
    print("*"*100)
    print("Hello, this is a demo of ChatGPT-powered commentator of committees in the Japanese parliament!\n\nI can currently answer based on minutes during last year's meetings from the safety and security committee.")
    print("*"*100)

    # Load agendas and contents for display and processing.
    agendas, contents = load_json_files()

    # Translate agendas from Japanese to English for user display.
    translated_agendas = translate_agendas(agendas)
    print("Discussion that was made last year are below. Ask me any questions! :)")
    print("Agenda: ")
    for agenda in translated_agendas:
        print(f"- {agenda}")

    # Load precomputed embeddings from a saved file.
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        all_embeddings = data["all_embeddings"]
        all_texts = data["all_texts"]
        text_to_content_index = data["text_to_content_index"]
        contents = data["contents"]

    # Accept a question from the user.
    print("*" * 100)
    question = input("Question: ")
    print("Making answer, please wait...")

    # Find the most relevant content using similarity matching.
    top_contents = calculate_similarity(question, contents, all_embeddings, text_to_content_index)
    print("\nI have collected the related discussions from the minutes! Making answer...\n")

    # Generate a detailed answer based on the top related content.
    answer = generate_answer(question, top_contents)
    print("\nAnswer: ")
    print(answer)

# Entry point for the script execution
if __name__ == "__main__":
    main()
