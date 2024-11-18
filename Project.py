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

# OpenAI API設定を読み込み
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"
gpt_model = "gpt-3.5-turbo"

# 議事録データを読み込む関数（アジェンダのため）
def load_json_files(base_dir="./", start=1, end=13):
    agendas = set()
    contents = []
    for i in range(start, end + 1):
        file_path = os.path.join(base_dir, f"Volume_{str(i).zfill(3)}", "full_text.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                agendas.update(data.get("agenda", []))
                contents.extend(data.get("content", []))
        else:
            print(f"ファイルが見つかりません: {file_path}")
    return list(agendas), contents

# アジェンダを翻訳する関数
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
        content = response["choices"][0]["message"]["content"]
        translated_agendas = ast.literal_eval(content)
        return translated_agendas["translated_agendas"]
    except Exception as e:
        raise Exception(f"Translation failed: {e}")

# 質問とコンテンツ間の類似度を計算する関数
def calculate_similarity(question, contents, all_embeddings, text_to_content_index):
    # 質問のEmbeddingを生成
    question_embedding = get_embedding(question)

    # 事前に計算したEmbeddingを使用して類似度を計算
    similarities = [cosine_similarity([question_embedding], [ce])[0][0] for ce in all_embeddings]

    # 類似度に基づいて上位のインデックスを取得
    top_indices = np.argsort(similarities)[-3:][::-1]

    # 元のコンテンツのインデックスにマッピング
    top_content_indices = [text_to_content_index[i] for i in top_indices]

    # ユニークなコンテンツを収集
    unique_content_indices = []
    seen_indices = set()
    for idx in top_content_indices:
        if idx not in seen_indices:
            seen_indices.add(idx)
            unique_content_indices.append(idx)
        if len(unique_content_indices) >= 10:
            break

    # 関連するコンテンツを返す
    return [contents[i] for i in unique_content_indices]

# Embeddingを生成する関数
def get_embedding(text):
    response = openai.Embedding.create(
        model=embedding_model,
        input=text
    )
    if "data" in response:
        return np.array(response["data"][0]["embedding"])
    else:
        raise Exception(f"Embedding failed: {response}")

# 回答を生成する関数
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
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Answer generation failed: {e}")

# メイン関数
def main():
    print("*"*100)
    print("Hello, this is a demo of ChatGPT-powered commentator of committees in the Japanese parliament!\n\nI can currently answer based on minutes during last year's meetings from the safety and security committee.")
    print("*"*100)
    # アジェンダとコンテンツを読み込む（アジェンダの表示のため）
    agendas, contents = load_json_files()

    # アジェンダを翻訳
    translated_agendas = translate_agendas(agendas)
    print("Discussion that was made last year are below. Ask me any questions! :)")
    print("Agenda: ")
    for agenda in translated_agendas:
        print(f"- {agenda}")

    # 事前に計算したEmbeddingを読み込む
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        all_embeddings = data["all_embeddings"]
        all_texts = data["all_texts"]
        text_to_content_index = data["text_to_content_index"]
        contents = data["contents"]

    # ユーザーの質問を受け付ける
    print("*" * 100)
    question = input("Question: ")
    print("Making answer, please wait...")

    # 類似度に基づいて関連するコンテンツを取得
    top_contents = calculate_similarity(question, contents, all_embeddings, text_to_content_index)
    print("\nI have collected the related discussions from the minutes! Making answer...\n")

    # 回答を生成
    answer = generate_answer(question, top_contents)
    print("\nAnswer: ")
    print(answer)

if __name__ == "__main__":
    main()
