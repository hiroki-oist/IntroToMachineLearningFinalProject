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

# 議事録データを読み込む関数
def load_json_files(base_dir="./", start=1, end=13):
    agendas = set()
    contents = []
    print("*"*100)
    print("Embedding計算用のデータを読み込んでいます...")
    for i in range(start, end + 1):
        file_path = os.path.join(base_dir, f"Volume_{str(i).zfill(3)}", "full_text.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                agendas.update(data.get("agenda", []))
                contents.extend(data.get("content", []))
        else:
            print(f"ファイルが見つかりません: {file_path}")
    print("データの読み込みが完了しました。")
    return list(agendas), contents

# テキストをトークン数に基づいて分割する関数
def split_text(text, max_tokens):
    """
    トークン数がmax_tokensを超える場合、テキストを分割します。
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    tokens = encoder.encode(text)
    splits = []
    for i in range(0, len(tokens), max_tokens):
        split_tokens = tokens[i:i+max_tokens]
        split_text = encoder.decode(split_tokens)
        splits.append(split_text)
    return splits

# コンテンツを処理して、トークン数の制限を満たすように分割する
def process_contents(contents, max_content_tokens=8000):
    """
    コンテンツを分割して、テキストのリストと各テキストが元のコンテンツのどのインデックスに対応するかのマッピングを返します。
    """
    encoder = tiktoken.encoding_for_model(embedding_model)
    texts = []
    text_to_content_index = []
    content_index = 0

    for content in contents:
        content_text = json.dumps(content)
        content_tokens = len(encoder.encode(content_text))
        if content_tokens > max_content_tokens:
            # テキストを分割
            content_splits = split_text(content_text, max_content_tokens)
            texts.extend(content_splits)
            text_to_content_index.extend([content_index]*len(content_splits))
        else:
            texts.append(content_text)
            text_to_content_index.append(content_index)
        content_index += 1

    return texts, text_to_content_index

# テキストをチャンクに分割する
def chunk_texts(texts, max_tokens=8000):
    """
    トークン数がmax_tokensを超えないように、テキストをチャンクに分割します。
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

# Embeddingを計算して保存するメイン関数
def main():
    # データを読み込む
    agendas, contents = load_json_files()

    # コンテンツを処理
    texts, text_to_content_index = process_contents(contents, max_content_tokens=8000)
    text_chunks = chunk_texts(texts, max_tokens=8000)

    all_embeddings = []
    all_texts = []

    print("\nEmbeddingの計算を開始します。少々お待ちください。")
    total_chunks = len(text_chunks)
    for idx, chunk in enumerate(text_chunks):
        response = openai.Embedding.create(
            model=embedding_model,
            input=chunk
        )
        embeddings = [np.array(data["embedding"]) for data in response["data"]]
        all_embeddings.extend(embeddings)
        all_texts.extend(chunk)
        # 進捗状況を表示
        print(f"進捗: {idx+1}/{total_chunks} チャンクのEmbeddingが完了しました。")
    print("全てのEmbeddingが計算されました。")

    # 結果を保存
    with open("embeddings.pkl", "wb") as f:
        pickle.dump({
            "all_embeddings": all_embeddings,
            "all_texts": all_texts,
            "text_to_content_index": text_to_content_index,
            "contents": contents
        }, f)
    print("Embeddingの結果が 'embeddings.pkl' に保存されました。")

if __name__ == "__main__":
    main()
