import os
import re
import json
import faiss
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from config import JSON_PATH, FAISS_INDEX_PATH, CHUNKS_PATH

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 根目錄 LLM

json_path = os.path.join(BASE_DIR, JSON_PATH)
chunk_path = os.path.join(BASE_DIR, CHUNKS_PATH)
faiss_index_path = os.path.join(BASE_DIR, FAISS_INDEX_PATH)

# 確保路徑所在資料夾存在
os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
os.makedirs(os.path.dirname(chunk_path), exist_ok=True)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[。！？!?])', text)
    return [s.strip() for s in sentences if s.strip()]

def smart_chunk_text(text, target_len=500, min_len=100, overlap_sentences=1):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_len = sum(len(s) for s in current_chunk)
        if current_len + len(sentence) > target_len:
            if current_len >= min_len:
                chunks.append("".join(current_chunk))
                current_chunk = current_chunk[-overlap_sentences:] + [sentence]
            else:
                current_chunk.append(sentence)
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def build_index(json_path=json_path, index_path=faiss_index_path, chunk_path=chunk_path):
    # 載入資料
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sections = defaultdict(list)
    for d in data:
        title = d["title"]
        sections[title].append(d["text"])

    # embedding 模型（e5）
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    # 切 chunk
    chunk_records = []
    for title, paragraphs in sections.items():
        full_text = "\n".join(paragraphs)
        chunks = smart_chunk_text(full_text)

        for i, c in enumerate(chunks):
            chunk_records.append({
                "text": f"這段內容來自 Python 教學。\n主題：{title}\n內容：{c}",
                "title": title,
                "chunk_id": i
            })

    # e5 必須加 prefix
    all_chunks = ["passage: " + c["text"] for c in chunk_records]

    # 生成向量
    embeddings = model.encode(all_chunks)

    # cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 存檔
    faiss.write_index(index, index_path)
    np.save(chunk_path, np.array(all_chunks))

    print("向量庫建立完成")

if __name__ == "__main__":
    build_index()