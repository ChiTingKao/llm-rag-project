import faiss
import requests
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import CHUNKS_PATH, HYBRID_ALPHA, LLM_MODEL, LLM_ENDPOINT, STREAM

class RAGPipeline:
    def __init__(self, chunks_path=CHUNKS_PATH, alpha=HYBRID_ALPHA):
        self.chunks = np.load(chunks_path, allow_pickle=True)
        self.alpha = alpha
        
        # BM25
        tokenized_chunks = [c.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Vector
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
        embeddings = self.embed_model.encode(["passage: "+c for c in self.chunks])
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        self.vector_index.add(embeddings)
        
        # Reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(self, query, top_k=10):
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Vector
        query_vec = self.embed_model.encode([f"query: {query}"])
        faiss.normalize_L2(query_vec)
        D, I = self.vector_index.search(query_vec, len(self.chunks))
        vector_scores = np.zeros(len(self.chunks))
        for rank, idx in enumerate(I[0]):
            vector_scores[idx] = D[0][rank]
        
        # Hybrid
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * vector_scores
        top_idx = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]

    def rerank(self, query, retrieved):
        pairs = [[query, c] for c in retrieved]
        scores = self.reranker.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, retrieved), reverse=True)]
        return reranked

    def generate(self, query, context_chunks):
        context = "\n\n".join(context_chunks)
        prompt = f"""
你是一個專業的python助教，
請根據提供的多段內容，整合資訊回答問題。
如果答案需要多個概念，請綜合說明。
回答後請附上引用的原文片段。

如果內容不足，請回答「我不知道」。

內容：
{context}

問題：
{query}
"""
        try:
            response = requests.post(
                LLM_ENDPOINT,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": STREAM}
            )
            return response.json()["response"]
        except Exception as e:
            return f"Ollama 呼叫失敗: {e}"

    def run(self, query, top_k=None):
        if top_k is None:
            from src.config import TOP_K
            top_k = TOP_K
        retrieved = self.retrieve(query, top_k=top_k)
        reranked = self.rerank(query, retrieved)
        answer = self.generate(query, reranked[:3])
        return answer, retrieved, reranked