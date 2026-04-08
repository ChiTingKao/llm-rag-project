from src.config import TOP_K
from src.pipeline import RAGPipeline


rag = RAGPipeline()

question = "Python 如何使用 for 迴圈？"
answer, retrieved, reranked = rag.run(question, top_k=TOP_K)

print("回答：", answer)
print("Top 3 Reranked Chunks：", reranked[:3])

