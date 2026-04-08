import pandas as pd
from src.pipeline import RAGPipeline
from src.evaluation import recall_at_k, mrr, exact_match
from src.config import TEST_CSV, EVAL_CSV, HYBRID_ALPHA, CHUNKS_PATH, TOP_K

rag = RAGPipeline(chunks_path=CHUNKS_PATH, alpha=HYBRID_ALPHA)

df = pd.read_csv(TEST_CSV)
records = []

for _, row in df.iterrows():
    query = row["query"]
    ground_truth = row["answer"]
    
    answer, retrieved, reranked = rag.run(query, top_k=TOP_K)
    
    metrics = {
        "recall@10": recall_at_k(retrieved, ground_truth, k=10),
        "mrr": mrr(retrieved, ground_truth),
        "exact_match": exact_match(answer, ground_truth)
    }
    
    records.append({
        "query": query,
        "ground_truth": ground_truth,
        "answer": answer,
        **metrics
    })

pd.DataFrame(records).to_csv(EVAL_CSV, index=False)
print("Evaluation 完成，結果存到", EVAL_CSV)

