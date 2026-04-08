from sentence_transformers import SentenceTransformer, util

def recall_at_k(retrieved_docs, ground_truth, k=None):
    if k:
        retrieved_docs = retrieved_docs[:k]
    return any(ground_truth in doc for doc in retrieved_docs)

def mrr(retrieved_docs, ground_truth):
    for i, doc in enumerate(retrieved_docs):
        if ground_truth in doc:
            return 1 / (i + 1)
    return 0

def exact_match(answer, ground_truth):
    return ground_truth.strip() in answer.strip()

semantic_model = SentenceTransformer("intfloat/multilingual-e5-base")

def semantic_match(answer, ground_truth, threshold=0.7):
    emb1 = semantic_model.encode(answer, convert_to_tensor=True)
    emb2 = semantic_model.encode(ground_truth, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score > threshold

