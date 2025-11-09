import numpy as np
from sentence_transformers import SentenceTransformer

def select_diverse_examples(query, candidates, embeddings, k=5, lambda_param=0.5):
    """
    Select diverse few-shot examples using Maximum Marginal Relevance.
    
    lambda_param controls the tradeoff:
    - 1.0 = pure relevance (ignores diversity)
    - 0.0 = pure diversity (ignores relevance)
    - 0.5 = balanced (usually a good starting point)
    """
    selected = []
    remaining = list(range(len(candidates)))
    
    # Compute similarity to query
    query_sim = np.dot(embeddings, query) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
    )
    
    for _ in range(k):
        if not selected:
            # First selection: most relevant to query
            best_idx = remaining[np.argmax(query_sim[remaining])]
        else:
            # Balance relevance and diversity
            mmr_scores = []
            for idx in remaining:
                relevance = query_sim[idx]
                
                # Max similarity to already selected examples
                similarity_to_selected = max([
                    np.dot(embeddings[idx], embeddings[s]) / (
                        np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[s])
                    )
                    for s in selected
                ])
                
                # MMR score: high relevance, low similarity to selected
                mmr = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected
                mmr_scores.append(mmr)
            
            best_idx = remaining[np.argmax(mmr_scores)]
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected

# Example usage
model = SentenceTransformer('all-MiniLM-L6-v2')

candidates = [
    "The movie was fantastic and entertaining",
    "I loved this film, it was amazing",
    "Great cinematography and acting",
    "Terrible movie, waste of time",
    "The plot was confusing and boring",
    "Best film I've seen this year",
    "Not my cup of tea, too slow",
    "Outstanding performances by the cast"
]

candidate_embeddings = model.encode(candidates)
query_embedding = model.encode("How was the movie?")

# Select 3 diverse examples
selected_indices = select_diverse_examples(
    query_embedding, 
    candidates, 
    candidate_embeddings,
    k=3,
    lambda_param=0.5
)

print("Selected examples:")
for idx in selected_indices:
    print(f"- {candidates[idx]}")