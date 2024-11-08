import torch
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load unique entities, enhanced embeddings, and context mappings
unique_entities = np.load("checkpoints/unique_entities.npy", allow_pickle=True).tolist()
enhanced_entity_embeddings = np.load("checkpoints/enhanced_entity_embeddings.npy")
context_mapping = np.load("checkpoints/context_mapping.npy", allow_pickle=True).item()

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Inference function with contextual relevance identification
def process_query_with_contextual_relevance(query, enhanced_entity_embeddings, unique_entities, context_mapping, top_n=5, top_contexts_per_entity=5):
    # Generate embedding for the query
    query_embedding = torch.tensor(sentence_model.encode([query]), dtype=torch.float)
    similarity_scores = cosine_similarity(query_embedding.detach().cpu().numpy(), enhanced_entity_embeddings).flatten()
    similarity_tensor = torch.tensor(similarity_scores)
    probability_distribution = F.softmax(similarity_tensor, dim=0).numpy()

    # Get top N entities based on similarity
    top_indices = np.argsort(probability_distribution)[::-1][:top_n]
    top_entities = [(unique_entities[i], probability_distribution[i]) for i in top_indices]

    # Display top entities and retrieve influential context sentences
    print("Top relevant entities for the query:")
    for i, (entity, prob) in enumerate(top_entities):
        print(f"\n{i + 1}. Entity: {entity}, Probability: {prob:.4f}")

        # Retrieve the highest-scoring context sentences for the entity
        entity_index = unique_entities.index(entity)  # Find the index of the entity
        if entity_index in context_mapping:
            # Sort context sentences by attention score and select top contexts
            top_contexts = sorted(context_mapping[entity_index], key=lambda x: x[1], reverse=True)[:top_contexts_per_entity]
            print("Most influential contexts for this entity:")
            for j, (ctx_sentence, score) in enumerate(top_contexts):
                print(f"  {j + 1}. Context: \"{ctx_sentence}\", Attention Score: {score:.4f}")

# Example Query
query = "What are some of the important considerations for carbon storage in India?"
process_query_with_contextual_relevance(query, enhanced_entity_embeddings, unique_entities, context_mapping)
