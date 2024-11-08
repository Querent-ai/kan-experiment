from ekan import KAN as eKAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import random


# Load sentence transformer model for embedding generation
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Node Embedding Layer focusing on Entity-Level Embeddings
class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int, grid_size: int = 3, spline_order: int = 2):
        super(NodeEmbeddingLayer, self).__init__()
        self.node_transform = nn.Linear(num_features, hidden_dim)
        self.context_transform = nn.Linear(num_features, hidden_dim)
        self.update_transform = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.node_kan = eKAN([output_dim, output_dim], grid_size=grid_size, spline_order=spline_order)

    def forward(self, x, context_map, attention_weights_map):
        updated_embeddings = []
        for node_idx in range(x.size(0)):
            relevant_contexts = context_map.get(node_idx, [])
            relevant_attention_weights = attention_weights_map.get(node_idx, [])

            if relevant_contexts:
                context_stack = torch.stack(relevant_contexts)
                attention_weights = torch.tensor(relevant_attention_weights, dtype=torch.float).view(-1, 1)
                
                x_transformed = self.node_transform(x[node_idx].unsqueeze(0))
                context_transformed = self.context_transform(context_stack)
                context_weighted = context_transformed * attention_weights
                context_aggregated = context_weighted.mean(dim=0)
                
                updated_embedding = self.update_transform(x_transformed + context_aggregated)
                node_embedding = self.node_kan(updated_embedding).squeeze(0)
            else:
                node_embedding = x[node_idx]

            updated_embeddings.append(node_embedding)
        
        return torch.stack(updated_embeddings)

# Load trained model, embeddings, and unique entities list
def load_trained_model_and_embeddings(model_path, embeddings_path, unique_entities_path):
    model = NodeEmbeddingLayer(num_features=384, hidden_dim=384, output_dim=384)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    enhanced_entity_embeddings = np.load(embeddings_path)
    unique_entities = np.load(unique_entities_path, allow_pickle=True)  # Load in consistent order
    return model, enhanced_entity_embeddings, unique_entities

# Inference function to process query and return probability distribution
def process_query_with_probability_distribution(query, sentence_model, enhanced_entity_embeddings, unique_entities, top_n=5):
    # Generate embedding for the query
    query_embedding = torch.tensor(sentence_model.encode([query]), dtype=torch.float)

    # Compute cosine similarity with enhanced entity embeddings
    similarity_scores = cosine_similarity(query_embedding.detach().cpu().numpy(), enhanced_entity_embeddings).flatten()
    similarity_tensor = torch.tensor(similarity_scores)

    # Convert to probability distribution
    probability_distribution = F.softmax(similarity_tensor, dim=0).numpy()

    # Retrieve top N entities based on probability
    top_indices = np.argsort(probability_distribution)[::-1][:top_n]
    top_entities = [(unique_entities[i], probability_distribution[i]) for i in top_indices]

    # Output the top entities and probabilities
    print("Top relevant entities for the query:")
    for i, (entity, prob) in enumerate(top_entities):
        print(f"{i + 1}. Entity: {entity}, Probability: {prob:.4f}")

# Example usage
if __name__ == "__main__":
    # Paths to the saved model and embeddings
    model_path = "checkpoints/best_model.pt"
    embeddings_path = "checkpoints/enhanced_entity_embeddings.npy"
    unique_entities_path = "checkpoints/unique_entities.npy"

    # Load the trained model, enhanced embeddings, and unique entities
    model, enhanced_entity_embeddings, unique_entities = load_trained_model_and_embeddings(
        model_path, embeddings_path, unique_entities_path
    )

    # Query example
    query = "What are some of the important considerations for carbon storage in India?"

    # Run inference
    process_query_with_probability_distribution(query, sentence_model, enhanced_entity_embeddings, unique_entities)
