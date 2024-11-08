from ekan import KAN as eKAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import random

# Set deterministic behavior for reproducibility
def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic_seed(42)

# Define Node Embedding Layer with Context Tracking
class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int, grid_size: int = 3, spline_order: int = 2):
        super(NodeEmbeddingLayer, self).__init__()
        self.node_transform = nn.Linear(num_features, hidden_dim)
        self.context_transform = nn.Linear(num_features, hidden_dim)
        self.update_transform = nn.Linear(hidden_dim, output_dim)
        self.node_kan = eKAN([output_dim, output_dim], grid_size=grid_size, spline_order=spline_order)
        self.context_mapping = {}  # Dictionary to track contexts

    def forward(self, x, context_map, attention_weights_map):
        updated_embeddings = []
        self.context_mapping = {}  # Clear previous contexts

        for node_idx in range(x.size(0)):
            relevant_contexts = context_map.get(node_idx, [])
            relevant_attention_weights = attention_weights_map.get(node_idx, [])
            context_info = []

            if relevant_contexts:
                context_stack = torch.stack(relevant_contexts)
                attention_weights = torch.tensor(relevant_attention_weights, dtype=torch.float).view(-1, 1)
                
                x_transformed = self.node_transform(x[node_idx].unsqueeze(0))
                context_transformed = self.context_transform(context_stack)
                context_weighted = context_transformed * attention_weights
                context_aggregated = context_weighted.mean(dim=0)
                
                updated_embedding = self.update_transform(x_transformed + context_aggregated)
                node_embedding = self.node_kan(updated_embedding).squeeze(0)

                for sentence, weight in zip(relevant_contexts, attention_weights):
                    context_info.append((sentence, weight.item()))

                self.context_mapping[node_idx] = context_info
            else:
                node_embedding = x[node_idx]

            updated_embeddings.append(node_embedding)
        
        return torch.stack(updated_embeddings)

# Define contrastive loss function
def contrastive_loss(embedding_a, embedding_b, similarity_label, margin=0.3):
    if embedding_a.dim() == 1:
        embedding_a = embedding_a.unsqueeze(0)
    if embedding_b.dim() == 1:
        embedding_b = embedding_b.unsqueeze(0)
    
    cosine_sim = F.cosine_similarity(embedding_a, embedding_b)
    positive_loss = similarity_label * (1 - cosine_sim)
    negative_loss = (1 - similarity_label) * torch.clamp(cosine_sim - margin, min=0)
    return (positive_loss + negative_loss).mean()

# Training function to save enhanced embeddings and context mappings
def train_and_save_embeddings_with_checkpoints(model, entity_embeddings, context_map, attention_weights_map, num_epochs=1, save_path="checkpoints"):
    os.makedirs(save_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for idx in range(entity_embeddings.size(0)):
            embedding_a = entity_embeddings[idx].detach().clone().requires_grad_(True)
            for context_idx in range(len(context_map[idx])):
                embedding_b = context_map[idx][context_idx].detach().clone().requires_grad_(True)
                
                similarity_label = torch.tensor(1.0 if context_idx % 2 == 0 else 0.0)
                loss = contrastive_loss(embedding_a, embedding_b, similarity_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        enhanced_entity_embeddings = model(entity_embeddings, context_map, attention_weights_map)
        
        # Save both enhanced embeddings and context mappings
        np.save(os.path.join(save_path, "enhanced_entity_embeddings.npy"), enhanced_entity_embeddings.cpu().numpy())
        
        # Convert context_mapping to a structured list before saving
        structured_context_mapping = {k: [(ctx.numpy().tolist(), score) for ctx, score in v] for k, v in model.context_mapping.items()}
        np.save(os.path.join(save_path, "context_mapping.npy"), structured_context_mapping)


# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess data
semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge.xlsx")
embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge.xlsx")


# Generate unique entity embeddings
unique_entities = sorted(set(semantic_knowledge_df['subject'].unique().tolist() + 
                             semantic_knowledge_df['object'].unique().tolist()))
entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}

# Save unique entities list for consistent mapping during inference
np.save("checkpoints/unique_entities.npy", unique_entities)

# Create context and attention mappings
context_map = {i: [] for i in range(len(unique_entities))}
attention_weights_map = {i: [] for i in range(len(unique_entities))}
event_to_attention = embedded_knowledge_df.set_index('event_id')['score'].to_dict()

for _, row in semantic_knowledge_df.iterrows():
    subject, object_, sentence, event_id = row['subject'], row['object'], row['sentence'], row['event_id']
    subj_idx, obj_idx = entity_mapping[subject], entity_mapping[object_]
    sentence_embedding = torch.tensor(sentence_model.encode([sentence])[0], dtype=torch.float)
    attention_weight = event_to_attention.get(event_id, 1.0)

    context_map[subj_idx].append(sentence_embedding)
    context_map[obj_idx].append(sentence_embedding)
    attention_weights_map[subj_idx].append(attention_weight)
    attention_weights_map[obj_idx].append(attention_weight)

# Initialize and train the model
model = NodeEmbeddingLayer(num_features=384, hidden_dim=384, output_dim=384)
train_and_save_embeddings_with_checkpoints(model, entity_embeddings, context_map, attention_weights_map)

