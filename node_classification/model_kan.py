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

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic_seed(42)

# Contrastive Loss Function with Summary Debugging
def contrastive_loss(embedding_a, embedding_b, similarity_label, margin=0.3):
    if embedding_a.dim() == 1:
        embedding_a = embedding_a.unsqueeze(0)
    if embedding_b.dim() == 1:
        embedding_b = embedding_b.unsqueeze(0)
    
    cosine_sim = F.cosine_similarity(embedding_a, embedding_b)
    positive_loss = similarity_label * (1 - cosine_sim)
    negative_loss = (1 - similarity_label) * torch.clamp(cosine_sim - margin, min=0)
    loss = (positive_loss + negative_loss).mean()
    
    return loss, positive_loss.mean().item(), negative_loss.mean().item()

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
                # context_weighted = context_transformed * attention_weights
                context_weighted = context_transformed * 1
                context_aggregated = context_weighted.mean(dim=0)
                
                updated_embedding = self.update_transform(x_transformed + context_aggregated)
                node_embedding = self.node_kan(updated_embedding).squeeze(0)
            else:
                node_embedding = x[node_idx]

            updated_embeddings.append(node_embedding)
        
        return torch.stack(updated_embeddings)

# Training Function with Debugging Summary
def train_and_save_embeddings_with_checkpoints(
    model, entity_embeddings, context_map, attention_weights_map, num_epochs=1, save_path="checkpoints"
):
    os.makedirs(save_path, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_positive_loss, total_negative_loss = 0, 0, 0
        num_pairs = 0

        for idx in range(entity_embeddings.size(0)):
            embedding_a = entity_embeddings[idx].detach().clone().requires_grad_(True)
            for context_idx in range(len(context_map[idx])):
                embedding_b = context_map[idx][context_idx].detach().clone().requires_grad_(True)
                
                similarity_label = torch.tensor(1.0 if context_idx % 2 == 0 else 0.0)

                # Calculate contrastive loss
                loss, pos_loss, neg_loss = contrastive_loss(embedding_a, embedding_b, similarity_label)
                
                # Add regularization loss from KAN
                regularization_loss = model.node_kan.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)
                total_loss_with_reg = loss + regularization_loss  # Combined loss
                
                optimizer.zero_grad()
                total_loss_with_reg.backward()
                optimizer.step()

                total_loss += loss.item()
                total_positive_loss += pos_loss
                total_negative_loss += neg_loss
                num_pairs += 1

        avg_loss = total_loss / num_pairs
        avg_pos_loss = total_positive_loss / num_pairs
        avg_neg_loss = total_negative_loss / num_pairs

        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, "
              f"Avg Positive Loss: {avg_pos_loss:.4f}, Avg Negative Loss: {avg_neg_loss:.4f}")

        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch + 1}")

    # Save enhanced entity embeddings after training
    model.eval()
    with torch.no_grad():
        enhanced_entity_embeddings = model(entity_embeddings, context_map, attention_weights_map)
        np.save(os.path.join(save_path, "enhanced_entity_embeddings.npy"), enhanced_entity_embeddings.cpu().numpy())
        print(f"Enhanced entity embeddings saved at {save_path}/enhanced_entity_embeddings.npy")
# Load sentence transformer model for embedding generation
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

set_deterministic_seed(42)

# Data Preparation (replace with your specific data loading logic)
# semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge_short.xlsx")
# embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge_short.xlsx")

semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge.xlsx")
embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge.xlsx")
unique_entities = sorted(set(semantic_knowledge_df['subject'].unique().tolist() + 
                             semantic_knowledge_df['object'].unique().tolist()))

entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}


np.save("checkpoints/unique_entities.npy", unique_entities)
print("Unique entities list saved in checkpoints/unique_entities.npy")

# Verify the list
print("Number of unique entities:", len(unique_entities))
print("Sample entities:", unique_entities[:10]) 



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

# Initialize and train the model with checkpoints
model = NodeEmbeddingLayer(num_features=384, hidden_dim=384, output_dim=384)
train_and_save_embeddings_with_checkpoints(model, entity_embeddings, context_map, attention_weights_map)

# Load the saved unique entities list and enhanced embeddings
def load_trained_model_and_embeddings(model_path, embeddings_path, unique_entities_path):
    model = NodeEmbeddingLayer(num_features=384, hidden_dim=384, output_dim=384)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    enhanced_entity_embeddings = np.load(embeddings_path)
    unique_entities = np.load(unique_entities_path, allow_pickle=True)  # Load entities in the correct order
    return model, enhanced_entity_embeddings, unique_entities

# Inference function to process query and return probability distribution
def process_query_with_probability_distribution(query, model, sentence_model, enhanced_entity_embeddings, unique_entities, top_n=5):
    query_embedding = torch.tensor(sentence_model.encode([query]), dtype=torch.float)
    similarity_scores = cosine_similarity(query_embedding.detach().cpu().numpy(), enhanced_entity_embeddings).flatten()
    similarity_tensor = torch.tensor(similarity_scores)
    probability_distribution = F.softmax(similarity_tensor, dim=0).numpy()

    top_indices = np.argsort(probability_distribution)[::-1][:top_n]
    top_entities = [(unique_entities[i], probability_distribution[i]) for i in top_indices]

    print("Top relevant entities for the query:")
    for i, (entity, prob) in enumerate(top_entities):
        print(f"{i + 1}. Entity: {entity}, Probability: {prob:.4f}")

# Load trained model, embeddings, and unique entities for inference
model, enhanced_entity_embeddings, unique_entities = load_trained_model_and_embeddings(
    "checkpoints/best_model.pt", 
    "checkpoints/enhanced_entity_embeddings.npy", 
    "checkpoints/unique_entities.npy"
)

# Example Query
query = "What are some of the important considerations for carbon storage in India?"

# Run inference
process_query_with_probability_distribution(query, model, sentence_model, enhanced_entity_embeddings, unique_entities)

