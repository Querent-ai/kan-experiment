from ekan import KAN as eKAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch_geometric.data import HeteroData
import numpy as np

# Utility function to create KAN layers
def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order)

# Node Embedding Layer
class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int, grid_size: int = 4, spline_order: int = 3):
        super(NodeEmbeddingLayer, self).__init__()
        self.node_transform = nn.Linear(num_features, hidden_dim)
        self.context_transform = nn.Linear(num_features, hidden_dim)
        self.update_transform = nn.Linear(hidden_dim, output_dim)
        self.node_kan = eKAN([output_dim, output_dim], grid_size=grid_size, spline_order=spline_order)

    def forward(self, x, contexts, attention_weights):
        if x.size(0) == 0:
            print("[ERROR] NodeEmbeddingLayer - Received empty node embeddings.")
            return x  # Return empty tensor if x is empty

        print(f"[DEBUG] NodeEmbeddingLayer - Input x size: {x.size()}")
        x_transformed = self.node_transform(x)
        print(f"[DEBUG] NodeEmbeddingLayer - Transformed x size: {x_transformed.size()}")

        context_updates = torch.stack([ctx * weight for ctx, weight in zip(contexts, attention_weights)], dim=0)
        context_aggregated = context_updates.mean(dim=0)  # Attention-weighted context aggregation
        print(f"[DEBUG] NodeEmbeddingLayer - Aggregated context size: {context_aggregated.size()}")

        context_aggregated_transformed = self.context_transform(context_aggregated)
        
        # Combine node and context information, then update
        updated_embedding = self.update_transform(x_transformed + context_aggregated_transformed)
        print(f"[DEBUG] NodeEmbeddingLayer - Updated embedding size before KAN: {updated_embedding.size()}")

        # Pass through KAN for non-linear transformation
        node_embeddings = self.node_kan(updated_embedding)
        print(f"[DEBUG] NodeEmbeddingLayer - Output node embeddings size: {node_embeddings.size()}")
        return node_embeddings

# Pair Embedding Layer
class PairEmbeddingLayer(nn.Module):
    def __init__(self, node_dim: int, context_dim: int, output_dim: int, grid_size: int = 4, spline_order: int = 3):
        super(PairEmbeddingLayer, self).__init__()
        self.context_transform = nn.Linear(context_dim, node_dim)
        self.pair_kan = eKAN([node_dim * 2, output_dim], grid_size=grid_size, spline_order=spline_order)

    def forward(self, node_embeddings, context_embeddings, pair_indices):
        if node_embeddings.size(0) == 0 or context_embeddings.size(0) == 0 or pair_indices.size(0) == 0:
            print("[ERROR] PairEmbeddingLayer - Received empty tensor(s).")
            return torch.empty(0, node_embeddings.size(1), requires_grad=True)  # Return empty tensor with requires_grad

        print(f"[DEBUG] PairEmbeddingLayer - Node embeddings size: {node_embeddings.size()}")
        print(f"[DEBUG] PairEmbeddingLayer - Context embeddings size: {context_embeddings.size()}")
        
        subj_embeddings = node_embeddings[pair_indices[:, 0]]
        obj_embeddings = node_embeddings[pair_indices[:, 1]]
        context_transformed = self.context_transform(context_embeddings)

        pair_input = torch.cat([subj_embeddings, obj_embeddings, context_transformed], dim=1)
        print(f"[DEBUG] PairEmbeddingLayer - Pair input size: {pair_input.size()}")

        pair_embeddings = self.pair_kan(pair_input)
        print(f"[DEBUG] PairEmbeddingLayer - Output pair embeddings size: {pair_embeddings.size()}")
        return pair_embeddings

# Memory Bank for Continuous Learning
class MemoryBank:
    def __init__(self, num_nodes, node_dim, num_pairs, pair_dim):
        self.node_memory = torch.zeros(num_nodes, node_dim, requires_grad=True)
        self.pair_memory = torch.zeros(num_pairs, pair_dim, requires_grad=True)

    def update_node_embeddings(self, indices, new_embeddings):
        if indices.size(0) == 0:
            print("[WARNING] MemoryBank - No node embeddings to update.")
            return
        self.node_memory[indices] = new_embeddings

    def update_pair_embeddings(self, indices, new_embeddings):
        if indices.size(0) == 0:
            print("[WARNING] MemoryBank - No pair embeddings to update.")
            return
        self.pair_memory[indices] = new_embeddings

    def get_node_embeddings(self, indices):
        print(f"[DEBUG] MemoryBank - Fetching node embeddings for indices: {indices}")
        return self.node_memory[indices]

    def get_pair_embeddings(self, indices):
        print(f"[DEBUG] MemoryBank - Fetching pair embeddings for indices: {indices}")
        return self.pair_memory[indices]

# Cosine Similarity Regularization
def cosine_similarity_regularization(node_embeddings, pair_embeddings, pair_indices):
    if node_embeddings.size(0) == 0 or pair_embeddings.size(0) == 0:
        print("[ERROR] Cosine similarity regularization received empty embeddings.")
        return torch.tensor(0.0, requires_grad=True)

    subj_embeddings = node_embeddings[pair_indices[:, 0]]
    obj_embeddings = node_embeddings[pair_indices[:, 1]]
    pair_sims = F.cosine_similarity(pair_embeddings, subj_embeddings + obj_embeddings)
    return pair_sims.mean()

# Two-Layer KAN Model
class TwoLayerKAN(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, pair_dim: int, context_dim: int, num_nodes: int, num_pairs: int):
        super(TwoLayerKAN, self).__init__()
        self.node_layer = NodeEmbeddingLayer(node_dim, hidden_dim, node_dim)
        self.pair_layer = PairEmbeddingLayer(node_dim, context_dim, pair_dim)
        self.memory_bank = MemoryBank(num_nodes, node_dim, num_pairs, pair_dim)

    def forward(self, data, context_embeddings, pair_indices, attention_weights):
        node_embeddings = self.memory_bank.get_node_embeddings(data['entity_indices'])
        if node_embeddings.size(0) == 0:
            print("[ERROR] TwoLayerKAN - No node embeddings found in memory bank.")
            return torch.empty(0, data['entity'].x.size(1), requires_grad=True), torch.empty(0, data['entity'].x.size(1), requires_grad=True)

        updated_node_embeddings = self.node_layer(node_embeddings, context_embeddings, attention_weights)
        self.memory_bank.update_node_embeddings(data['entity_indices'], updated_node_embeddings)
        
        pair_embeddings = self.pair_layer(updated_node_embeddings, context_embeddings, pair_indices)
        self.memory_bank.update_pair_embeddings(data['pair_indices'], pair_embeddings)

        return updated_node_embeddings, pair_embeddings

# Load sentence transformer model for embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data and create embeddings (add debug logs for these)
semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge_short.xlsx")
embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge_short.xlsx")

unique_entities = list(set(semantic_knowledge_df['subject'].unique().tolist() + semantic_knowledge_df['object'].unique().tolist()))
entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}

print(f"[DEBUG] Unique entities count: {len(unique_entities)}")
print(f"[DEBUG] Entity mapping: {entity_mapping}")

# Initialize HeteroData structure
data = HeteroData()
data['entity'].x = entity_embeddings

event_to_attention = embedded_knowledge_df.set_index('event_id')['score'].to_dict()

# Create edges and edge attributes
edges = []
edge_attrs = []
attention_weights = []

for _, row in semantic_knowledge_df.iterrows():
    subject = row['subject']
    object_ = row['object']
    sentence = row['sentence']
    event_id = row['event_id']
    subj_idx = entity_mapping[subject]
    obj_idx = entity_mapping[object_]
    
    edges.append([subj_idx, obj_idx])
    sentence_embedding = sentence_model.encode([sentence])[0]
    edge_attrs.append(torch.tensor(sentence_embedding, dtype=torch.float))
    attention_weight = event_to_attention.get(event_id, 1.0)  # Default to 1.0 if no attention weight found
    attention_weights.append(attention_weight)

# Convert edges and edge attributes to PyTorch tensors
data['entity', 'interaction', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data['entity', 'interaction', 'entity'].edge_attr = torch.stack(edge_attrs)
attention_weights = torch.tensor(attention_weights, dtype=torch.float)

# Initialize the TwoLayerKAN model
model = TwoLayerKAN(
    node_dim=384,
    hidden_dim=64,
    pair_dim=384,
    context_dim=384,
    num_nodes=len(unique_entities),
    num_pairs=len(edges)
)

# Define Training Function
def train_model(model, data, context_embeddings, pair_indices, attention_weights, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        node_embeddings, pair_embeddings = model(data, context_embeddings, pair_indices, attention_weights)

        # Cosine similarity regularization as a loss
        reg_loss = cosine_similarity_regularization(node_embeddings, pair_embeddings, pair_indices)
        loss = -reg_loss  # Negative to maximize similarity

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Define input for training (context_embeddings, attention_weights, etc.)
context_embeddings = data['entity', 'interaction', 'entity'].edge_attr
pair_indices = data['entity', 'interaction', 'entity'].edge_index.t()

# Train the model
train_model(model, data, context_embeddings, pair_indices, attention_weights, num_epochs=10)
