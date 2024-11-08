from ekan import KAN as eKAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.cluster import DBSCAN



# Simplified Node Embedding Layer
class NodeEmbeddingLayer(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, output_dim: int, grid_size: int = 4, spline_order: int = 4):
        super(NodeEmbeddingLayer, self).__init__()
        # Transformations for node and context
        self.node_transform = nn.Linear(num_features, hidden_dim)
        self.context_transform = nn.Linear(num_features, hidden_dim)
        self.update_transform = nn.Linear(hidden_dim, output_dim)
        # Kernel Activation Network (KAN) layer
        self.node_kan = eKAN([output_dim, output_dim], grid_size=grid_size, spline_order=spline_order)

    def forward(self, x, context_map, attention_weights_map):
        updated_embeddings = []
        for node_idx in range(x.size(0)):
            # Only aggregate contexts where the current node appears
            relevant_contexts = context_map.get(node_idx, [])
            relevant_attention_weights = attention_weights_map.get(node_idx, [])

            if relevant_contexts:
                # Stack relevant contexts and attention weights
                context_stack = torch.stack(relevant_contexts)
                attention_weights = torch.tensor(relevant_attention_weights, dtype=torch.float).view(-1, 1)
                
                # Transform node and aggregate context
                x_transformed = self.node_transform(x[node_idx].unsqueeze(0))
                context_transformed = self.context_transform(context_stack)
                context_weighted = context_transformed * attention_weights
                context_aggregated = context_weighted.mean(dim=0)
                
                # Combine node and context information
                updated_embedding = self.update_transform(x_transformed + context_aggregated)
                node_embedding = self.node_kan(updated_embedding).squeeze(0)
            else:
                # If no contexts found, just use the original transformed node
                node_embedding = x[node_idx]

            updated_embeddings.append(node_embedding)
        
        # Stack all updated embeddings for output
        node_embeddings = torch.stack(updated_embeddings)
        return node_embeddings

# Load sentence transformer model for embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data and create embeddings (replace this with your data loading logic)
semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge_short.xlsx")
embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge_short.xlsx")
unique_entities = list(set(semantic_knowledge_df['subject'].unique().tolist() + semantic_knowledge_df['object'].unique().tolist()))
entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}

# Initialize HeteroData structure
data = HeteroData()
data['entity'].x = entity_embeddings

# Create mappings for context and attention weights
context_map = {i: [] for i in range(len(unique_entities))}
attention_weights_map = {i: [] for i in range(len(unique_entities))}

# Fill context_map and attention_weights_map with relevant data
event_to_attention = embedded_knowledge_df.set_index('event_id')['score'].to_dict()

for _, row in semantic_knowledge_df.iterrows():
    subject = row['subject']
    object_ = row['object']
    sentence = row['sentence']
    event_id = row['event_id']

    subj_idx = entity_mapping[subject]
    obj_idx = entity_mapping[object_]

    # Get sentence embedding and attention weight
    sentence_embedding = torch.tensor(sentence_model.encode([sentence])[0], dtype=torch.float)
    attention_weight = event_to_attention.get(event_id, 1.0)  # Default to 1.0 if no attention weight found

    # Add to context_map and attention_weights_map for subject and object
    context_map[subj_idx].append(sentence_embedding)
    context_map[obj_idx].append(sentence_embedding)
    attention_weights_map[subj_idx].append(attention_weight)
    attention_weights_map[obj_idx].append(attention_weight)

# Initialize the Node Embedding Layer
node_layer = NodeEmbeddingLayer(num_features=384, hidden_dim=384, output_dim=384)

# Forward pass through NodeEmbeddingLayer
node_embeddings = node_layer(data['entity'].x, context_map, attention_weights_map)

# Convert node embeddings to NumPy for clustering
node_embeddings_np = node_embeddings.detach().cpu().numpy()

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=2)
dbscan_labels = dbscan.fit_predict(node_embeddings_np)

# Count the number of clusters (excluding noise)
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"[INFO] Number of clusters detected by DBSCAN: {num_clusters}")

# Display entities by cluster
dbscan_cluster_df = pd.DataFrame({
    'Entity': unique_entities,
    'DBSCAN_Cluster': dbscan_labels
})

print("Entities in each DBSCAN Cluster:")
for cluster in dbscan_cluster_df['DBSCAN_Cluster'].unique():
    if cluster == -1:
        print(f"Noise (cluster -1):")
    else:
        print(f"Cluster {cluster}:")
    entities_in_cluster = dbscan_cluster_df[dbscan_cluster_df['DBSCAN_Cluster'] == cluster]['Entity'].tolist()
    print(entities_in_cluster)
    print("------------------------------------------------")

# Save DBSCAN results for further analysis or visualization
dbscan_cluster_df.to_csv("dbscan_cluster_results.csv", index=False)
