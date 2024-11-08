from ekan import KAN as eKAN
import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from collections import defaultdict
from sklearn.cluster import DBSCAN

# Updated Pair-Context Embedding Layer using KAN
class PairContextEmbeddingLayer(nn.Module):
    def __init__(self, node_dim: int, context_dim: int, output_dim: int, grid_size: int = 4, spline_order: int = 4):
        super(PairContextEmbeddingLayer, self).__init__()
        # Initialize a KAN layer for each pair-context embedding
        self.kan = eKAN([node_dim * 2 + context_dim, output_dim], grid_size=grid_size, spline_order=spline_order)

    def forward(self, subject_embedding, object_embedding, context_embedding, attention_weight):
        # Adjust context embedding by attention weight
        weighted_context = context_embedding * attention_weight

        # Concatenate subject, object, and weighted context embeddings
        combined_input = torch.cat([subject_embedding, object_embedding, weighted_context], dim=-1)
        
        # Pass through KAN for a non-linear transformation
        pair_context_embedding = self.kan(combined_input)
        
        return pair_context_embedding

# Load data and create embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge_short.xlsx")
embedded_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/embedded_knowledge_short.xlsx")
unique_entities = list(set(semantic_knowledge_df['subject'].unique().tolist() + semantic_knowledge_df['object'].unique().tolist()))
entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}

# Initialize HeteroData structure
data = HeteroData()
data['entity'].x = entity_embeddings

# Mapping for attention weights
event_to_attention = embedded_knowledge_df.set_index('event_id')['score'].to_dict()

# Initialize the PairContextEmbeddingLayer
node_dim = 384
context_dim = 384
output_dim = 384
pair_context_layer = PairContextEmbeddingLayer(node_dim=node_dim, context_dim=context_dim, output_dim=output_dim)

# Dictionary to collect unique pair-context embeddings
unique_pair_context_embeddings = []
unique_pairs = []
pair_sentences = []

for _, row in semantic_knowledge_df.iterrows():
    subject = row['subject']
    object_ = row['object']
    sentence = row['sentence']
    event_id = row['event_id']
    
    # Get entity indices
    subj_idx = entity_mapping[subject]
    obj_idx = entity_mapping[object_]
    
    # Encode sentence to get the context embedding
    context_embedding = torch.tensor(sentence_model.encode([sentence])[0], dtype=torch.float)
    
    # Fetch attention weight for this context
    attention_weight = event_to_attention.get(event_id, 1.0)  # Default to 1.0 if no attention weight found
    
    # Generate pair-context embedding using KAN
    subject_embedding = entity_embeddings[subj_idx].unsqueeze(0)
    object_embedding = entity_embeddings[obj_idx].unsqueeze(0)
    context_embedding = context_embedding.unsqueeze(0)
    pair_context_embedding = pair_context_layer(subject_embedding, object_embedding, context_embedding, attention_weight)
    
    # Store results for clustering
    unique_pair_context_embeddings.append(pair_context_embedding.squeeze(0))  # Remove the batch dimension
    unique_pairs.append((unique_entities[subj_idx], unique_entities[obj_idx]))
    pair_sentences.append(sentence)

# Stack embeddings for clustering
unique_pair_context_embeddings = torch.stack(unique_pair_context_embeddings)
print(f"[INFO] Generated unique pair-context embeddings shape: {unique_pair_context_embeddings.shape}")

# Perform clustering on unique pair-context embeddings
pair_embeddings_np = unique_pair_context_embeddings.detach().cpu().numpy()
dbscan = DBSCAN(eps=0.2, min_samples=2)
dbscan_labels = dbscan.fit_predict(pair_embeddings_np)

# Summarize clusters with unique pairs
pair_cluster_df = pd.DataFrame({
    'Subject': [pair[0] for pair in unique_pairs],
    'Object': [pair[1] for pair in unique_pairs],
    'DBSCAN_Cluster': dbscan_labels,
    'Sentence': pair_sentences  # Add sentence for verification
})

# Print total number of clusters including noise
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"[INFO] Total number of clusters (excluding noise): {num_clusters}")
print(f"[INFO] Total number of clusters (including noise): {len(set(dbscan_labels))}")

# Display pairs by cluster
print("Unique Pair-Context Clusters:")
for cluster in pair_cluster_df['DBSCAN_Cluster'].unique():
    if cluster == -1:
        print("Noise (cluster -1):")
    else:
        print(f"Cluster {cluster}:")
    pairs_in_cluster = pair_cluster_df[pair_cluster_df['DBSCAN_Cluster'] == cluster][['Subject', 'Object', 'Sentence']].values
    for pair in pairs_in_cluster:
        # print(f"Pair: ({pair[0]}, {pair[1]}), Sentence: {pair[2]}")
        print(f"Pair: ({pair[0]}, {pair[1]})")
    print("------------------------------------------------")

# Save clustering results for further analysis
pair_cluster_df.to_csv("unique_pair_context_dbscan_results_with_sentences.csv", index=False)
