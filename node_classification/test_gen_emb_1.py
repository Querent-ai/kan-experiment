from ekan import KAN as eKAN
from fastkan import FastKAN
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch.nn as nn
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Model and utility definitions (same as provided above)

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order)

class GCKANLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, grid_size: int = 4, spline_order: int = 3):
        super(GCKANLayer, self).__init__()
        self.kan = eKAN([out_feat, out_feat], grid_size=grid_size, spline_order=spline_order)
        self.edge_transform = nn.Linear(384, out_feat)  # Transform edge attributes
        self.node_transform = nn.Linear(in_feat, out_feat)  # Transform node features

    def forward(self, x, edge_index, edge_attr):
        edge_attr_transformed = self.edge_transform(edge_attr)
        x_transformed = self.node_transform(x)
        
        row, col = edge_index
        edge_messages = edge_attr_transformed * x_transformed[row]
        
        aggregated_messages = torch.zeros_like(x_transformed)
        aggregated_messages.index_add_(0, col, edge_messages)
        
        x = self.kan(aggregated_messages)
        return x

class GKAN_Nodes(nn.Module):
    def __init__(self, conv_type: str, mp_layers: int, num_features: int, hidden_channels: int,
                 embedding_dim: int, skip: bool = True, grid_size: int = 4, spline_order: int = 3,
                 hidden_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(mp_layers - 1):
            in_features = num_features if i == 0 else hidden_channels
            self.convs.append(GCKANLayer(in_features, hidden_channels, grid_size, spline_order))
                
        self.skip = skip
        # Set the final output dimension to embedding_dim (384)
        dim_out_message_passing = num_features + (mp_layers - 1) * hidden_channels if skip else hidden_channels
        self.conv_out = GCKANLayer(dim_out_message_passing, embedding_dim, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data['entity'].x
        edge_index = data['entity', 'interaction', 'entity'].edge_index
        edge_attr = data['entity', 'interaction', 'entity'].edge_attr
        
        l = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.dropout(x)
            l.append(x)
        
        if self.skip:
            x = torch.cat(l, dim=1)
        
        x = self.conv_out(x, edge_index, edge_attr)  # Output embedding_dim (384) dimensions
        return x

# Load sentence transformer model for embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
semantic_knowledge_df = pd.read_excel("/home/nishant/backup/KANs/semantic_knowledge_short.xlsx")

# Combine subjects and objects into a single set of unique entities
unique_entities = list(set(semantic_knowledge_df['subject'].unique().tolist() +
                           semantic_knowledge_df['object'].unique().tolist()))

# Generate embeddings for each unique entity
entity_embeddings = torch.tensor(sentence_model.encode(unique_entities), dtype=torch.float)
entity_mapping = {name: idx for idx, name in enumerate(unique_entities)}

# Initialize HeteroData structure
data = HeteroData()
data['entity'].x = entity_embeddings

# Create edges and edge attributes
edges = []
edge_attrs = []

for _, row in semantic_knowledge_df.iterrows():
    subject = row['subject']
    object_ = row['object']
    sentence = row['sentence']
    
    subj_idx = entity_mapping[subject]
    obj_idx = entity_mapping[object_]
    
    edges.append([subj_idx, obj_idx])
    
    sentence_embedding = sentence_model.encode([sentence])[0]
    edge_attrs.append(torch.tensor(sentence_embedding, dtype=torch.float))

# Convert edges and edge attributes to PyTorch tensors
data['entity', 'interaction', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data['entity', 'interaction', 'entity'].edge_attr = torch.stack(edge_attrs)

# Verify data structure
print("Unified Graph Data Structure:", data)
print("Edge Index Shape:", data['entity', 'interaction', 'entity'].edge_index.shape)
print("Edge Attributes Shape:", data['entity', 'interaction', 'entity'].edge_attr.shape)

# Initialize the GKAN model with final output embedding_dim of 384
model = GKAN_Nodes(
    conv_type="gcn",
    mp_layers=1,
    num_features=384,
    hidden_channels=64,
    embedding_dim=384,  # Set output embedding dimension to 384
    grid_size=4,
    spline_order=3,
    hidden_layers=2,
    dropout=0.2
)

# Perform forward pass
output = model(data)
node_embeddings = output.detach().cpu().numpy()

# KMeans clustering with Elbow Method and Silhouette Analysis
k_values = range(2, 11)
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(node_embeddings)
    inertia_values.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(node_embeddings, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Save Elbow plot
plt.figure()
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig("elbow_plot.png")

# Save Silhouette plot
plt.figure()
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.savefig("silhouette_plot.png")

# Choosing optimal K based on silhouette score
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters (Silhouette): {optimal_k}")

# Final KMeans clustering with optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(node_embeddings)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=2)
dbscan_labels = dbscan.fit_predict(node_embeddings)
num_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters detected by DBSCAN: {len(set(dbscan_labels))}")

# Visualize clusters using t-SNE for both KMeans and DBSCAN
# Reduce dimensions for visualization
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(node_embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

# Save KMeans t-SNE plot
plt.figure()
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='KMeans Cluster Label')
plt.title('t-SNE Visualization of KMeans Clustering')
plt.savefig("kmeans_tsne_plot.png")

# Save DBSCAN t-SNE plot
plt.figure()
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='DBSCAN Cluster Label')
plt.title('t-SNE Visualization of DBSCAN Clustering')
plt.savefig("dbscan_tsne_plot.png")


# Assuming `unique_entities` is the list of entity names and `kmeans_labels` contains cluster labels from KMeans
kmeans_cluster_df = pd.DataFrame({
    'Entity': unique_entities,
    'KMeans_Cluster': kmeans_labels
})

# Display entities by cluster for KMeans
print("Entities in each KMeans Cluster:")
for cluster in kmeans_cluster_df['KMeans_Cluster'].unique():
    entities_in_cluster = kmeans_cluster_df[kmeans_cluster_df['KMeans_Cluster'] == cluster]['Entity'].tolist()
    print(f"Cluster {cluster}:")
    print(entities_in_cluster)
    print("------------------------------------------------")

# Assuming `dbscan_labels` contains cluster labels from DBSCAN
dbscan_cluster_df = pd.DataFrame({
    'Entity': unique_entities,
    'DBSCAN_Cluster': dbscan_labels
})

# Display entities by cluster for DBSCAN
print("Entities in each DBSCAN Cluster:")
for cluster in dbscan_cluster_df['DBSCAN_Cluster'].unique():
    entities_in_cluster = dbscan_cluster_df[dbscan_cluster_df['DBSCAN_Cluster'] == cluster]['Entity'].tolist()
    print(f"Cluster {cluster}:")
    print(entities_in_cluster)
    print("------------------------------------------------")
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

# Calculate silhouette scores for each entity
silhouette_vals = silhouette_samples(node_embeddings, kmeans_labels)
silhouette_avg = silhouette_score(node_embeddings, kmeans_labels)

# Visualize silhouette scores by cluster
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_values = silhouette_vals[kmeans_labels == i]
    ith_cluster_silhouette_values.sort()
    y_upper = y_lower + len(ith_cluster_silhouette_values)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)
    plt.text(-0.05, y_lower + 0.5 * len(ith_cluster_silhouette_values), str(i))
    y_lower = y_upper + 10  # Space between clusters

plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster")
plt.title(f"Silhouette Plot for KMeans Clustering (Average: {silhouette_avg:.2f})")
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.savefig("kmeans_silhouette_plot.png")



from sklearn.metrics import pairwise_distances_argmin_min

# Calculate the distances of each point to its cluster center
distances = []
for i in range(optimal_k):
    cluster_points = node_embeddings[kmeans_labels == i]
    cluster_center = kmeans.cluster_centers_[i]
    dists = np.linalg.norm(cluster_points - cluster_center, axis=1)
    distances.append(dists.mean())

print(f"Average Compactness (Mean distance to center) per cluster: {distances}")
