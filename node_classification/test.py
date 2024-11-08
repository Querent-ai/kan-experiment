from ekan import KAN as eKAN
from fastkan import FastKAN
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.data import HeteroData

def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return eKAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order)

class GCKANLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, grid_size: int = 4, spline_order: int = 3):
        super(GCKANLayer, self).__init__()
        self.kan = eKAN([out_feat, out_feat], grid_size=grid_size, spline_order=spline_order)
        self.edge_transform = nn.Linear(384, out_feat)  # Adjusted to map from 384 to out_feat (e.g., 64)
        self.node_transform = nn.Linear(in_feat, out_feat)  # Transform node features to hidden channels

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
                 num_classes: int, skip: bool = True, grid_size: int = 4, spline_order: int = 3,
                 hidden_layers: int = 2, dropout: float = 0.):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(mp_layers - 1):
            if conv_type == "gcn":
                in_features = num_features if i == 0 else hidden_channels
                self.convs.append(GCKANLayer(in_features, hidden_channels, grid_size, spline_order))
                
        self.skip = skip
        dim_out_message_passing = num_features + (mp_layers - 1) * hidden_channels if skip else hidden_channels
        self.conv_out = GCKANLayer(dim_out_message_passing, num_classes, grid_size, spline_order)
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
        
        x = self.conv_out(x, edge_index, edge_attr)
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

# Initialize the GKAN model
model = GKAN_Nodes(
    conv_type="gcn",
    mp_layers=3,
    num_features=384,
    hidden_channels=64,
    num_classes=10,
    grid_size=4,
    spline_order=3,
    hidden_layers=2,
    dropout=0.2
)

# Perform forward pass
output = model(data)
print("Model output shape:", output.shape)
