import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

########################################################
########################################################
############ GRAPH CLASSIFICATION MODELS ###############
########################################################
########################################################
class GCN_graph_classif_model(torch.nn.Module):
    def __init__(self, in_dim, hid_1_dim, hid_2_dim, hid_3_dim, out_dim):
        super(GCN_graph_classif_model, self).__init__()
        
        # GCN Layers 
        self.conv1 = GCNConv(in_dim, hid_1_dim)
        self.conv2 = GCNConv(hid_1_dim, hid_2_dim)
        self.conv3 = GCNConv(hid_2_dim, hid_3_dim)
        self.lin1 = Linear(hid_3_dim, out_dim)

    def forward(self, x, edge_index, batch):
        
        ## Node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # Merge embedding into one single vector
        x = global_mean_pool(x, batch)
        
        # Apply a final classifier
        x = self.lin1(x)
        
        return x

#######################################################
#######################################################
##### NODE CLASSIFICATION MODELS (DO NOT TOUCH) #######
#######################################################
#######################################################
class MLP_1_hidden_model(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP_1_hidden_model, self).__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(in_dim, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    
class GCN_1_hidden_model(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GCN_1_hidden_model, self).__init__()
        torch.manual_seed(123)
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    
class GCN_2_hidden_model(torch.nn.Module):
    def __init__(self, in_dim, hid_1_dim, hid_2_dim, out_dim):
        super(GCN_2_hidden_model, self).__init__()
        torch.manual_seed(12345678)
        self.conv1 = GCNConv(in_dim, hid_1_dim)
        self.conv2 = GCNConv(hid_1_dim, hid_2_dim)
        self.conv3 = GCNConv(hid_2_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x
   