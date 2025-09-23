import numpy as np
import torch
import torch_geometric

# Define some global constants for the project
N_NODE_FEATURES = 7
N_CLASS = 5
N_consecutive_bscans = 41
N_BV0 = 5
N_NODE_FEATURES_TOTAL = N_NODE_FEATURES*N_consecutive_bscans*N_BV0

class MLP4(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(10086)
        self.linear1 = torch.nn.Linear(N_NODE_FEATURES_TOTAL, hidden_channels[0])
        self.linear2 = torch.nn.Linear(hidden_channels[0], hidden_channels[1])
        self.linear3 = torch.nn.Linear(hidden_channels[1], hidden_channels[2])
        self.linear4 = torch.nn.Linear(hidden_channels[2], N_CLASS)

    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.linear2(x)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.linear3(x)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.linear4(x)
        return x

class GCN4(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN4, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = torch_geometric.nn.GCNConv(N_NODE_FEATURES, hidden_channels[0])
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = torch_geometric.nn.GCNConv(hidden_channels[1], hidden_channels[2])
        self.linear = torch.nn.Linear(hidden_channels[2], N_CLASS)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training) # CLASSIFIER
        x = self.linear(x)        
        return x

class GAT4(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT4 , self).__init__()
        torch.manual_seed(000000)
        self.conv1 = torch_geometric.nn.GATv2Conv(N_NODE_FEATURES      , hidden_channels[0], heads = 4, edge_dim=1, add_self_loops=True)
        self.conv2 = torch_geometric.nn.GATv2Conv(hidden_channels[0]*4 , hidden_channels[1], heads = 4, edge_dim=1, add_self_loops=True)
        self.conv3 = torch_geometric.nn.GATv2Conv(hidden_channels[1]*4 , hidden_channels[2], heads = 4, edge_dim=1, add_self_loops=True)
        self.linear = torch.nn.Linear(hidden_channels[2]*4 , N_CLASS)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr, batch)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr, batch)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr, batch)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        return x

class PNA4(torch.nn.Module):    
    def __init__(self, deg_histogram , hidden_channels):
        super(PNA4 , self).__init__()
        torch.manual_seed(000000)
        aggregators = ['mean' , 'min' , 'max' , 'std']
        scalers = ['identity' , 'amplification' , 'attenuation']
        self.conv1 = torch_geometric.nn.PNAConv(in_channels=N_NODE_FEATURES , out_channels = hidden_channels[0] , aggregators = aggregators , scalers = scalers , deg = deg_histogram , edge_dim = 1)
        self.conv2 = torch_geometric.nn.PNAConv(in_channels=hidden_channels[0] , out_channels = hidden_channels[1] , aggregators = aggregators , scalers = scalers , deg = deg_histogram , edge_dim = 1)
        self.conv3 = torch_geometric.nn.PNAConv(in_channels=hidden_channels[1] , out_channels = hidden_channels[2] , aggregators = aggregators , scalers = scalers , deg = deg_histogram , edge_dim = 1)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_channels[2] , int(hidden_channels[2]/2)) , torch.nn.ReLU() , torch.nn.Linear(int(hidden_channels[2]/2) , N_CLASS))
      
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        return x

class GIN4(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GIN4 , self).__init__()
        torch.manual_seed(000000)
        self.conv1 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(N_NODE_FEATURES    , hidden_channels[0]) , torch.nn.ReLU()) , edge_dim=1)
        self.conv2 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels[0] , hidden_channels[1]) , torch.nn.ReLU()) , edge_dim=1)
        self.conv3 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels[1] , hidden_channels[2]) , torch.nn.ReLU()) , edge_dim=1)
        self.linear = torch.nn.Linear(hidden_channels[2], N_CLASS)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training) # CLASSIFIER
        x = self.linear(x)
        return x

class VGN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(VGN , self).__init__()
        torch.manual_seed(000000)
        self.conv1 = torch_geometric.nn.GATv2Conv(N_NODE_FEATURES      , hidden_channels[0], heads = 4, edge_dim=1, add_self_loops=True)
        self.conv2 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels[0]*4 , hidden_channels[1]) , torch.nn.GELU()) , edge_dim=1)
        self.conv3 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels[1] , hidden_channels[2]) , torch.nn.GELU()) , edge_dim=1)
        self.conv4 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels[2] , hidden_channels[3]) , torch.nn.GELU()) , edge_dim=1)
        self.linear = torch.nn.Linear(hidden_channels[3], N_CLASS)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.linear(x)
        return x

