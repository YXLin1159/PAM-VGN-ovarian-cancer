import torch
import numpy as np
from tqdm import tqdm
from losses import FocalLoss, CLASS_WTS

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_MLP(model, loader, optimizer, device):
    class_weights = torch.tensor(CLASS_WTS, dtype=torch.float, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    batch_size = loader.batch_size
    model.train()
    loss_sum = 0.0
    total_samples = 0
    for data in tqdm(loader, desc = 'TRAINING MLP', leave = False):
        x = data.x.to(device)
        y = data.y.to(device).long()
        x = x.view(batch_size, -1)  # Flatten the input to (batch_size, N_NODE_FEATURES_TOTAL)
        optimizer.zero_grad()
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch_size
        total_samples += batch_size
        torch.cuda.empty_cache()
    return loss_sum / total_samples

def train_GNN(model, loader, optimizer, device):
    class_weights = torch.tensor(CLASS_WTS, dtype=torch.float, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    batch_size = loader.batch_size
    model.train()
    loss_sum = 0.0
    total_samples = 0
    for data in tqdm(loader, desc = 'TRAINING GNN', leave = False):
        x = torch.cat([torch.tensor(f, dtype=torch.float32) for f in data.x], dim=0).to(device)
        edge_attr = torch.cat([torch.tensor(3 - e/1000, dtype=torch.float32) for e in data.edge_attr], dim=0).to(device)
        edge_index = torch.cat([torch.tensor(e, dtype=torch.long) for e in data.edge_index], dim=1).to(device)
        batch_id = data.batch.to(device) # batch_id indicates which graph the nodes belong to
        y = torch.tensor(data.y, dtype=torch.long).to(device)
        optimizer.zero_grad()
        pred_y = model(x, edge_index, edge_attr, batch_id)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch_size
        total_samples += batch_size
        torch.cuda.empty_cache()
    return loss_sum / total_samples