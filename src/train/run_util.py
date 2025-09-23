import torch
import os
import numpy as np
from models import MLP4, VGN
from .train_util import train_MLP, train_GNN, get_device
from .test_util import test_MLP, test_GNN, test_GNN_LOO
from src.utils.save_utils import make_model_save_path

N_CLASS = 5

def run_MLP(train_dataloader , valid_dataloader , test_dataloader , hidden_channels, N_epoch:int=100 , model_save_path:str=None , device=None):
    device = get_device() if device is None else device
    model = MLP4(hidden_channels = hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-6)
    optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,90], gamma=0.1)

    # initialize logs
    valid_acc_cm_log = np.zeros(N_epoch)
    test_acc_cm_log = np.zeros(N_epoch)
    best_acc = 0

    model_save_path = make_model_save_path(filename="best_MLP_wts.pth", subfolder="model_state_dict_log") if model_save_path is None else model_save_path
    for epoch in range(1, (N_epoch + 1)):
        loss = train_MLP(model , train_dataloader , optimizer, device)
        _ , valid_cm = test_MLP(model , valid_dataloader, device)        
        valid_acc_cm = np.trace(valid_cm) / N_CLASS
        
        _  , test_cm  = test_MLP(model , test_dataloader, device)
        test_acc_cm = np.trace(test_cm) / N_CLASS
                
        valid_acc_cm_log[epoch-1] = valid_acc_cm
        test_acc_cm_log[epoch-1]  = test_acc_cm
        optimizer_scheduler.step()
        
        if valid_acc_cm > best_acc and epoch > 20:
            torch.save(model.state_dict(), model_save_path)
            best_acc = valid_acc_cm
            print(f"Epoch {epoch}: Best model saved with validation acc_cm = {best_acc:.4f}")
            
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Valid Acc_cm: {valid_acc_cm:.4f} | Test Acc_cm: {test_acc_cm:.4f}")
    
    model.load_state_dict(torch.load(model_save_path))
    _ , final_test_cm = test_MLP(model, test_dataloader, device)
    result_log = {"valid_acc_cm_log": valid_acc_cm_log,"test_acc_cm_log": test_acc_cm_log,"final_test_cm": final_test_cm}
    return result_log

def run_GNN(train_dataloader , valid_dataloader , test_dataloader , hidden_channels, N_epoch:int=200 , model_save_path:str=None , device=None):
    device = get_device() if device is None else device
    model = VGN(hidden_channels = hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-6)
    optimizer_warmup = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-6)
    optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,30,60,100,150,195], gamma=0.5)
    
    # initialize logs
    valid_acc_cm_log = np.zeros(N_epoch)
    test_acc_cm_log = np.zeros(N_epoch)
    best_acc = 0
    
    model_save_path = make_model_save_path(filename="best_GNN_wts.pth", subfolder="model_state_dict_log") if model_save_path is None else model_save_path
    # WARM UP ITERATION
    _ = train_GNN(model , train_dataloader, optimizer_warmup, device)
    for epoch in range(1, (N_epoch + 1)):
        loss = train_GNN(model , train_dataloader , optimizer, device)
        _ , valid_cm , _ = test_GNN(model , valid_dataloader, device)
        _  , test_cm  , _  = test_GNN(model , valid_dataloader, device)
        
        valid_acc_cm = np.trace(valid_cm)/N_CLASS
        test_acc_cm = np.trace(test_cm)/N_CLASS
        
        valid_acc_cm_log[epoch-1] = valid_acc_cm
        test_acc_cm_log[epoch-1]  = test_acc_cm
        optimizer_scheduler.step()
        
        if valid_acc_cm > best_acc and epoch > 45:
            torch.save(model.state_dict(), model_save_path)
            best_acc = valid_acc_cm
            print(f"Epoch {epoch}: Best model saved (validation acc_cm={best_acc:.4f})")
            
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Valid Acc_cm: {valid_acc_cm:.4f} | Test Acc_cm: {test_acc_cm:.4f}")
    
    model.load_state_dict(torch.load(model_save_path))
    _, final_test_cm, final_test_roc = test_GNN(model , test_dataloader , device)
    result_log = {
        "valid_acc_cm_log": valid_acc_cm_log,
        "test_acc_cm_log": test_acc_cm_log,
        "final_test_cm": final_test_cm,
        "final_test_roc": final_test_roc
    }
    return result_log

def run_GNN_LOO(train_dataloader , valid_dataloader , test_dataloader , hidden_channels, N_epoch:int=200 , model_save_path:str=None , device=None):
    device = get_device() if device is None else device
    model = VGN(hidden_channels = hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-6)
    optimizer_warmup = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-6)
    optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,30,60,100,150,195], gamma=0.5)
    
    # initialize logs
    valid_acc_cm_log = np.zeros(N_epoch)
    test_acc_cm_log = np.zeros(N_epoch)
    best_acc = 0
    
    model_save_path = make_model_save_path(filename="best_GNN_wts.pth", subfolder="model_state_dict_log") if model_save_path is None else model_save_path
    # WARM UP ITERATION
    _ = train_GNN(model , train_dataloader, optimizer_warmup, device)
    for epoch in range(1, (N_epoch + 1)):
        loss = train_GNN(model , train_dataloader , optimizer, device)
        _ , valid_cm , _ = test_GNN(model , valid_dataloader, device)
        _  , test_cm  = test_GNN_LOO(model , valid_dataloader, device)
        
        valid_acc_cm = np.trace(valid_cm)/N_CLASS
        test_acc_cm = np.trace(test_cm)/N_CLASS
        
        valid_acc_cm_log[epoch-1] = valid_acc_cm
        test_acc_cm_log[epoch-1]  = test_acc_cm
        optimizer_scheduler.step()
        
        if valid_acc_cm > best_acc and epoch > 45:
            torch.save(model.state_dict(), model_save_path)
            best_acc = valid_acc_cm
            print(f"Epoch {epoch}: Best model saved (validation acc_cm={best_acc:.4f})")
            
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Valid Acc_cm: {valid_acc_cm:.4f} | Test Acc_cm: {test_acc_cm:.4f}")
    
    model.load_state_dict(torch.load(model_save_path))
    _, final_test_cm, final_test_roc = test_GNN(model , test_dataloader , device)
    result_log = {
        "valid_acc_cm_log": valid_acc_cm_log,
        "test_acc_cm_log": test_acc_cm_log,
        "final_test_cm": final_test_cm,
    }
    return result_log , model