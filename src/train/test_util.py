import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import MulticlassConfusionMatrix
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
N_CLASS = 5

def test_MLP(model , loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    test_cm = torch.zeros((N_CLASS,N_CLASS), dtype=torch.float32, device=device)
    cm_metric = MulticlassConfusionMatrix(N_CLASS).to(device)
    batch_size = loader.batch_size
    with torch.no_grad():
        for data in tqdm(loader, desc = 'TESTING MLP', leave = False):
            x = data.x.to(device).view(batch_size, -1)  # Flatten the input to (batch_size, N_NODE_FEATURES_TOTAL)
            y = data.y.to(device).long()
            out = model(x)
            pred = out.argmax(dim=1)
            correct_preds += (pred == y).sum().item()
            total_preds += batch_size
            test_cm += cm_metric(out , y)
    test_acc = correct_preds / total_preds
    # Normalize the confusion matrix to get per-true-class accuracy
    test_cm_normalized = test_cm.cpu().numpy()
    row_sums = test_cm_normalized.sum(axis=1, keepdims=True)
    test_cm_normalized = np.divide(test_cm_normalized, row_sums, out=np.zeros_like(test_cm_normalized), where=row_sums!=0)
    return test_acc , test_cm_normalized

def test_GNN(model , loader, device):
    '''
    Evaluate the GNN model on the test dataset and compute accuracy, confusion matrix, and ROC curves.
    '''
    model.eval()
    correct_preds = 0
    total_preds = 0
    test_cm = torch.zeros((N_CLASS,N_CLASS), dtype=torch.float32, device=device)
    cm_metric = MulticlassConfusionMatrix(N_CLASS).to(device)
    output_list = []
    truelabel_list = []
    batch_size = loader.batch_size
    L_interp_roc = 101
    thresholds = np.linspace(0,1,num=L_interp_roc,endpoint=True)
    with torch.no_grad():
        for data in tqdm(loader, desc = 'TESTING GNN', leave = False):
            x = torch.cat([torch.tensor(f, dtype=torch.float32) for f in data.x], dim=0).to(device)
            edge_attr = torch.cat([torch.tensor(3 - e/1000, dtype=torch.float32) for e in data.edge_attr], dim=0).to(device)
            edge_index = torch.cat([torch.tensor(e, dtype=torch.long) for e in data.edge_index], dim=1).to(device)
            batch_id = data.batch.to(device) # batch_id indicates which graph the nodes belong to
            y = torch.tensor(data.y, dtype=torch.long).to(device)
            out = model(x, edge_index, edge_attr, batch_id)
            pred = out.argmax(dim=1)
            correct_preds += (pred == y).sum().item()
            total_preds += batch_size
            test_cm += cm_metric(out , y)
            output_list.append(out.cpu())
            truelabel_list.append(y.cpu())
    output_log = torch.cat(output_list, dim=0)
    truelabel_log = torch.cat(truelabel_list, dim=0)
    output_log = torch.softmax(output_log , dim = 1).numpy()
    truelabel_log = truelabel_log.numpy()
    truelabel_onehot = label_binarize(truelabel_log, classes=np.arange(N_CLASS))
    roc_iter = np.zeros((N_CLASS,L_interp_roc)) # one-vs-rest ROC curves for each class
    for i in range(N_CLASS):
        output_tmp_class = output_log[:, i]
        n_pos = np.sum(output_tmp_class)
        n_tot = len(output_tmp_class) 
        n_neg = n_tot - n_pos
        sample_wt = np.zeros_like(output_tmp_class,dtype=np.float32)
        sample_wt = np.where(output_tmp_class == 1, n_tot/n_pos , n_tot/n_neg)
        fpr, tpr, _ = roc_curve(truelabel_onehot[:, i], output_log[:, i] , sample_weight=sample_wt)
        roc_iter[i] = np.interp(thresholds,fpr,tpr)
    test_acc = correct_preds / total_preds
    # Normalize the confusion matrix to get per-true-class accuracy
    test_cm_normalized = test_cm.cpu().numpy()
    row_sums = test_cm_normalized.sum(axis=1, keepdims=True)
    test_cm_normalized = np.divide(test_cm_normalized, row_sums, out=np.zeros_like(test_cm_normalized), where=row_sums!=0)
    return test_acc , test_cm_normalized , roc_iter

def test_GNN_LOO(model , loader , device):
    model.eval()
    test_acc = 0
    test_cm = torch.zeros((N_CLASS,N_CLASS), dtype=torch.float32, device=device)
    cm_metric = MulticlassConfusionMatrix(N_CLASS).to(device)
    batch_size = loader.batch_size
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for data in tqdm(loader, desc = 'TESTING GNN', leave = False):
            x = torch.cat([torch.tensor(f, dtype=torch.float32) for f in data.x], dim=0).to(device)
            edge_attr = torch.cat([torch.tensor(3 - e/1000, dtype=torch.float32) for e in data.edge_attr], dim=0).to(device)
            edge_index = torch.cat([torch.tensor(e, dtype=torch.long) for e in data.edge_index], dim=1).to(device)
            batch_id = data.batch.to(device) # batch_id indicates which graph the nodes belong to
            y = torch.tensor(data.y, dtype=torch.long).to(device)
            out = model(x, edge_index, edge_attr, batch_id)
            pred = out.argmax(dim=1)
            correct_preds += (pred == y).sum().item()
            total_preds += batch_size
            test_cm += cm_metric(out , y)
    test_acc = correct_preds / total_preds
    # Normalize the confusion matrix to get per-true-class accuracy
    test_cm_normalized = test_cm.cpu().numpy()
    row_sums = test_cm_normalized.sum(axis=1, keepdims=True)
    test_cm_normalized = np.divide(test_cm_normalized, row_sums, out=np.zeros_like(test_cm_normalized), where=row_sums!=0)            
    return test_acc , test_cm_normalized