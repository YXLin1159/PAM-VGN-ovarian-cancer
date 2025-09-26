from scipy import io as sio
import numpy as np
from tqdm import tqdm
from numba import njit
import math
import torch
from torch_geometric.data import Data
from typing import List, Tuple
from pathlib import Path

N_BV = 5
N_FEATURE = 9  # including ID and x-coordinate
FEAT_MEAN = np.array([0.0, 0.0, 210.628, -4.8416, 20.542, 63.6982, 14.9155, -0.031468, 0.963684])
FEAT_STD_INV = 1.0 / np.array([1.0, 1.0, 220.309, 28.864, 9.52464, 30.9709, 7.96389, 0.0174921, 0.429755])
LY_INCREMENT = 6  # y coordinate increment between two adjacent B scans

@njit(cache=True)
def _compute_edge_attr(edge_index: np.ndarray , x_coord: np.adarray) -> np.ndarray:
    '''
    Compute edge attributes (Euclidean distances) based on edge indices and x-coordinates of nodes.
    edge_index: 2 x N_edges numpy array
    x_coord: 1D numpy array of length N_nodes
    return: edge_attr: N_edges x 1 numpy array
    '''
    n_edges = edge_index.shape[1]
    edge_attr = np.empty((n_edges,1) , dtype = np.float32)
    for i in range(n_edges):
        node1 = edge_index[0,i]
        node2 = edge_index[1,i]
        diff_index = node1 - node2
        if diff_index < 0:
            diff_index = -diff_index
        diff_y = LY_INCREMENT*(diff_index//N_BV)
        diff_x = x_coord[node1] - x_coord[node2]
        if diff_x < 0:
            diff_x = -diff_x
        edge_attr[i,0] = math.sqrt(diff_y * diff_y + diff_x * diff_x)
    return edge_attr

@njit(cache=True)
def _generate_one_graph(D_sample: np.ndarray, idx_graph: int, N_nbh: int, edge_index: np.ndarray):
    '''
    Generate graph data (node features and edge attributes) for one sample.
    '''
    total_rows = N_BV * N_nbh
    data_tmp = np.empty((total_rows, N_FEATURE), dtype=D_sample.dtype)
    start = (idx_graph - N_nbh + 1) * N_BV
    offset_step = 2 * N_BV

    for nb in range(N_nbh):
        src = start + nb * offset_step
        dst = nb * N_BV
        data_tmp[dst:dst + N_BV, :] = D_sample[src:src + N_BV, :]
    node_feature = data_tmp[:, 2:N_FEATURE].copy()
    x_coord = data_tmp[:, 1].copy()
    edge_attr = _compute_edge_attr(edge_index, x_coord)
    return node_feature, edge_attr

def _choose_idx_graph_step(N_nbh: int) -> int:
    if 25 < N_nbh < 35:
        return 13
    if 35 < N_nbh < 55:
        return 18
    if 55 < N_nbh < 81:
        return 28
    if N_nbh > 81:
        return 41
    return 5

def process_one_sample(D_sample: np.ndarray, category_id: int, N_nbh: int) -> List[Data]:
    '''
    Process one sample's data into a list of graph data objects.
    '''
    D_sample = np.ascontiguousarray(D_sample)
    N_row, _ = D_sample.shape
    N_bv = N_BV * N_nbh
    N_bscan = int(N_row // N_BV)

    # Build edge_index
    iu, ju = np.triu_indices(N_bv, k=1)
    edge_index_np = np.vstack((iu, ju)).astype(np.int64)  # shape (2, N_edges)
    edge_index_pt = torch.tensor(edge_index_np, dtype=torch.long)

    idx_step = _choose_idx_graph_step(N_nbh)

    graphs: List[Data] = []
    # iterate over graph centers/starts preserving original index semantics
    start_idx = int(N_nbh - 1)
    stop_idx = int(N_bscan - N_nbh + 1)  # exclusive in range
    if stop_idx <= start_idx:
        return graphs # not enough B-scans to form a graph

    for idx_graph in range(start_idx, stop_idx, idx_step):
        node_feat_np, edge_attr_np = _generate_one_graph(D_sample, idx_graph, N_nbh, edge_index_np)
        x = torch.tensor(np.ascontiguousarray(node_feat_np), dtype=torch.float32)
        edge_attr = torch.tensor(np.ascontiguousarray(edge_attr_np), dtype=torch.float32)
        y = torch.tensor([int(category_id)], dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index_pt, edge_attr=edge_attr, y=y)
        graphs.append(graph)

    return graphs

def _normalize_graph_data(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    return (data - FEAT_MEAN[np.newaxis, :]) * FEAT_STD_INV[np.newaxis, :]

def processMAT(train_data_filepath_list , train_label_list , N_nbh):
    '''
    Process all training samples from .mat files into a list of graph data objects.
    '''
    graph_all = []
    N_sample = len(train_label_list)
    for idx_sample in tqdm(range(N_sample) , desc = 'CREATING GRAPH DATA', leave = False):
        data_tmp = sio.loadmat(train_data_filepath_list[idx_sample])
        data_tmp = data_tmp['graph_data']
        data_tmp = _normalize_graph_data(data_tmp)
        graph = process_one_sample(data_tmp , train_label_list[idx_sample] , N_nbh)
        graph_all.append(graph)
    return graph_all

def split_data(main_folder_path: str , val_frac: float = 0.2 , test_frac: float = 0.2 , min_val: int = 2 , min_test: int = 1) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    root = Path(main_folder_path)
    if not root.is_dir():
        raise ValueError(f"Provided path {main_folder_path} is not a valid directory.")
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {main_folder_path}")

    train_files: List[str] = []
    val_files: List[str]   = []
    test_files: List[str]  = []
    train_labels: List[int] = []
    val_labels: List[int]   = []
    test_labels: List[int]  = []

    for class_idx, class_dir in enumerate(class_dirs):
        files = sorted([str(p) for p in class_dir.iterdir() if p.is_file()])
        n_files = len(files)
        if n_files == 0:
            continue

        n_val = max(int(round(n_files * val_frac)), min_val)
        n_test = max(int(round(n_files * test_frac)), min_test)

        n_train = n_files - n_val - n_test
        if n_train < 0:
            n_val = min(n_val, n_files - 1)
            n_test = min(n_test, n_files - 1 - n_val)
            n_train = n_files - n_val - n_test
            if n_train <= 0:
                n_train = 1
                remaining = n_files - 1
                n_val = remaining // 2
                n_test = remaining - n_val

        perm = rng.permutation(n_files)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]

        for i in train_idx:
            train_files.append(files[int(i)])
            train_labels.append(class_idx)
        for i in val_idx:
            val_files.append(files[int(i)])
            val_labels.append(class_idx)
        for i in test_idx:
            test_files.append(files[int(i)])
            test_labels.append(class_idx)

    train_labels_arr = np.array(train_labels, dtype = np.int32)
    val_labels_arr   = np.array(val_labels, dtype = np.int32)
    test_labels_arr  = np.array(test_labels, dtype = np.int32)
    return train_files, val_files, test_files, train_labels_arr, val_labels_arr, test_labels_arr