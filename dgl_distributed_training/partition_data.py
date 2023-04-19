import os

import torch

import dgl
from ogb.nodeproppred import DglNodePropPredDataset

data = DglNodePropPredDataset(name='ogbn-arxiv')
graph, labels = data[0]
graph = dgl.add_reverse_edges(graph)
labels = labels[:, 0]
graph.ndata['labels'] = labels
print(graph)
print(labels)
node_features = graph.ndata["feat"]
num_features = node_features.shape[1]
num_classes = (labels.max() + 1).item()
print("Number of classes:", num_classes)

splitted_idx = data.get_idx_split()
train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
train_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
train_mask[train_nid] = True
val_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
val_mask[val_nid] = True
test_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
test_mask[test_nid] = True
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask
print(len(graph.ndata['train_mask']))
# graph partitioning
dgl.distributed.partition_graph(graph, graph_name='ogbn-arxiv',
                                             num_parts=2,
                                             out_path='4part_data',
                                             balance_ntypes=graph.ndata['train_mask'],
                                             balance_edges=True)