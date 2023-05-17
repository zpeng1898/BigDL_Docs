# DGL distributed training example on node classsification: 
# https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py
# https://docs.dgl.ai/en/latest/tutorials/large/L1_large_node_classification.html#sphx-glr-tutorials-large-l1-large-node-classification-py
# https://github.com/dmlc/dgl/blob/master/python/dgl/distributed/dist_dataloader.py

import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch as th


## Prepare Datasets
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
train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
train_mask[train_nid] = True
val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
val_mask[val_nid] = True
test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
test_mask[test_nid] = True
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask


## Graph Partitioning
dgl.distributed.partition_graph(graph, graph_name='ogbn-arxiv',
                                             num_parts=1,
                                             out_path='4part_data',
                                             balance_ntypes=graph.ndata['train_mask'],
                                             balance_edges=True)

## distribued training script
dgl.distributed.initialize(ip_config='ip_config.txt')
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = '29500'

g = dgl.distributed.DistGraph('ogbn-arxiv', part_config="4part_data/ogbn-arxiv.json")
train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])

sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
train_dataloader = dgl.dataloading.DistNodeDataLoader(
                             g, train_nid, sampler, batch_size=1024,
                             shuffle=True, drop_last=False)
valid_dataloader = dgl.dataloading.DistNodeDataLoader(
                             g, valid_nid, sampler, batch_size=1024,
                             shuffle=False, drop_last=False)


## Define GNN Model
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.optim as optim

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

    def forward(self, blocks, x):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x

num_hidden = 256
num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))
num_layers = 2
lr = 0.001
model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)
loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

th.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
model = th.nn.parallel.DistributedDataParallel(model)


## Distributed training
import sklearn.metrics
import numpy as np

for epoch in range(3):
    # Loop over the dataloader to sample mini-batches.
    losses = []
    with model.join():
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # Load the input features as well as output labels
            batch_inputs = g.ndata['feat'][input_nodes]
            batch_labels = g.ndata['labels'][seeds]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

    # validation
    predictions = []
    labels = []
    with th.no_grad(), model.join():
        for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
            inputs = g.ndata['feat'][input_nodes]
            labels.append(g.ndata['labels'][seeds].numpy())
            predictions.append(model(blocks, inputs).argmax(1).numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {}: Validation Accuracy {}'.format(epoch, accuracy))


# Results:
# WARNING:root:The OGB package is out of date. Your version is 1.3.5, while the latest version is 1.3.6.                  
# Graph(num_nodes=169343, num_edges=2332486,                                                                                    
#     ndata_schemes={'year': Scheme(shape=(1,), dtype=torch.int64), 
#     'feat': Scheme(shape=(128,), dtype=torch.float32), 
#     'labels': Scheme(shape=(), dtype=torch.int64)}                                                                                 
#     edata_schemes={})                                                                                                 
# tensor([ 4,  5, 28,  ..., 10,  4,  1])                                                                                  
# Number of classes: 40                                                                                                   
# Converting to homogeneous graph takes 0.033s, peak mem: 3.207 GB                                                        
# Save partitions: 0.366 seconds, peak memory: 3.368 GB                                                                   
# There are 2332486 edges in the graph and 0 edge cuts for 1 partitions.                                                  
# Start to load partition from 4part_data/part0/graph.dgl which is 69848781 bytes. It may take non-trivial time for large partition.                                                                                                              
# Finished loading partition.                                                                                             
# Start to load node data from 4part_data/part0/node_feat.dgl which is 89921568 bytes.                                    
# Finished loading node data.                                                                                             
# Start to load edge data from 4part_data/part0/edge_feat.dgl which is 24 bytes.                                          
# Finished loading edge data.                                                                                             
# Epoch 0: Validation Accuracy 0.6043826974059532                                                                         
# Epoch 1: Validation Accuracy 0.6410282224235713                                                                         
# Epoch 2: Validation Accuracy 0.6562636330078191 
