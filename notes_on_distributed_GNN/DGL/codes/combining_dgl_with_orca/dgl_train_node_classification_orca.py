#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------
# DGL distributed training example on node classsification: 
# https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py
# https://docs.dgl.ai/en/latest/tutorials/large/L1_large_node_classification.html#sphx-glr-tutorials-large-l1-large-node-classification-py

# Step 0: Import necessary libraries
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.dataloading import DistNodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

os.environ['DGLBACKEND'] = 'pytorch'
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = '29500'

# Step 1: Init Orca Context
sc = init_orca_context(cluster_mode="local")


# Step 2: Define train and test datasets as DataLoader
def load_dataset():
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

    # graph partitioning
    dgl.distributed.partition_graph(graph, graph_name='ogbn-arxiv',
                                                 num_parts=1,
                                                 out_path='4part_data',
                                                 balance_ntypes=graph.ndata['train_mask'],
                                                 balance_edges=True)

    ## distribued training script
    dgl.distributed.initialize(ip_config='ip_config.txt')


    g = dgl.distributed.DistGraph('ogbn-arxiv', part_config="4part_data/ogbn-arxiv.json")
    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])

    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 10])
    return g, sampler, train_nid, valid_nid


class new_DistNodeDataLoader(DistNodeDataLoader):
    def __init__(self, g, train_nid, sampler, batch_size=1024,
                        shuffle=True, drop_last=False):
        super().__init__(g, train_nid, sampler, batch_size=batch_size,
                         shuffle=shuffle, drop_last=drop_last)
        self.g = g

    def __len__(self):
        return int(len(self.dataset) / self.batch_size) + 1  # number of batches

    def __next__(self):
        if self.pool is None:
            num_reqs = 1
        else:
            num_reqs = self.queue_size - self.num_pending
        for _ in range(num_reqs):
            self._request_next_batch()
        if self.recv_idxs < self.expected_idxs:
            node_in_out, blocks = self._get_data_from_result_queue()
            input_nodes, seeds = node_in_out[0], node_in_out[1]
            self.recv_idxs += 1
            self.num_pending -= 1
            batch_inputs = self.g.ndata['feat'][input_nodes]
            batch_labels = self.g.ndata['labels'][seeds]
            result = ((batch_labels, blocks, batch_inputs), batch_labels)
            return result
        else:
            assert self.num_pending == 0
            raise StopIteration


def train_loader_func(config, batch_size):
    g, sampler, train_nid, _ = load_dataset()
    train_loader = new_DistNodeDataLoader(
                        g, train_nid, sampler, batch_size=batch_size,
                        shuffle=True, drop_last=False)
    return train_loader


def test_loader_func(config, batch_size):
    g, sampler, _, valid_nid = load_dataset()
    test_loader = new_DistNodeDataLoader(
                        g, valid_nid, sampler, batch_size=batch_size,
                        shuffle=False, drop_last=False)
    return test_loader


# Step 3: Define the model, optimizer and loss
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

    def forward(self, labels, blocks, x):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x


def model_creator(config):
    g, _, _, _ = load_dataset()
    model = SAGE(g.ndata['feat'].shape[1],
                 config["num_hidden"],
                 len(torch.unique(g.ndata['labels'][0:g.num_nodes()])),
                 config["num_layers"])
    model.train()
    return model


def optimizer_creator(model, config):
    return torch.optim.Adam(model.parameters(), lr=config["lr"])


# Step 4: Distributed training with Orca PyTorch Estimator
config = dict(
    num_hidden=256,
    num_layers=2,
    lr=0.001,
)

est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=nn.CrossEntropyLoss(),
                           metrics=[Accuracy()],
                           config=config,
                           backend="ray",
                           use_tqdm=True,
                           workers_per_node=1)
train_stats = est.fit(train_loader_func,
                      epochs=10,
                      batch_size=1024,
                      validation_data=test_loader_func)
print("Train results:")
for epoch_stats in train_stats:
    for k, v in epoch_stats.items():
        print("{}: {}".format(k, v))
    print()


# Step 5: Distributed evaluation of the trained model
eval_stats = est.evaluate(test_loader_func, batch_size=1024)
print("Evaluation results:")
for k, v in eval_stats.items():
    print("{}: {}".format(k, v))


# Step 7: Save the trained PyTorch model
est.save(os.path.join("./GNN_model"))


# Step 8: Shutdown the Estimator and stop Orca Context when the program finishes
est.shutdown()
stop_orca_context()
