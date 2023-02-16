### Introduction
Graph neural networks are widely used in the field of deep learning. Now we discuss the feasibility of extending the GNN algorithms to the distributed big data and AI tool orca based on the popular GNN library PyG.

[PyG](https://www.pyg.org/)(PyTorch Geometric) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. And [Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html) is a distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray.

### Feasibility Analysis
PyG is characterized by formalizing the data structure for graph and implementing the popular GNN algorithms. Generally, PyG defines a class [torch_geometric.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) to store the graph and pass it to [torch_geometric.Dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) which is inherited from `pytorch dataset`, and then pass the dataset to [torch_geometric.Dataloader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html) which is inherited from `pytorch dataloader`. Then the dataloader is passed to any GNN algorithm implemented in [torch_geometric.nn](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) to train the network and update the graph.

There are many GNN-algorithms classes implemented in `torch_geometric.nn` such as [GCNConv](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gcn_conv.py) and they are inherited from `torch.nn.Module`. 

So it's feasible to scale codes based on PyG to Orca. Take the [PyG tutorial example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#) as an example to show how to scale the code to orca:
- init orca context
```
init_orca_context()
```
- define the graph and the dataloader
We can define the graph using `torch_geometric.Data`.
```
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
y = torch.tensor([[1], [2], [1]], dtype=torch.float)
data1 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
data2 = Data(...)
...
datan = Data(...)

data_list = [data1, data2, ..., datan]
```
Then define the dataset using `torch_geometric.Dataset`. (See more about customizing the dataset from [here.](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html))
```
from torch_geometric.data import Dataset

class MyOwnDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
```
Then create the dataloader using `pytorch dataloader`.
```
from torch.utils.data import DataLoader

def train_loader_func(config, batch_size):
    train_dataset = MyOwnDataset(data_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    return train_loader
```
- define the Graph Neural Network
Then we define our own GNN class `GCN` consisting of `torch_geometric.nn.GCNConv` and some non-linear layers. (See more about how to customize GNN from [here.](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html))
```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```
```
def model_creator(config):
    model = GCN()
    model.train()
    return model

def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=0.01)
```
- train the network using orca
Then train the network distributedly using orca.
```
est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           loss=nn.BCEWithLogitsLoss(),
                           metrics=[Accuracy()],
                           backend="ray")
train_stats = est.fit(train_loader_func,
                      epochs=2,
                      batch_size=256)
```
- predict the graph
```
model = est.get_model()
result = model(test_data)
```
- stop orca context
```
stop_orca_context()
```

Besides, PyG also provides the tutorial of [loading graph from csv](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html) using [MovieLens dataset](https://grouplens.org/datasets/movielens/). So it's convinent to write a tutorial of "scaling GNN to orca based on PyG" using the dataset `ml-1m` to maintain the consistency of the dataset with [NCF tutorial](https://github.com/intel-analytics/BigDL/tree/main/python/orca/tutorial/NCF).


### Links
[PyG](https://www.pyg.org/)

[PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

[PyG source code](https://github.com/pyg-team/pytorch_geometric)

[orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html)















