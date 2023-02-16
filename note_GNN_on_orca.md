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
- define the graph 
We can define the graph using `torch_geometric.Data` and create a dataloader.
```
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
```
- define the Graph Neural Network
Then we define our GNN inherited from `torch_geometric.nn`.
```

```
- train the network using orca
Then train the network distributedly using orca.
```

```
- stop orca context
```
stop_orca_context()
```

Besides, PyG also provides the tutorial of [loading graph from csv](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html) using [MovieLens dataset](https://grouplens.org/datasets/movielens/). So it's convinent to write a tutorial of "scaling GNN to orca based on PyG" using the dataset `ml-1m` to maintain the consistency of the dataset with `NCF tutorial`.


### Links
[PyG](https://www.pyg.org/)

[PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

[PyG source code](https://github.com/pyg-team/pytorch_geometric)

[orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/index.html)















