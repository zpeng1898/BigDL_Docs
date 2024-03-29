### 1. Libraries for basic GNN
- PyG(PyTorch Geometric)

[PyG.org](https://www.pyg.org/)

[gcn-node-classification](https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html)
- Keras examples

[Keras MPNN example](https://keras.io/examples/graph/mpnn-molecular-graphs/)

[Keras Node Classification example](https://keras.io/examples/graph/gnn_citations/)
- tensorflow examples

[tensorflow GNN](https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html)
- dgl examples: tensorflow gcn

[dgl examples: tensorflow gcn](https://github.com/dmlc/dgl/tree/master/examples/tensorflow/gcn)

- StellarGraph

[StellarGraph](https://stellargraph.readthedocs.io/en/stable/README.html)

### 2. Libraries for distributed GNN using data parallelism
- graphx & graphframes *

[graphx](https://spark.apache.org/graphx/)

[graphx-programming-guide](https://spark.apache.org/docs/latest/graphx-programming-guide.html)

[graphframes](https://graphframes.github.io/graphframes/docs/_site/index.html)

[graphframes-subgraphs](https://graphframes.github.io/graphframes/docs/_site/user-guide.html#subgraphs)

- dgl *

[dgl docs * ](https://docs.dgl.ai/)

[dgl distributed docs * ](https://docs.dgl.ai/en/latest/guide/distributed.html)

[7.2 distributed APIs](https://docs.dgl.ai/en/latest/guide_cn/distributed-apis.html#guide-cn-distributed-apis)

[7.4 Advanced Graph Partitioning](https://docs.dgl.ai/en/latest/guide/distributed-partition.html#guide-distributed-partition)

[tutorial-Distributed Node Classification * ](https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py)

[tutorial-Distributed Link Prediction](https://docs.dgl.ai/en/latest/tutorials/dist/2_link_prediction.html)

[dgl source code github](https://github.com/dmlc/dgl)

- TF-GNN

[github * ](https://github.com/tensorflow/gnn/tree/main)

[docs * ](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md)

[node classification tutorial * ](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb)

- graphlearn-for-pytorch 

[graphlearn-for-pytorch](https://github.com/alibaba/graphlearn-for-pytorch)

[GraphScope](https://github.com/alibaba/GraphScope)

- PyGAS

[PyGAS: Auto-Scaling GNNs in PyG](https://github.com/rusty1s/pyg_autoscale)

- METIS (Graph Partitioning toolkit)

[METIS - Serial Graph Partitioning and Fill-reducing Matrix Ordering](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) 

- keras distributed deep learning
[distribute keras](https://www.tensorflow.org/tutorials/distribute/keras)

### 3. Other links for distributed GNN
1. github notes

[awesome-gnn-systems](https://github.com/chwan1016/awesome-gnn-systems#distributed-gnn-training-systems)

2. libraries
- euler

[euler](https://github.com/alibaba/euler)

[Euler-2.0-Message-Passing](https://github.com/alibaba/euler/wiki/Euler-2.0-Message-Passing%E6%8E%A5%E5%8F%A3)

[Euler 2.0 在图分类上的应用](https://github.com/alibaba/euler/wiki/Euler-2.0-%E5%9C%A8%E5%9B%BE%E5%88%86%E7%B1%BB%E4%B8%8A%E7%9A%84%E5%BA%94%E7%94%A8)
- NeutronStarLite(supports CPU-GPU heterogeneous computation on multiple workers)

[NeutronStarLite](https://github.com/iDC-NEU/NeutronStarLite)
- quiver(based on PyG)

[quiver](https://github.com/quiver-team/torch-quiver)

[quiver-feature](https://github.com/quiver-team/quiver-feature)

3. papers
- cluster_GCN (algorithm for training large GCN)

[cluster_GCN](https://github.com/zhengjingwei/cluster_GCN)
- GSplit (algorithm for distributed GNN using split-parallelism)

[GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism](https://arxiv.org/abs/2303.13775)
- papers of graph-partitioning

[graph-partitioning paperswithcode](https://paperswithcode.com/task/graph-partitioning/codeless)

4. previous notes and experiments on combining distributed GNN with orca

### 4. Experiments on combining distributed GNN libraries with orca

- PyG

Combining [PyG](https://www.pyg.org/) with Orca. See directory `./PyG` for more notes and codes.

PyG only supports PyTorch.

- DGL

Combining [DGL](https://docs.dgl.ai/) with Orca. See directory `./DGL` for more notes and codes.

DGL supports training GNN locally on PyTorch and TensorFlow, but supports distributed training only on PyTorch.

`./DGL/codes/dgl_distributed_training` trains GNN on 2 nodes distributedly using `DGL` and `tf.distributed.strategy` without `Orca`. 
`./DGL/codes/combing_dgl_with_orca` trains GNN on 1 node, using `DGL` to get a whole graph with only one part, and training it on `Orca` with `local` mode.

- TF-GNN

Combining [TF-GNN](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md) with Orca. See directory `./TF-GNN` for more notes and codes.

TF-GNN supports GNN distributed training based on TensorFlow.

TODO: 
1. Run TF-GNN distributed training example without Orca. The notebook tutorial [node classification](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb) on `colab` can run successfully. However, it failed when I run the code on my own cluster. The reason is the code was built on the latest version of `TF-GNN-0.5.0` which is dependent on [bazel](https://docs.bazel.build/versions/2.0.0/updating-bazel.html), but I failed to build `bazel` when following [the instruction](https://github.com/tensorflow/gnn/tree/main). See `./TF-GNN/codes/TF-GNN_distributed_training` for more. 
2. Combine TF-GNN with Orca.





