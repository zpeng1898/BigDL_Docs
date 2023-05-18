### Node Classification Task - distributed training on DGL
`dgl_train_node_classification.py` trains GNN on 1 node without Orca. The code is adapted from [here](https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py)
 
- Run command `python partition_data.py` to generate a directory `1part_data` containing one part graph.
- Run training file locally `python dgl_train_node_classification.py`.

### Node Classification Task - distributed training on DGL and Orca
`dgl_train_node_classification_orca.py` trains GNN on 1 node, using DGL to get a whole graph with only one part, and training it on Orca with local mode.

- Run command `python partition_data.py` to generate a directory `1part_data` containing one part graph.
- Run training file locally `python dgl_train_node_classification_orca.py`.
- Results of `dgl_train_node_classification_orca.py`
![image](https://user-images.githubusercontent.com/42887453/231736686-3f5a3145-199f-40ad-9e2e-f794d3ed7915.png)
![image](https://user-images.githubusercontent.com/42887453/231736862-a8731079-9e69-456d-b2de-76b195075386.png)
![image](https://user-images.githubusercontent.com/42887453/231736920-3a46f0d8-f0a6-4d5b-91e4-1b50339462ef.png)
