### Introduction: TF-GNN distributed training (without Orca)
The codes is from [TF-GNN](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md)'s node classification [tutorial](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb).

The notebook tutorial on `colab` can run successfully. However, it failed when I run the code on my own cluster containing two machine nodes(The python file `solving_ogbn_mag_end_to_end_with_tf_gnn.py` is converted from the notebook on `colab`). The reason is the code was built on the latest version of `TF-GNN-0.5.0` which is dependent on [bazel](https://docs.bazel.build/versions/2.0.0/updating-bazel.html), but I failed to build `bazel` when following [the instruction](https://github.com/tensorflow/gnn/tree/main).

### Reproduction
1. Login in the master node of BigDL's cluster
```
ssh kai@Almaren-Node-107  # password: 1234qwer
```

2. Prepare the environment

(Follow [the instruction](https://github.com/tensorflow/gnn/tree/main) to install TF-GNN.)

- Prepare conda environment:
```
conda create -n py38 python=3.8
pip install tensorflow==2.9.1 tqdm 
```
- Install `bazel`(failed):

Following [here](https://docs.bazel.build/versions/2.2.0/install-redhat.html).

 - Install `gnn-0.5.0`:
```
su kai
conda activate py38
wget https://github.com/tensorflow/gnn/archive/refs/tags/v0.5.0.zip
unzip v0.5.0.zip
cd gnn-0.5.0
pip install .
```

3. Execute the python script to train GNN distributedly(failed)

(Follow the node classification [tutorial](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb).)

Run the command on the master node:

`python solving_ogbn_mag_end_to_end_with_tf_gnn.py`

